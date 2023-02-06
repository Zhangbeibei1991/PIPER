import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from rule_pools import tbd, matres
from graph import GatedGCN, DistillModule


class SymmetryLoss(nn.Module):
    """
    alpha <-> beta, where beta is the opposite to alpha
    """

    def __init__(self):
        super(SymmetryLoss, self).__init__()

    def forward(self, alpha, beta, alpha_idx, beta_idx):
        return torch.abs(alpha[:, alpha_idx] - beta[:, beta_idx])
        # return torch.abs(alpha[alpha_idx] - beta[beta_idx])


class TransitivityLossYes(nn.Module):
    """
    alpha ^ beta -> gamma, where gamma is not conflict
    """

    def __init__(self, device):
        super(TransitivityLossYes, self).__init__()
        self.zero = torch.tensor(0, dtype=torch.float, requires_grad=False).to(device)

    def forward(self, alpha, beta, gamma, alpha_idx, beta_idx, gamma_idx):
        conj_loss_pos = torch.max(self.zero, alpha[:, alpha_idx] + beta[:, beta_idx] - gamma[:, gamma_idx])
        conj_loss = conj_loss_pos
        return conj_loss


class ConjunctiveNot(nn.Module):
    """
    alpha ^ beta -> not delta, where delta is conflict
    """

    def __init__(self, device):
        super(ConjunctiveNot, self).__init__()
        self.zero = torch.tensor(0, dtype=torch.float, requires_grad=False).to(device)
        self.one = torch.tensor(1, dtype=torch.float, requires_grad=False).to(device)

    def forward(self, alpha, beta, gamma, alpha_idx, beta_idx, gamma_idx):
        very_small = 1e-8
        not_gamma = (self.one - gamma.exp()).clamp(very_small).log()
        return torch.max(self.zero, alpha[:, alpha_idx] + beta[:, beta_idx] - not_gamma[:, gamma_idx])


class PIPERModel(nn.Module):
    def __init__(self, args, num_classes, word_embed_table, lab2id, rev_map, device):
        super(PIPERModel, self).__init__()
        self.args = args
        self.lab2id = lab2id
        self.id2lab = {v: k for k, v in self.lab2id.items()}
        self.device = device
        self.rev_map = rev_map
        self.feat_list = self.args.feat_list
        self.num_classes = num_classes
        self.word_embed = nn.Embedding.from_pretrained(embeddings=torch.FloatTensor(word_embed_table),
                                                       freeze=True)  # TB-Dense: True
        self.tag_embed = nn.Embedding(num_embeddings=1000, embedding_dim=self.args.tag_dim)

        self.encoder = AutoModel.from_pretrained(self.args.bert_path)
        self.config = AutoConfig.from_pretrained(self.args.bert_path)

        input_dim = self.args.embed_dim + self.args.tag_dim + self.config.hidden_size
        hidden_dim = self.config.hidden_size
        # hidden_dim = input_dim
        if self.args.encoder == "BiGRU":
            self.BiEncoder = nn.GRU(input_size=input_dim, hidden_size=hidden_dim // 2, bidirectional=True)
        elif self.args.encoder == "BiLSTM":
            self.BiEncoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim // 2, bidirectional=True)
        self.dropout = nn.Dropout(self.args.drop_rate)
        if self.args.feat_pair == "GRU":
            self.pair_encoder = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, bidirectional=False)
        elif self.args.feat_pair == "LSTM":
            self.pair_encoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, bidirectional=False)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)
        if self.args.task_name == "TB-Dense":
            self.rule_pools = tbd
        elif self.args.task_name == "MATRES":
            self.rule_pools = matres

        # gcn
        self.num_layers = self.args.num_layers
        self.gcn_layers = nn.ModuleList(
            [GatedGCN(hidden_size=hidden_dim, activation_ru=self.args.activation_ru,
                      activation_x=self.args.activation_x) for _ in range(self.num_layers)])
        self.dm_layers = nn.ModuleList(
            [DistillModule(hidden_dim=hidden_dim, para_attention=False) for _ in range(self.num_layers)])
        self.aggregate = nn.Linear(hidden_dim * (self.num_layers + 1), hidden_dim)

        n_repeat = 0
        for x in self.feat_list:
            if x in ["avg", "times", "pair_encode", "minus"]:
                n_repeat += 1
            elif x in ["concat"]:
                n_repeat += 2
            else:
                raise ValueError("UnKnown feature !")

        self.hidden = nn.Linear(hidden_dim * n_repeat, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)
        # self.classifier = nn.Linear(hidden_dim * 2, num_classes)

        self.sym_loss = SymmetryLoss()
        self.conj_loss_yes = TransitivityLossYes(device=self.device)
        self.conj_loss_not = TransitivityLossYes(device=self.device)
        self.cross_backbone = nn.CrossEntropyLoss()

    def forward(self, batch, mode="train", teacher_logit=None):
        ab_info_tensor, bc_info_tensor, ac_info_tensor, info = batch
        if self.args.stage == "backbone":
            ab_logit, ab_lab, ab_dis_ns = self.context_embeddings(info_tensor=ab_info_tensor, e1=0, e2=1)
            if mode == "train":
                ab_loss = self.cross_backbone(ab_logit, ab_lab).mean() + self.args.dis_wt * ab_dis_ns
                return ab_loss
            else:
                return ab_logit
        elif self.args.stage == "symmetry":
            ab_logit, ab_lab, ab_dis_ns = self.context_embeddings(info_tensor=ab_info_tensor, e1=0, e2=1)
            if mode == "train":
                ba_logit, ba_lab, ba_dis_ns = self.context_embeddings(info_tensor=ab_info_tensor, e1=1, e2=0)
                ab_loss = self.cross_backbone(ab_logit, ab_lab) + self.args.dis_wt * ab_dis_ns
                ba_loss = self.cross_backbone(ba_logit, ba_lab) + self.args.dis_wt * ba_dis_ns
                sym_loss = self.compute_sym(alpha=ab_logit, beta=ba_logit)
                batch_loss = (ab_loss + ba_loss) / 2.0 + self.args.sym_wt * sym_loss
                return batch_loss
            else:
                return ab_logit
        elif self.args.stage == "transitivity":
            ab_logit, ab_lab, ab_dis_ns = self.context_embeddings(info_tensor=ab_info_tensor, e1=0, e2=1)
            if mode == "train":
                bc_logit, bc_lab, bc_dis_ns = self.context_embeddings(info_tensor=bc_info_tensor, e1=0, e2=1)
                ac_logit, ac_lab, ac_dis_ns = self.context_embeddings(info_tensor=ac_info_tensor, e1=0, e2=1)
                ab_loss = self.cross_backbone(ab_logit, ab_lab) + self.args.dis_wt * ab_dis_ns
                bc_loss = self.cross_backbone(bc_logit, bc_lab) + self.args.dis_wt * bc_dis_ns
                ac_loss = self.cross_backbone(ac_logit, ac_lab) + self.args.dis_wt * ac_dis_ns
                trans_loss = self.compute_trans(alpha=F.log_softmax(ab_logit, dim=-1),
                                                beta=F.log_softmax(bc_logit, dim=-1),
                                                gamma=F.log_softmax(ac_logit, dim=-1))
                batch_loss = (ab_loss + bc_loss + ac_loss) / 3.0 + self.args.trans_wt * trans_loss
                return batch_loss
            else:
                ba_logit, ab_lab, ab_dis_ns = self.context_embeddings(info_tensor=ab_info_tensor, e1=1, e2=0)
                return ab_logit, ba_logit
        elif self.args.stage == "sym+trans":
            ab_logit, ab_lab, ab_dis_ns = self.context_embeddings(info_tensor=ab_info_tensor, e1=0, e2=1)
            if mode == "train":
                ba_logit, ba_lab, ba_dis_ns = self.context_embeddings(info_tensor=ab_info_tensor, e1=1, e2=0)
                bc_logit, bc_lab, bc_dis_ns = self.context_embeddings(info_tensor=bc_info_tensor, e1=0, e2=1)
                cb_logit, cb_lab, cb_dis_ns = self.context_embeddings(info_tensor=bc_info_tensor, e1=1, e2=0)
                ac_logit, ac_lab, ac_dis_ns = self.context_embeddings(info_tensor=ac_info_tensor, e1=0, e2=1)
                ca_logit, ca_lab, ca_dis_ns = self.context_embeddings(info_tensor=ac_info_tensor, e1=1, e2=0)

                ab_loss = self.cross_backbone(ab_logit, ab_lab) + self.args.dis_wt * ab_dis_ns
                ba_loss = self.cross_backbone(ba_logit, ba_lab) + self.args.dis_wt * ba_dis_ns
                bc_loss = self.cross_backbone(bc_logit, bc_lab) + self.args.dis_wt * bc_dis_ns
                cb_loss = self.cross_backbone(cb_logit, cb_lab) + self.args.dis_wt * cb_dis_ns
                ac_loss = self.cross_backbone(ac_logit, ac_lab) + self.args.dis_wt * ac_dis_ns
                ca_loss = self.cross_backbone(ca_logit, ca_lab) + self.args.dis_wt * ca_dis_ns

                ab_sym_loss = self.compute_sym(alpha=ab_logit, beta=ba_logit)
                bc_sym_loss = self.compute_sym(alpha=bc_logit, beta=cb_logit)
                ac_sym_loss = self.compute_sym(alpha=ac_logit, beta=ca_logit)

                trans_loss = self.compute_trans(alpha=F.log_softmax(ab_logit, dim=-1),
                                                beta=F.log_softmax(bc_logit, dim=-1),
                                                gamma=F.log_softmax(ac_logit, dim=-1))
                batch_loss = (ab_loss + ba_loss + bc_loss + cb_loss + ac_loss + ca_loss) / 6.0 + self.args.sym_wt * (
                        ab_sym_loss + bc_sym_loss + ac_sym_loss) / 3.0 + self.args.trans_wt * trans_loss
                return batch_loss
            else:
                ba_logit, ab_lab, ab_dis_ns = self.context_embeddings(info_tensor=ab_info_tensor, e1=1, e2=0)
                return ab_logit, ba_logit
        else:
            raise ("error stage !")

    def gnn_nodule(self, adj, out, mask):
        adj = adj.to(torch.bool).to(torch.float)
        gcn_inputs = out
        gcn_outputs = gcn_inputs
        layer_list = [gcn_inputs]

        for _, layer in enumerate(self.gcn_layers):
            gcn_outputs = layer(gcn_outputs, adj)
            gcn_outputs = self.dropout(gcn_outputs)
            layer_list.append(gcn_outputs)

        dm_mask = mask.to(torch.float)
        ns = torch.tensor(0, device=dm_mask.device)
        for i in range(self.num_layers):
            o1, o2, n1, n2 = self.dm_layers[i](layer_list[i], layer_list[i + 1], dm_mask, dm_mask)
            ns = ns + n1 + n2
            layer_list[i] = o1
            layer_list[i + 1] = o2
        ns = ns / (2 * self.num_layers)

        out = self.layer_norm(self.dropout(self.aggregate(torch.cat(layer_list, dim=-1))) + out)
        return out, ns

    def context_embeddings(self, info_tensor, e1=0, e2=1):
        input_ids, word_ids, tag_ids, offsets, adj, spans, seq_lens, labs, rev_labs, tasks, flags = info_tensor
        mask_ids = (input_ids > 0).long()
        # last_hidden_state = self.encoder(input_ids, mask_ids).last_hidden_state
        last_hidden_state = self.encoder(input_ids, mask_ids)[0]

        batch_size = last_hidden_state.shape[0]
        range_vector = torch.LongTensor(batch_size).to(last_hidden_state.device).fill_(1).cumsum(0) - 1
        bert_encode = last_hidden_state[range_vector.unsqueeze(1), offsets]
        bert_encode = self.dropout(bert_encode)

        word_encode = self.word_embed(word_ids)
        tag_encode = self.tag_embed(tag_ids)

        token_encode = torch.cat([bert_encode, word_encode, tag_encode], dim=-1)

        gru_context = self.dropout(self.combine(token_encode=token_encode, seq_lens=seq_lens))

        out_encode = self.layer_norm(gru_context + bert_encode)
        # out_encode = gru_context

        token_mask = (offsets > 0).float()
        out_encode, dis_ns = self.gnn_nodule(adj=adj.float(), out=out_encode, mask=token_mask)

        spans_list = spans.tolist()

        # reform the features
        e1_encode = []
        for i in range(batch_size):
            e1_encode.append(out_encode[i][spans_list[i][e1][0]: spans_list[i][e1][1]].max(dim=0)[0])
        e1_encode = torch.stack(e1_encode, dim=0)

        e2_encode = []
        for i in range(batch_size):
            e2_encode.append(out_encode[i][spans_list[i][e2][0]: spans_list[i][e2][1]].max(dim=0)[0])
        e2_encode = torch.stack(e2_encode, dim=0)  # (batch_size, dim)

        features = list()
        if "times" in self.feat_list:
            features.append(e1_encode * e2_encode)  # commutative feat: times
        if "avg" in self.feat_list:
            features.append((e1_encode + e2_encode) / 2)  # commutative feat: avg

        if "pair_encode" in self.feat_list:
            e1_e2_hidden = torch.stack([e1_encode, e2_encode], dim=1)  # (batch_size, 2, dim)
            e1_e2_encoded = self.pair_encoder(e1_e2_hidden)[0]  # (batch_size, 2, dim)
            features.append(e1_e2_encoded[:, -1, :])  # non-commutative feat: pair_encode
        if "concat" in self.feat_list:
            features.append(torch.cat([e1_encode, e2_encode], dim=-1))  # non-commutative feat: concat

        if "minus" in self.feat_list:
            features.append(e1_encode - e2_encode)  # non-commutative feat: minus

        features = torch.cat(features, dim=-1)
        # state = torch.cat([e1_encode, e2_encode], dim=-1)

        # state = self.dropout(self.hidden(features) + self.state_init[:features.size(0), :])
        state = self.dropout(self.hidden(features))
        logits = self.classifier(state)  # (batch_size, num_classes)

        if e1 == 1:
            labs = rev_labs

        return logits, labs, dis_ns
        # return logits, labs, 0.0

    def combine(self, token_encode, seq_lens):
        # 加入BiLSTM获得上下文编码
        token_encode_pack = torch.nn.utils.rnn.pack_padded_sequence(token_encode, seq_lens.cpu(), batch_first=True,
                                                                    enforce_sorted=False)
        gru_encode, _ = self.BiEncoder(token_encode_pack)
        gru_encode_unpack, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_encode, batch_first=True)
        return gru_encode_unpack

    def compute_sym(self, alpha, beta):
        sym_loss = torch.tensor(0.0, dtype=torch.float, requires_grad=False).to(self.device)
        for label, idx in self.lab2id.items():
            alpha_label = label
            beta_label = self.rev_map[alpha_label]
            alpha_idx = idx
            beta_idx = self.lab2id[beta_label]
            sym_loss += self.sym_loss(alpha, beta, alpha_idx, beta_idx).mean()
            # sym_loss += self.sym_loss(alpha, beta, alpha_idx, beta_idx)
        return sym_loss

    # 学习率缩小80倍, 不需要conj_loss_not的性能为67.23
    def compute_trans(self, alpha, beta, gamma):
        trans_loss = torch.tensor(0, dtype=torch.float, requires_grad=False).to(self.device)
        nx = 0
        for k, v in self.rule_pools.items():
            alpha_label, beta_label = k
            alpha_idx, beta_idx = self.lab2id[alpha_label], self.lab2id[beta_label]
            for gamma_label in v['yes']:
                nx += 1
                gamma_idx = self.lab2id[gamma_label]
                trans_loss += self.conj_loss_yes(alpha, beta, gamma, alpha_idx, beta_idx, gamma_idx).mean()
                # trans_loss += self.conj_loss(alpha, beta, gamma, alpha_idx, beta_idx, gamma_idx)

            # for gamma_label in v['not']: #
            #     nx += 1
            #     gamma_idx = self.lab2id[gamma_label]
            #     trans_loss += self.conj_loss_not(alpha, beta, gamma, alpha_idx, beta_idx, gamma_idx).mean()

        return trans_loss / nx
