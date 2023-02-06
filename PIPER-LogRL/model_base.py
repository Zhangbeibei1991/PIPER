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


class TransitivityLoss(nn.Module):
    """
    alpha ^ beta -> gamma, where gamma is not conflict
    """

    def __init__(self, device):
        super(TransitivityLoss, self).__init__()
        self.zero = torch.tensor(0, dtype=torch.float, requires_grad=False).to(device)

    def forward(self, alpha, beta, gamma, alpha_idx, beta_idx, gamma_idx):
        conj_loss = torch.max(self.zero, alpha[:, alpha_idx] + beta[:, beta_idx] - gamma[:, gamma_idx])
        return conj_loss


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
        self.word_embed = nn.Embedding.from_pretrained(embeddings=torch.FloatTensor(word_embed_table), freeze=True)
        self.tag_embed = nn.Embedding(num_embeddings=1000, embedding_dim=self.args.tag_dim)
        self.lab_embed = nn.Embedding(num_embeddings=1000, embedding_dim=self.args.lab_dim, padding_idx=0)

        self.encoder = AutoModel.from_pretrained(self.args.bert_path)
        self.config = AutoConfig.from_pretrained(self.args.bert_path)

        input_dim = self.args.embed_dim + self.args.tag_dim + self.config.hidden_size
        hidden_dim = self.config.hidden_size
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
        self.state_dim = hidden_dim * n_repeat
        self.hidden = nn.Linear(hidden_dim * n_repeat + self.state_dim + self.args.lab_dim, self.state_dim)
        self.classifier = nn.Linear(self.state_dim, num_classes)

        self.state_init = nn.Embedding(1000, self.state_dim).from_pretrained(
            torch.rand([1000, self.state_dim], dtype=torch.float), freeze=False).weight

        self.sym_loss = SymmetryLoss()
        self.conj_loss = TransitivityLoss(device=self.device)
        self.cross_backbone = nn.CrossEntropyLoss(reduction="none")

        self.true_pair = set()
        self.pred_pair = set()
        self.modify_pair = set()

    def calc_PRF(self, gold, pred):
        corr = len(gold & pred)
        pred_num = len(pred)
        gold_num = len(gold)
        P = corr / (pred_num + 1e-10)
        R = corr / (gold_num + 1e-10)
        F = P * R * 2 / (P + R + 1e-10)
        return P, R, F

    def sampling(self, prob):
        if not self.training:
            return torch.max(prob, 1).indices
        else:
            return torch.stack([torch.multinomial(prob[k], 1) for k in range(prob.size(0))], dim=0).squeeze(dim=-1)

    def reinforcement_learning(self, feats, ab_lab, ba_lab):
        batch_rl_loss = torch.tensor(0, dtype=torch.float, requires_grad=False).to(self.device)
        total_rewards = []
        for k in range(self.args.sample_round):
            state, lab_idx, actions, act_probs = None, None, [], []
            for k, feat in enumerate(feats):
                state, prob, logit = self.agent(feat, state=state, lab_idx=lab_idx)
                action = self.sampling(prob=prob)
                act_prob = torch.stack([prob[n][s] for n, s in enumerate(action.tolist())], dim=0)
                lab_idx = action
                actions.append(action)
                act_probs.append(act_prob)
            actions = torch.stack(actions, dim=1).tolist()
            act_probs = torch.stack(act_probs, dim=1)

            rewards = torch.zeros(size=(act_probs.size(0), 2), dtype=torch.float).to(self.device)
            batch_size, path_len = act_probs.size()
            grads = torch.tensor(0., requires_grad=True).to(self.device)
            for n in range(batch_size):
                if actions[n][0] == ab_lab[n] and actions[n][1] == ba_lab[n]:
                    rewards[n][0] = 1
                    rewards[n][1] = 1
                if actions[n][0] != ab_lab[n] and actions[n][1] == ba_lab[n]:
                    rewards[n][0] = -1
                    rewards[n][1] = 1
                if actions[n][0] == ab_lab[n] and actions[n][1] != ba_lab[n]:
                    rewards[n][0] = 1
                    rewards[n][1] = -1
                if actions[n][0] != ab_lab[n] and actions[n][1] != ba_lab[n]:
                    rewards[n][0] = -1
                    rewards[n][1] = -1

                decay_r = 0.
                avg = 0.
                for i in range(path_len):
                    avg = (avg + rewards[n][i]) / (i + 1)
                    decay_r = decay_r * 0.95 + rewards[n][i]
                    to_grad = -torch.log(act_probs[n][i])
                    to_grad *= torch.tensor(decay_r - avg, requires_grad=True).to(self.device)
                    # to_grad *= torch.tensor(decay_r, requires_grad=True).to(self.device)
                    grads = grads + to_grad
            total_rewards.append(rewards.sum())
            # grads.backward(retain_graph=True)
            batch_rl_loss += grads
        batch_rl_loss = batch_rl_loss / self.args.sample_round
        avg_rewards = torch.sum(torch.stack(total_rewards, dim=0)) / self.args.sample_round / feats[0].size(0)
        ab_state, ab_prob, ab_logit = self.agent(feats[0], state=None, lab_idx=None)
        lab_idx = torch.argmax(ab_logit, dim=-1)
        _, _, ba_logit = self.agent(feats[1], state=ab_state, lab_idx=lab_idx)
        return ab_logit, ba_logit, batch_rl_loss, avg_rewards

    def forward(self, batch, stage="backbone", mode="train"):
        ab_info_tensor, bc_info_tensor, ac_info_tensor, info = batch
        if stage == "backbone":
            feat_ab, ab_lab, ab_dis_ns = self.context_embeddings(info_tensor=ab_info_tensor, e1=0, e2=1)
            state, prob, ab_logit = self.agent(feat_ab)
            if mode == "train":
                ab_loss = self.cross_backbone(ab_logit, ab_lab).mean() + self.args.dis_wt * ab_dis_ns
                return ab_loss
            else:
                return ab_logit
        elif stage == "symmetry":
            feat_ab, ab_lab, ab_dis_ns = self.context_embeddings(info_tensor=ab_info_tensor, e1=0, e2=1)
            if mode == "train":
                feat_ba, ba_lab, ba_dis_ns = self.context_embeddings(info_tensor=ab_info_tensor, e1=1, e2=0)
                feats = [feat_ab, feat_ba]
                ab_logit, ba_logit, batch_rl_loss, avg_r = self.reinforcement_learning(feats=feats,
                                                                                       ab_lab=ab_lab.tolist(),
                                                                                       ba_lab=ba_lab.tolist())
                ab_loss = self.cross_backbone(ab_logit, ab_lab).sum() + self.args.dis_wt * ab_dis_ns
                ba_loss = self.cross_backbone(ba_logit, ba_lab).sum() + self.args.dis_wt * ba_dis_ns
                sym_loss = self.compute_sym(alpha=ab_logit, beta=ba_logit)
                batch_loss = self.args.cross_wt * (ab_loss + ba_loss) + self.args.sym_wt * sym_loss
                return batch_loss, batch_rl_loss, avg_r, ab_logit, ba_logit
            else:
                ab_state, ab_prob, ab_logit = self.agent(feat_ab, state=None, lab_idx=None)
                lab_idx = torch.argmax(ab_logit, dim=-1)
                feat_ba, ba_lab, ba_dis_ns = self.context_embeddings(info_tensor=ab_info_tensor, e1=1, e2=0)
                _, _, ba_logit = self.agent(feat_ba, state=ab_state, lab_idx=lab_idx)
                return ab_logit
        elif stage == "transitivity":
            ab_logit, ab_lab, ab_dis_ns = self.context_embeddings(info_tensor=ab_info_tensor, e1=0, e2=1)
            if mode == "train":
                bc_logit, bc_lab, bc_dis_ns = self.context_embeddings(info_tensor=bc_info_tensor, e1=0, e2=1)
                ac_logit, ac_lab, ac_dis_ns = self.context_embeddings(info_tensor=ac_info_tensor, e1=0, e2=1)
                ab_loss = self.cross_backbone(ab_logit, ab_lab).sum() + self.args.dis_wt * ab_dis_ns
                bc_loss = self.cross_backbone(bc_logit, bc_lab).sum() + self.args.dis_wt * bc_dis_ns
                ac_loss = self.cross_backbone(ac_logit, ac_lab).sum() + self.args.dis_wt * ac_dis_ns
                trans_loss = self.compute_trans(alpha=F.log_softmax(ab_logit, dim=-1),
                                                beta=F.log_softmax(bc_logit, dim=-1),
                                                gamma=F.log_softmax(ac_logit, dim=-1))
                batch_loss = self.args.cross_wt * (ab_loss + bc_loss + ac_loss) + self.args.trans_wt * trans_loss
                return batch_loss
            else:
                return ab_logit
        elif stage == "sym+trans" or stage == "backbone->sym+trains":
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
                batch_loss = self.args.cross_wt * (
                        ab_loss + ba_loss + bc_loss + cb_loss + ac_loss + ca_loss) + self.args.sym_wt * (
                                     ab_sym_loss + bc_sym_loss + ac_sym_loss) + self.args.trans_wt * trans_loss
                return batch_loss
            else:
                return ab_logit

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
        if e1 == 1:
            labs = rev_labs
        return features, labs, dis_ns

    def agent(self, feat, state=None, lab_idx=None):  # state: 768, 50
        if state is None:
            state = self.state_init[:feat.size(0), :]
        if lab_idx is None:
            lab_idx = torch.zeros([feat.size(0), ], dtype=torch.long).to(self.device)
        lab_embed = self.lab_embed(lab_idx)
        feats = torch.cat([feat, state, lab_embed], dim=-1)
        state = self.dropout(F.gelu(self.hidden(feats)))
        logits = self.classifier(state)  # (batch_size, num_classes)
        prob = F.softmax(logits, dim=-1)
        return state, prob, logits

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
        return sym_loss

    def compute_trans(self, alpha, beta, gamma):
        trans_loss = torch.tensor(0, dtype=torch.float, requires_grad=False).to(self.device)
        nx = 0
        for k, v in self.rule_pools.items():
            alpha_label, beta_label = k
            alpha_idx, beta_idx = self.lab2id[alpha_label], self.lab2id[beta_label]
            for gamma_label in v['yes']:
                nx += 1
                gamma_idx = self.lab2id[gamma_label]
                trans_loss += self.conj_loss(alpha, beta, gamma, alpha_idx, beta_idx, gamma_idx).mean()
        return trans_loss
