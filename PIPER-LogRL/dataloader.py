import copy
import os
import re
import json
import pickle
import torch
import math
import random
import numpy as np
from tqdm import tqdm
from collections import Counter, OrderedDict
from transformers import AutoTokenizer


class PIPERDataloader:
    def __init__(self, train_path, devel_path, test_path, tokenizer, logger, max_len, task_name, stage, rules):
        self.train_path = train_path
        self.devel_path = devel_path
        self.test_path = test_path
        self.tokenizer = tokenizer
        self.logger = logger
        self.max_len = max_len
        self.task_name = task_name
        self.stage = stage
        self.rules = rules

        self.rev_map = OrderedDict([('VAGUE', 'VAGUE'),
                                    ('BEFORE', 'AFTER'),
                                    ('AFTER', 'BEFORE'),
                                    ('SIMULTANEOUS', 'SIMULTANEOUS'),
                                    ('INCLUDES', 'IS_INCLUDED'),
                                    ('IS_INCLUDED', 'INCLUDES')])

    def load_data(self, file_path):
        lines = json.load(open(file_path, mode="r", encoding="utf-8"))
        return lines

    def build_map(self):
        if not os.path.exists(f"experiments/{self.task_name}/schema.json"):
            data = self.load_data(file_path=self.train_path)
            data += self.load_data(file_path=self.devel_path)
            data += self.load_data(file_path=self.test_path)

            deps, tags, trigger, entity, tense, labs, tokens = [], [], [], [], [], [], []
            for instance in tqdm(data, desc="build schema"):
                if instance["label"].lower() == "equal":
                    labs.append("SIMULTANEOUS")
                    # continue
                else:
                    labs.append(instance['label'])
                deps.extend([item[0] for item in instance["dep"]])
                tags.extend(instance["pos"])
                if "e1_type" in instance:
                    entity.append(instance["e1_type"])
                    entity.append(instance["e2_type"])
                else:
                    entity.append("NONE")

                if "e1_tense" in instance:
                    tense.append(instance['e1_tense'])
                    tense.append(instance['e2_tense'])
                else:
                    tense.append("NONE")
                tokens.extend(instance['token'])

            word_tuple = ["[PAD]", "[UNK]", "$EVENT$"] + [item[0] for item in
                                                          sorted(dict(Counter(tokens)).items(), key=lambda x: x[1],
                                                                 reverse=True)]
            dep_tuple = ["O"] + [item[0] for item in
                                 sorted(dict(Counter(deps)).items(), key=lambda x: x[1], reverse=True)] + ["self-loop"]
            tag_tuple = ["O"] + [item[0] for item in
                                 sorted(dict(Counter(tags)).items(), key=lambda x: x[1], reverse=True)]
            ten_tuple = [item[0] for item in
                         sorted(dict(Counter(tense)).items(), key=lambda x: x[1], reverse=True)]
            ent_tuple = [item[0] for item in
                         sorted(dict(Counter(entity)).items(), key=lambda x: x[1], reverse=True)]
            lab_tuple = [item[0] for item in
                         sorted(dict(Counter(labs)).items(), key=lambda x: x[1], reverse=True)]

            dep2id = {key: i for i, key in enumerate(dep_tuple)}
            tag2id = {key: i for i, key in enumerate(tag_tuple)}
            ent2id = {key: i for i, key in enumerate(ent_tuple)}
            lab2id = {key: i for i, key in enumerate(lab_tuple)}
            tense2id = {key: i for i, key in enumerate(ten_tuple)}
            word2id = {key: i for i, key in enumerate(word_tuple)}

            schema = {"lab": lab2id, "ent": ent2id, "dep": dep2id, "tag": tag2id, "tense": tense2id, "word": word2id}
            with open(f"experiments/{self.task_name}/schema.json", encoding="utf-8", mode="w") as f:
                json.dump(schema, f, ensure_ascii=False, indent=4)
        else:
            schema = json.load(open(f"experiments/{self.task_name}/schema.json", encoding="utf-8", mode="r"))
        return schema

    def load_word2vec(self, emb_path, id_to_word, word_dim, old_weights):
        """
        load word embedding from pre-trained file
        embedding size must match
        """
        new_weights = old_weights
        self.logger.info('Loading pretrained embeddings from {}...'.format(emb_path))
        pre_trained = {}
        emb_invalid = 0
        embed_name = os.path.basename(emb_path)
        for i, line in tqdm(enumerate(open(emb_path, mode='r', encoding='utf-8'), start=0),
                            desc=f"loading {embed_name}"):
            line = line.rstrip().split()
            if len(line) == word_dim + 1:
                pre_trained[line[0]] = np.array(
                    [float(x) for x in line[1:]]
                ).astype(np.float32)
            else:
                emb_invalid += 1
        if emb_invalid > 0:
            self.logger.info('WARNING: %i invalid lines' % emb_invalid)
        c_found = 0
        c_lower = 0
        c_zeros = 0
        n_words = len(id_to_word)
        for i in range(n_words):
            word = id_to_word[i]
            if word in pre_trained:
                new_weights[i] = pre_trained[word]
                c_found += 1
            elif word.lower() in pre_trained:
                new_weights[i] = pre_trained[word.lower()]
                c_lower += 1
            elif re.sub('\d', '0', word.lower()) in pre_trained:
                new_weights[i] = pre_trained[
                    re.sub('\d', '0', word.lower())
                ]
                c_zeros += 1
        self.logger.info('>> Loaded %i pretrained embedding.' % len(pre_trained))
        self.logger.info('>> %i / %i (%.4f%%) words have been initialized with'
                         'pretrained embeddings.' % (
                             c_found + c_lower + c_zeros, n_words,
                             100. * (c_found + c_lower + c_zeros) / n_words)
                         )
        self.logger.info('>> %i found directly, %i after lowercasing, '
                         '%i after lowercasing + zero.' % (
                             c_found, c_lower, c_zeros
                         ))
        info = ['>> Loaded %i pretrained embedding.' % len(pre_trained),
                '>> %i / %i (%.4f%%) words have been initialized with'
                'pretrained embeddings.' % (
                    c_found + c_lower + c_zeros, n_words,
                    100. * (c_found + c_lower + c_zeros) / n_words),
                '>> %i found directly, %i after lowercasing, '
                '>> %i after lowercasing + zero.' % (
                    c_found, c_lower, c_zeros
                )]
        return new_weights, info

    def get_tensor(self, data_path, dataset_name, schema):
        all_data = self.load_data(file_path=data_path)
        ent2id, lab2id, tag2id = schema["ent"], schema["lab"], schema["tag"],
        dep2id, ten2id, word2id = schema["dep"], schema["tense"], schema["word"]
        pp_data, sent_set, event_set, event_pair_set = {}, set(), set(), set()
        for i in tqdm(range(len(all_data)), desc=f"generating {dataset_name} instances"):
            instance = all_data[i]
            # if instance["label"].lower() == "equal":
            #     # labs.append("SIMULTANEOUS")
            #     continue
            doc_id = instance["doc_id"]
            e1, e2 = instance["event_pairs_id"]
            e1_span, e2_span = instance["e1_span"], instance["e2_span"]
            tokens = instance["token"]
            token_ids = [word2id[tk] for tk in tokens]
            lab = instance["label"]
            if lab.lower() == "equal":
                lab = "SIMULTANEOUS"
            lab_id = lab2id[lab]
            rev_lab = self.rev_map[lab]
            rev_lab_id = lab2id[rev_lab]
            if "e1_type" not in instance:
                instance["e1_type"] = "NONE"
                instance["e2_type"] = "NONE"
            if "e1_tense" not in instance:
                instance["e1_tense"] = "NONE"
                instance["e2_tense"] = "NONE"
            ents = [ent2id[instance["e1_type"]], ent2id[instance["e2_type"]], instance["e1_type"], instance["e2_type"]]
            tenses = [ten2id[instance["e1_tense"]], ten2id[instance["e2_tense"]], instance["e1_tense"],
                      instance["e2_tense"]]
            deps = [[i, dep2id[item[0]], item[1]] for item in instance["dep"] if item[1] != -1] + [
                [-1, dep2id["self-loop"], -1]]
            tags = [tag2id[tg] for tg in instance["pos"]]
            sent_set.add(" ".join(tokens))
            event_set.add(f"{doc_id}&&{e1}")
            event_set.add(f"{doc_id}&&{e2}")
            event_pair_set.add((f"{doc_id}&&{e1}", f"{doc_id}&&{e2}"))
            token_sent = [self.tokenizer.cls_token]
            bert_span = []
            offsets = []
            bert_start, bert_end = 1, 0
            for token_index, token in enumerate(tokens):
                bert_tk = self.tokenizer.tokenize(token)
                bert_end = bert_start + len(bert_tk)
                token_sent.extend(bert_tk)
                bert_span.append((bert_start, bert_end))
                offsets.append(bert_start)
                bert_start = bert_end

            token_sent.append(self.tokenizer.sep_token)

            bert_tk_id = self.tokenizer.encode(' '.join(tokens))

            assert len(token_sent) == len(bert_tk_id)
            assert len(tokens) == len(offsets)

            if doc_id not in pp_data:
                pp_data[doc_id] = []
            assert len(tokens) == len(token_ids)
            pp_data[doc_id].append({
                "doc_id": doc_id,
                "text": tokens,
                "bert":
                    {"tokens": tokens, "token_ids": token_ids, "bert-tokens": token_sent,
                     "input_ids": bert_tk_id, "offsets": offsets},
                "id": [e1, e2],
                "span": [e1_span, e2_span],
                "lab": [lab_id, lab, rev_lab_id, rev_lab],
                "ent": ents,
                "tense": tenses,
                "dep": deps,
                "tag": tags,
            })
        instances = []
        symmetry_count, transitivity_count = 0, 0
        symmetry_sent, transitivity_sent = set(), set()
        for doc_id, doc_contents in tqdm(pp_data.items(), desc="generating logic instances"):
            already_set = []  # 用多任务来实现联合抽取
            for m, data_e12 in enumerate(doc_contents):
                assert len(data_e12["bert"]["tokens"]) == len(data_e12["bert"]["token_ids"]) == len(data_e12["text"])
                instances.append({"ab": data_e12, "bc": data_e12, "ac": data_e12, "reverse": [True, True, True],
                                  "task": "symmetry"})
                symmetry_sent.add(f"{doc_id}&&sent-{m}")
                symmetry_count += 1
                for n, data_e23 in enumerate(doc_contents):
                    assert len(data_e23["bert"]["tokens"]) == len(data_e23["bert"]["token_ids"]) == len(
                        data_e23["text"])
                    for q, data_e13 in enumerate(doc_contents):
                        assert len(data_e13["bert"]["tokens"]) == len(data_e13["bert"]["token_ids"]) == len(
                            data_e13["text"])
                        reverse_flag, e_box = self.judge_conj(ab=data_e12["id"], bc=data_e23["id"], ac=data_e13["id"])
                        if reverse_flag is not None and e_box not in already_set:
                            already_set.append(e_box)
                            line = {"ab": data_e12, "bc": data_e23, "ac": data_e13, "reverse": reverse_flag,
                                    "task": "transitivity"}
                            if line not in instances:
                                instances.append(line)
                                transitivity_sent.add(f"{doc_id}&&sent-{m}")
                                transitivity_sent.add(f"{doc_id}&&sent-{n}")
                                transitivity_sent.add(f"{doc_id}&&sent-{q}")
                                transitivity_count += 1
        total_sym = len(symmetry_sent)
        total_trans = len(transitivity_sent)
        sym_trans = len(symmetry_sent & transitivity_sent)
        P = sym_trans / total_sym
        R = sym_trans / total_trans
        F = 2 * P * R / (P + R)
        info = [f"sent num: {len(sent_set)}",
                f"evt num: {len(event_set)}",
                f"pair num: {len(event_pair_set)}",
                f"symmetry count: {symmetry_count}",
                f"transitivity count: {transitivity_count}",
                f"cover rate: F: {round(P, 4)}, R: {round(R, 4)}, F1: {round(F, 4)}"]
        return instances, info

    def judge_conj(self, ab, bc, ac):
        # (a->ab[0], b->ab[1]), (b->bc[0], c->bc[1]), (a->ac[0], c->ac[1])
        judge_abc1 = ab[1] == bc[0] and ab[0] == ac[0] and bc[1] == ac[1]
        if judge_abc1:
            e_box = []
            e_box.append(ab)
            e_box.append(bc)
            e_box.append(ac)
            return [True, True, True], e_box
        # (a->ab[0], b->ab[1]), (b->bc[0], c->bc[1]), (c->ac[0], a->ac[1])
        judge_abc2 = ab[1] == bc[0] and ab[0] == ac[1] and bc[1] == ac[0]
        if judge_abc2:
            e_box = []
            e_box.append(ab)
            e_box.append(bc)
            e_box.append(ac)
            return [True, True, False], e_box
        # (a->ab[0], b->ab[1]), (c->bc[0], b->bc[1]), (a->ac[0], c->ac[1])
        judge_abc3 = ab[1] == bc[1] and ab[0] == ac[0] and bc[0] == ac[1]
        if judge_abc3:
            e_box = []
            e_box.append(ab)
            e_box.append(bc)
            e_box.append(ac)
            return [True, False, True], e_box
        # (a->ab[0], b->ab[1]), (c->bc[0], b->bc[1]), (c->ac[0], a->ac[1])
        judge_abc4 = ab[1] == bc[1] and ab[0] == ac[1] and bc[0] == ac[0]
        if judge_abc4:
            e_box = []
            e_box.append(ab)
            e_box.append(bc)
            e_box.append(ac)
            return [True, False, False], e_box
        # (b->ab[0], a->ab[1]), (b->bc[0], c->bc[1]), (a->ac[0], c->ac[1])
        judge_abc5 = ab[0] == bc[0] and ab[1] == ac[0] and bc[1] == ac[1]
        if judge_abc5:
            e_box = []
            e_box.append(ab)
            e_box.append(bc)
            e_box.append(ac)
            return [False, True, True], e_box
        # (b->ab[0], a->ab[1]), (b->bc[0], c->bc[1]), (c->ac[0], a->ac[1])
        judge_abc6 = ab[0] == bc[0] and ab[1] == ac[1] and bc[1] == ac[0]
        if judge_abc6:
            e_box = []
            e_box.append(ab)
            e_box.append(bc)
            e_box.append(ac)
            return [False, True, False], e_box
        # (b->ab[0], a->ab[1]), (c->bc[0], b->bc[1]), (a->ac[0], c->ac[1])
        judge_abc7 = ab[0] == bc[1] and ab[1] == ac[0] and bc[0] == ac[1]
        if judge_abc7:
            e_box = []
            e_box.append(ab)
            e_box.append(bc)
            e_box.append(ac)
            return [False, False, True], e_box
        # (b->ab[0], a->ab[1]), (c->bc[0], b->bc[1]), (c->ac[0], a->ac[1])
        judge_abc8 = ab[0] == bc[1] and ab[1] == ac[1] and bc[0] == ac[0]
        if judge_abc8:
            e_box = []
            e_box.append(ab)
            e_box.append(bc)
            e_box.append(ac)
            return [False, False, True], e_box
        return None, None

    def get_batch_data(self, data, mode, batch_size):
        if self.stage == "backbone":
            data = [line for line in data if line["task"] == "symmetry"]
        elif self.stage == "symmetry":
            if mode == "train":
                data = [line for line in data if line["task"] == "symmetry"]
                # 用一下过采样
                class_nums = {}
                class_x = {}
                for line in data:
                    lab = line["ab"]["lab"][1]
                    if lab not in class_nums:
                        class_nums[lab] = 1
                        class_x[lab] = [line]
                    else:
                        class_nums[lab] += 1
                        class_x[lab].append(line)
                max_num = max(list(class_nums.values()))
                oversample_data = []
                for key, num in class_nums.items():
                    x_aux = np.random.choice(data,size=max_num - num, replace=True).tolist()
                    x = class_x[key] + x_aux
                    oversample_data.extend(x)
                data = oversample_data
            else:
                data = [line for line in data if line["task"] == "symmetry"]
        else:
            if mode == "train":
                data = [line for line in data if
                        line["task"] == "transitivity" and line["reverse"] == [True, True, True]]
                # # 用一下过采样
                # class_nums = {}
                # class_x = {}
                # for line in data:
                #     lab_ab = line["ab"]["lab"][1]
                #     lab_bc = line["bc"]["lab"][1]
                #     lab_ac = line["ac"]["lab"][1]
                #     lab = f"{lab_ab}->{lab_bc}->{lab_ac}"
                #     # lab = lab_ab
                #     if "VAGUE" in lab:
                #         continue
                #     if (lab_ab, lab_bc) not in self.rules:
                #         continue
                #     if (lab_ab, lab_bc) in self.rules and lab_ac not in self.rules[((lab_ab, lab_bc))]["yes"]:
                #         continue
                #     if lab not in class_nums:
                #         class_nums[lab] = 1
                #         class_x[lab] = [line]
                #     else:
                #         class_nums[lab] += 1
                #         class_x[lab].append(line)
                # max_num = max(list(class_nums.values()))
                # oversample_data = []
                # for key, num in class_nums.items():
                #     # x_aux = np.random.choice(data, size=max_num - num, replace=True).tolist()
                #     x_aux = []
                #     x = class_x[key] + x_aux
                #     oversample_data.extend(x)
                # data = oversample_data
            else:
                data = [line for line in data if line["task"] == "symmetry"]
        self.num_batch = int(math.ceil(len(data) / batch_size))
        self.logger.info("{} num_batch: {} fom {}".format(mode, self.num_batch, len(data)))
        sorted_data = sorted(data,
                             key=lambda x: len(x["ab"]["bert"]['input_ids']) + len(x["bc"]["bert"]['input_ids']) + len(
                                 x["ac"]["bert"]['input_ids']), reverse=True)
        batch_data = list()
        for i in range(self.num_batch):
            batch_data.append(self.pad_data(sorted_data[i * batch_size: (i + 1) * batch_size]))
        return batch_data

    def pad_data(self, data):
        ab_bert, bc_bert, ac_bert = [], [], []
        ab_words, bc_words, ac_words = [], [], []
        ab_tags, bc_tags, ac_tags = [], [], []
        ab_spans, bc_spans, ac_spans = [], [], []
        ab_offsets, bc_offsets, ac_offsets = [], [], []
        ab_adj, bc_adj, ac_adj = [], [], []
        ab_lens, bc_lens, ac_lens = [], [], []
        ab_labs, bc_labs, ac_labs = [], [], []
        ab_rev_labs, bc_rev_labs, ac_rev_labs = [], [], []
        tasks, flags = [], []

        ab_b_max = max([len(item["ab"]['bert']['input_ids']) for item in data])
        bc_b_max = max([len(item["bc"]['bert']['input_ids']) for item in data])
        ac_b_max = max([len(item["ac"]['bert']['input_ids']) for item in data])

        ab_w_max = max([len(item["ab"]['bert']['token_ids']) for item in data])
        bc_w_max = max([len(item["bc"]['bert']['token_ids']) for item in data])
        ac_w_max = max([len(item["ac"]['bert']['token_ids']) for item in data])

        info = {"len": [], "doc_id": [], "evt_id": [], "text": [], "gold": [], "task": [], "rev": [], "ent": []}

        for line in data:
            ab_list = self.parse(line=line, key="ab", b_max=ab_b_max, w_max=ab_w_max)
            ab_input_id, ab_word_id, ab_tag, ab_offset, ab_evt_span, ab_lab, ab_doc_id, ab_evt_id, ab_dep, ab_word_len, ab_tokens, ab_graph, ab_ent = ab_list

            bc_list = self.parse(line=line, key="bc", b_max=bc_b_max, w_max=bc_w_max)
            bc_input_id, bc_word_id, bc_tag, bc_offset, bc_evt_span, bc_lab, bc_doc_id, bc_evt_id, bc_dep, bc_word_len, bc_tokens, bc_graph, bc_ent = bc_list

            ac_list = self.parse(line=line, key="ac", b_max=ac_b_max, w_max=ac_w_max)
            ac_input_id, ac_word_id, ac_tag, ac_offset, ac_evt_span, ac_lab, ac_doc_id, ac_evt_id, ac_dep, ac_word_len, ac_tokens, ac_graph, ac_ent = ac_list

            ab_bert.append(ab_input_id)
            bc_bert.append(bc_input_id)
            ac_bert.append(ac_input_id)

            ab_words.append(ab_word_id)
            bc_words.append(bc_word_id)
            ac_words.append(ac_word_id)

            ab_tags.append(ab_tag)
            bc_tags.append(bc_tag)
            ac_tags.append(ac_tag)

            ab_offsets.append(ab_offset)
            bc_offsets.append(bc_offset)
            ac_offsets.append(ac_offset)

            ab_adj.append(ab_graph)
            bc_adj.append(bc_graph)
            ac_adj.append(ac_graph)

            ab_spans.append(ab_evt_span)
            bc_spans.append(bc_evt_span)
            ac_spans.append(ac_evt_span)

            ab_labs.append(ab_lab[0])
            bc_labs.append(bc_lab[0])
            ac_labs.append(ac_lab[0])

            ab_rev_labs.append(ab_lab[2])
            bc_rev_labs.append(bc_lab[2])
            ac_rev_labs.append(ac_lab[2])

            ab_lens.append(ab_word_len)
            bc_lens.append(bc_word_len)
            ac_lens.append(ac_word_len)

            info["len"].append([ab_word_len, bc_word_len, ac_word_len])
            info["doc_id"].append([ab_doc_id, bc_doc_id, ac_doc_id])
            info["evt_id"].append([ab_evt_id, bc_evt_id, ac_evt_id])
            info["text"].append([ab_tokens, bc_tokens, ac_tokens])
            info["gold"].append([ab_lab[1], bc_lab[1], ac_lab[1]])
            info["rev"].append([ab_lab[3], bc_lab[3], ac_lab[3]])
            info["ent"].append([ab_ent[2:], bc_ent[2:], ac_ent[2:]])
            info["task"].append(line["task"])
            task_attr = line["task"]
            reversed_flag = line["reverse"]
            if task_attr == "symmetry":
                tasks.append(0)
                reversed_flag = [0, 0, 0]
            else:
                tasks.append(1)
                reversed_flag = [int(item) for item in reversed_flag]
            flags.append(reversed_flag)

        ab_info = [ab_bert, ab_words, ab_tags, ab_offsets, ab_adj, ab_spans, ab_lens, ab_labs, ab_rev_labs, tasks,
                   flags]
        bc_info = [bc_bert, bc_words, bc_tags, bc_offsets, bc_adj, bc_spans, bc_lens, bc_labs, bc_rev_labs, tasks,
                   flags]
        ac_info = [ac_bert, ac_words, ac_tags, ac_offsets, ac_adj, ac_spans, ac_lens, ac_labs, ac_rev_labs, tasks,
                   flags]
        ab_info_tensor = [torch.LongTensor(item) for item in ab_info]
        bc_info_tensor = [torch.LongTensor(item) for item in bc_info]
        ac_info_tensor = [torch.LongTensor(item) for item in ac_info]

        return ab_info_tensor, bc_info_tensor, ac_info_tensor, info

    def parse(self, line, key, b_max, w_max):
        bert_info = line[key]['bert']
        input_id = bert_info['input_ids']
        word_id = bert_info['token_ids']
        offset = bert_info['offsets']
        doc_id = line[key]["doc_id"]
        evt_id = line[key]["id"]
        evt_span = line[key]["span"]
        token = line[key]["text"]
        tag = line[key]["tag"]
        dep = line[key]["dep"]
        lab = line[key]["lab"]
        ent = line[key]["ent"]
        word_len = len(token)
        assert len(word_id) == len(offset) == len(token)
        if len(input_id) <= b_max:  # Bert补全
            pad_token = [0] * (b_max - len(input_id))
            input_id = input_id + pad_token
        if len(offset) <= w_max:
            pad_token = [0] * (w_max - len(offset))
            offset = offset + pad_token
            word_id = word_id + pad_token
            tag = tag + pad_token
        graph = np.zeros(shape=(w_max, w_max))
        for k, item in enumerate(dep[:-1]):
            graph[k, item[2]] = 1.0
            graph[item[2], k] = 1.0
        for k in range(w_max):
            graph[k, k] = 1.0
        return input_id, word_id, tag, offset, evt_span, lab, doc_id, evt_id, dep, word_len, token, graph, ent

    def iter_batch(self, batch_data, shuffle=False):
        if shuffle:
            random.shuffle(batch_data)
        len_data = len(batch_data)
        for idx in range(len_data):
            yield batch_data[idx]


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased")
    # loader = PIPERDataloader(task_name="TB-Dense",
    #                          train_path="sources/TB-Dense/train.json",
    #                          devel_path="sources/TB-Dense/dev.json",
    #                          test_path="sources/TB-Dense/test.json",
    #                          tokenizer=tokenizer,
    #                          logger=None,
    #                          max_len=512)
    # schema = loader.build_map()
    # pp_data, info = loader.get_tensor(data_path="sources/TB-Dense/train.json", dataset_name="train", schema=schema)
    # with open("experiments/TB-Dense/train-bert-base-cased.pt", mode="wb") as f:
    #     pickle.dump([pp_data, info], f)
    '''
    sent num: 615
    evt num: 1042
    pair num: 4032
    symmetry count: 4032
    transitivity count: 54000
    cover rate: F: 0.9936, R: 1.0, F1: 0.9968
    '''
    loader = PIPERDataloader(task_name="MATRES",
                             train_path="../sources/MATRES/matres_qiangning/train.json",
                             devel_path="../sources/MATRES/matres_qiangning/dev.json",
                             test_path="../sources/MATRES/matres_qiangning/test.json",
                             tokenizer=tokenizer,
                             logger=None,
                             max_len=512)
    schema = loader.build_map()
    pp_data, info = loader.get_tensor(data_path="../sources/MATRES/matres_qiangning/train.json", dataset_name="train",
                                      schema=schema)
    with open("experiments/MATRES/train-bert-base-uncased.pt", mode="wb") as f:
        pickle.dump([pp_data, info], f)
