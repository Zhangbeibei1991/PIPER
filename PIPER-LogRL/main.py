"""
这个脚本重构了唐婧尧的代码，希望在此基础之上有所进展
"""
import os
import time
import random
import torch
import copy
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from logger import Log
from dataloader import PIPERDataloader
from transformers import AutoTokenizer
from model_base import PIPERModel
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, \
    roc_auc_score
from rule_pools import tbd, matres
import torch.nn.functional as F
import warnings

torch.cuda.current_device()
torch.cuda._initialized = True
warnings.filterwarnings("ignore")


class PIPERMain(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        os.environ['PYTHONHASHSEED'] = str(args.seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.seed)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.cuda.set_device(args.gpu_id)

        self.logger.info("System information".center(60, '+'))
        self.logger.info(f'cuda available: {torch.cuda.is_available()}')

        self.logger.info(f"using cuda device: {torch.cuda.current_device()}")
        self.logger.info(f'torch version: {torch.__version__}')
        self.logger.info(f'torch cuda version: {torch.version.cuda}')
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        self.logger.info(f'start running time: {time_str}')
        self.logger.info("Data preprocessing".center(60, "+"))

        self.output_path = args.output_path
        self.train_path = args.train_path
        self.devel_path = args.devel_path
        self.test_path = args.test_path

        self.encoder = args.encoder
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(f"{self.output_path}/{self.args.task_name}", exist_ok=True)

        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.train_batch_size = args.train_batch_size
        self.devel_batch_size = args.devel_batch_size
        self.test_batch_size = args.test_batch_size

        self.bert_path = args.bert_path
        self.max_len = self.args.max_len
        self.epoch = self.args.epoch
        self.max_grad_norm = self.args.max_grad_norm

        if self.args.task_name == "TB-Dense":
            self.rule_pools = tbd
        elif self.args.task_name == "MATRES":
            self.rule_pools = matres

        self.bert_name = self.bert_path.split("/")[-1] if "/" in self.bert_path else self.bert_path.split("\\")[-1]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"Invalid gradient_accumulation_steps parameter: {self.gradient_accumulation_steps}, must be >= 1")

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.bert_path)
        self.processor = PIPERDataloader(train_path=self.train_path,
                                         devel_path=self.devel_path,
                                         test_path=self.test_path,
                                         tokenizer=self.tokenizer,
                                         max_len=self.max_len,
                                         task_name=self.args.task_name,
                                         logger=self.logger,
                                         stage=self.args.stage,
                                         rules=self.rule_pools)
        schema = self.processor.build_map()
        self.word2id, self.lab2id, self.tag2id = schema["word"], schema["lab"], schema["tag"]
        self.id2lab = {v: k for k, v in self.lab2id.items()}
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.rev_lab = self.processor.rev_map
        self.embed_weight_np = None
        old_weights = np.random.rand(len(self.id2word), args.embed_dim)
        data_file = os.path.join(self.output_path, f"{self.args.task_name}/embed_seed={self.args.seed}.pt")
        if os.path.exists(data_file):
            self.logger.info(f"loading embed_weight from {data_file}")
            self.embed_weight_np, info = torch.load(data_file)
            for item in info:
                self.logger.info(item)
        else:
            self.logger.info(f"creating embed_weight:")
            self.embed_weight_np, info = self.processor.load_word2vec(emb_path=self.args.embed_path,
                                                                      id_to_word=self.id2word,
                                                                      word_dim=self.args.embed_dim,
                                                                      old_weights=old_weights)
            torch.save([self.embed_weight_np, info], data_file)

        if self.args.do_train:
            self.logger.info(f"data mode [sup train], starting:")
            data_file = os.path.join(self.output_path, f"{self.args.task_name}/train-{self.bert_name}.pt")
            if os.path.exists(data_file):
                self.logger.info(f"Loading data from {data_file}")
                train_data, train_info = torch.load(data_file)
            else:
                train_data, train_info = self.processor.get_tensor(data_path=self.train_path,
                                                                   dataset_name=f"{self.args.task_name}-train",
                                                                   schema=schema)
                torch.save([train_data, train_info], data_file)
            for item in train_info:
                self.logger.info(item)

            self.train_batch_data = self.processor.get_batch_data(data=train_data,
                                                                  mode="train",
                                                                  batch_size=self.train_batch_size)

            self.train_data_len = len(train_data)
            self.train_batch_len = len(self.train_batch_data)
            self.logger.info(f"instance number: {self.train_data_len}")
            self.logger.info(f"batch number: {self.train_batch_len}")

        if self.args.do_devel:
            self.logger.info(f"data mode [sup devel], starting:")
            data_file = os.path.join(self.output_path, f"{self.args.task_name}/devel-{self.bert_name}.pt")
            if os.path.exists(data_file):
                self.logger.info(f"Loading data from {data_file}")
                devel_data, devel_info = torch.load(data_file)
            else:
                devel_data, devel_info = self.processor.get_tensor(data_path=self.devel_path,
                                                                   dataset_name=f"{self.args.task_name}-devel",
                                                                   schema=schema)
                torch.save([devel_data, devel_info], data_file)
            for item in devel_info:
                self.logger.info(item)

            self.devel_batch_data = self.processor.get_batch_data(data=devel_data,
                                                                  mode="devel",
                                                                  batch_size=self.devel_batch_size)

            self.devel_data_len = len(devel_data)
            self.devel_batch_len = len(self.devel_batch_data)
            self.logger.info(f"instance number: {self.devel_data_len}")
            self.logger.info(f"batch number: {self.devel_batch_len}")

        if self.args.do_test:
            self.logger.info(f"data mode [sup test], starting:")
            data_file = os.path.join(self.output_path, f"{self.args.task_name}/test-{self.bert_name}.pt")
            if os.path.exists(data_file):
                self.logger.info(f"Loading data from {data_file}")
                test_data, test_info = torch.load(data_file)
            else:
                test_data, test_info = self.processor.get_tensor(data_path=self.test_path,
                                                                 dataset_name=f"{self.args.task_name}-test",
                                                                 schema=schema)
                torch.save([test_data, test_info], data_file)
            for item in test_info:
                self.logger.info(item)
            self.test_batch_data = self.processor.get_batch_data(data=test_data,
                                                                 mode="test",
                                                                 batch_size=self.test_batch_size)
            self.test_data_len = len(test_data)
            self.test_batch_len = len(self.test_batch_data)
            self.logger.info(f"instance number: {self.test_data_len}")
            self.logger.info(f"batch number: {self.test_batch_len}")

        self.PIPERModel = PIPERModel(args=self.args, num_classes=len(self.lab2id),
                                     word_embed_table=self.embed_weight_np,
                                     lab2id=self.lab2id, rev_map=self.rev_lab,
                                     device=self.device)

    def prepare_train(self, data_len, batch_size, epochs):
        self.teacher_logits = None
        if self.args.stage == "symmetry":
            model_param = torch.load(
                f"experiments/{self.args.task_name}/best-backbone-{args.encoder}-{args.feat_pair}-model.pt")
            self.PIPERModel.load_state_dict(model_param)
            self.args.lr = self.args.lr / 100.0
        elif self.args.stage == "transitivity":
            model_param = torch.load(
                f"experiments/{self.args.task_name}/best-symmetry-{args.encoder}-{args.feat_pair}-model.pt")
            model_dict = self.PIPERModel.state_dict()
            pretrained_dict = {k: v for k, v in model_param.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.PIPERModel.load_state_dict(model_dict)
            self.args.lr = self.args.lr / 80.0  # 80 -> epoch: 1->67.2 # 不要负的传递性
        elif self.args.stage == "sym+trans":
            model_param = torch.load(
                # f"experiments/{self.args.task_name}/best-backbone-{args.encoder}-{args.feat_pair}-model.pt")
                f"experiments/{self.args.task_name}/best-sym+trans-{args.encoder}-{args.feat_pair}-model-67.8%.pt")
            model_dict = self.PIPERModel.state_dict()
            pretrained_dict = {k: v for k, v in model_param.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.PIPERModel.load_state_dict(model_dict)
            self.args.lr = self.args.lr / 100.0  # 80 -> epoch: 1->67.2 # 不要负的传递性
        model = self.PIPERModel.to(self.device)
        num_train_steps = int(data_len / batch_size / self.args.gradient_accumulation_steps) * epochs
        warmup_steps = int(self.args.warmup_proportion * num_train_steps)
        optimizer = AdamW(model.parameters(), lr=self.args.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_steps)
        return model, optimizer, scheduler

    def teacher_prediction(self, batch_data, model):
        self.args.stage = "backbone"
        model.to(device=self.device)
        model.eval()
        use_obj = tqdm(self.processor.iter_batch(batch_data, shuffle=False))
        pred = {}
        for step, batch in enumerate(use_obj, start=1):
            batch = [[item.to(self.device) for item in batch[i]] for i in range(len(batch) - 1)] + [batch[-1]]
            info = batch[-1]
            with torch.no_grad():
                ab_logit = model(batch, mode="inference")
            for s, (doc_ids, evt_triple) in enumerate(zip(info["doc_id"], info["evt_id"])):
                pred[(f"{doc_ids[0]}_{evt_triple[0][0]}", f"{doc_ids[0]}_{evt_triple[0][1]}")] = ab_logit[s].detach()
        self.args.stage = "sym+trans"
        return pred

    def run_train(self):
        model, optimizer, scheduler = self.prepare_train(data_len=self.train_data_len,
                                                         epochs=self.epoch,
                                                         batch_size=self.train_batch_size, )

        global_best_F1, global_best_epoch = 0.0, 0
        for epoch in range(self.epoch):
            model.train()
            self.logger.info(f"staring training epoch: {epoch + 1}")
            all_loss, step, rl_loss, reward = 0.0, 1, 0.0, 0.0
            use_obj = tqdm(self.processor.iter_batch(self.train_batch_data, shuffle=True))
            for step, batch in enumerate(use_obj, start=1):
                batch = [[item.to(self.device) for item in batch[i]] for i in range(len(batch) - 1)] + [batch[-1]]

                batch_loss = model(batch)
                optimizer.zero_grad()
                batch_loss.backward()

                clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                all_loss += batch_loss.item()

                data_time = str(datetime.datetime.now())[:-3]
                use_obj.set_description(
                    "{0} - [\tINFO] - INFO - {1}-loss: {2:.4f}".format(data_time, self.args.stage, all_loss / step))

            self.logger.info(f"loss_overall: {round(all_loss / step, 4)}")

            if self.args.stage not in ["transitivity", "sym+trans"]:
                f1 = self.evaluate(batch_data=self.devel_batch_data, model=self.PIPERModel, name="devel")
            else:
                f1 = self.evaluate_modification(batch_data=self.devel_batch_data, model=self.PIPERModel, name="devel")

            if global_best_F1 < f1 or True:
                global_best_F1 = f1
                global_best_epoch = epoch

                if self.args.stage not in ["transitivity", "sym+trans"]:
                    self.evaluate(batch_data=self.test_batch_data, model=self.PIPERModel, name="test")
                else:
                    self.evaluate_modification(batch_data=self.test_batch_data, model=self.PIPERModel, name="test")

                torch.save(model.state_dict(),
                           f"experiments/{self.args.task_name}/best-{self.args.stage}-{self.args.encoder}-{self.args.feat_pair}-model.pt")
            self.logger.info(
                "current best micro-F1: {0:>.4%} at epoch: {1}".format(global_best_F1, global_best_epoch + 1))

            if self.args.stage == "transitivity" and epoch == 0:
                self.logger.info("early stop!")
                break

            if abs(global_best_epoch - epoch) >= self.args.early_stop:
                self.logger.info("early stop!")
                break

    def calc_f1(self, predicted_labels, all_labels, label_type):
        confusion = np.zeros((len(label_type), len(label_type)))
        for i in range(len(predicted_labels)):
            confusion[all_labels[i]][predicted_labels[i]] += 1

        acc = 1.0 * np.sum([confusion[i][i] for i in range(4)]) / np.sum(confusion)
        true_positive = 0
        for i in range(len(label_type) - 2):
            true_positive += confusion[i][i]
        prec = true_positive / (np.sum(confusion) - np.sum(confusion, axis=0)[-1])
        rec = true_positive / (np.sum(confusion) - np.sum(confusion[-1][:]))
        f1 = 2 * prec * rec / (rec + prec)

        return acc, prec, rec, f1, confusion

    def evaluate(self, batch_data, model, name):
        model.eval()
        use_obj = tqdm(self.processor.iter_batch(batch_data, shuffle=False))
        pred, gold, p_rule, g_rule, prob = [], [], [], [], []
        for step, batch in enumerate(use_obj, start=1):
            batch = [[item.to(self.device) for item in batch[i]] for i in range(len(batch) - 1)] + [batch[-1]]
            info = batch[-1]
            with torch.no_grad():
                ab_logit = model(batch, mode="inference")
            for lab in info["gold"]:
                gold.append(self.lab2id[lab[0]])
            prob.append(F.softmax(ab_logit, dim=-1))
            if self.args.stage == "MATRES":
                labs = []
                for pred_id in torch.argmax(ab_logit, dim=-1).tolist():
                    if self.id2lab[pred_id] == "SIMULTANEOUS":
                        labs.append(self.id2lab["AFTER"])
                    else:
                        labs.append(pred_id)
                # pred.extend(labs)
                pred.extend(torch.argmax(ab_logit, dim=-1).tolist())
            else:
                pred.extend(torch.argmax(ab_logit, dim=-1).tolist())
        prob = torch.cat(prob, dim=0).detach().cpu().numpy()
        if self.args.stage == "TB-Dense":
            labels = [idx for key, idx in self.lab2id.items() if key != 'SIMULTANEOUS']
            target_names = [key for key, idx in self.lab2id.items() if key != 'SIMULTANEOUS']
        else:
            labels = [idx for key, idx in self.lab2id.items() if key != 'VAGUE' and key != 'SIMULTANEOUS']
            target_names = [key for key, idx in self.lab2id.items() if key != 'VAGUE' and key != 'SIMULTANEOUS']

        acc = accuracy_score(y_true=np.array(gold), y_pred=np.array(pred))
        P, R, F1, s = precision_recall_fscore_support(y_true=np.array(gold), y_pred=np.array(pred),
                                                      average='micro', labels=labels)
        # report = classification_report(y_true=np.array(gold), y_pred=np.array(pred), labels=labels,
        #                                target_names=target_names, digits=5)
        exp_logger.info(f"{name}-ETR".center(20, "="))
        exp_logger.info("  Acc: {0:.9f}".format(round(acc, 5)))
        exp_logger.info("  P: {0:.9f}".format(round(P, 5)))
        exp_logger.info("  R: {0:.9f}".format(round(R, 5)))
        exp_logger.info("  F1: {0:.9f}".format(round(F1, 5)))
        exp_logger.info("  metrics report:")
        # exp_logger.info("\n" + report)

        ss = confusion_matrix(y_true=[self.id2lab[idx] for idx in gold], y_pred=[self.id2lab[idx] for idx in pred],
                              labels=[key for key, idx in self.lab2id.items()])
        exp_logger.info("\n" + "confusion matrix")
        for k in self.lab2id.values():
            p = ss[k][k] / (np.sum(ss[k, :]) + 1e-10)
            r = ss[k][k] / (np.sum(ss[:, k]) + 1e-10)
            f = 2 * p * r / (p + r + 1e-10)
            self.logger.info(f"{self.id2lab[k]}: P: {round(p, 5)}, R: {round(r, 5)}, F1: {round(f, 5)}")

        for item in ss.tolist():
            exp_logger.info(item)

        return F1

    def evaluate_modification(self, batch_data, model, name):
        model.eval()
        use_obj = tqdm(self.processor.iter_batch(batch_data, shuffle=False))
        pred_global, pred, gold, p_rule, g_rule, prob = [], [], [], [], [], []
        doc_ii, count, remember1, remember2 = {}, 0, {}, {}

        for step, batch in enumerate(use_obj, start=1):
            batch = [[item.to(self.device) for item in batch[i]] for i in range(len(batch) - 1)] + [batch[-1]]
            info = batch[-1]
            with torch.no_grad():
                ab_logit, ba_logit = model(batch, mode="inference")
            soft_prob_ab = F.softmax(ab_logit, dim=-1)
            soft_prob_ba = F.softmax(ba_logit, dim=-1)
            prob.append(soft_prob_ab)
            for k, (lab, doc_id, e_pair) in enumerate(zip(info["gold"], info["doc_id"], info["evt_id"])):
                gold.append(self.lab2id[lab[0]])
                lab_p = self.id2lab[torch.argmax(soft_prob_ab[k], dim=-1).tolist()]
                lab_p_rev = self.id2lab[torch.argmax(soft_prob_ba[k], dim=-1).tolist()]
                info_ = [f"{doc_id[0]}_{e_pair[0][0]}", f"{doc_id[0]}_{e_pair[0][1]}",
                         self.id2lab[torch.argmax(soft_prob_ab[k], dim=-1).tolist()],
                         self.id2lab[torch.argmax(soft_prob_ba[k], dim=-1).tolist()],
                         lab[0], soft_prob_ab.tolist()[k], soft_prob_ba.tolist()[k]]
                if doc_id[0] not in doc_ii:
                    doc_ii[doc_id[0]] = {}
                if (f"{doc_id[0]}_{e_pair[0][0]}", f"{doc_id[0]}_{e_pair[0][1]}") not in doc_ii[doc_id[0]]:
                    pair = (f"{doc_id[0]}_{e_pair[0][0]}", f"{doc_id[0]}_{e_pair[0][1]}")
                    doc_ii[doc_id[0]][pair] = [lab_p, lab_p_rev, lab[0], soft_prob_ab.tolist()[k],
                                               soft_prob_ba.tolist()[k], "forward"]
                    pair_rev = (f"{doc_id[0]}_{e_pair[0][1]}", f"{doc_id[0]}_{e_pair[0][0]}")
                    doc_ii[doc_id[0]][pair_rev] = [lab_p_rev, lab_p, self.rev_lab[lab[0]], soft_prob_ba.tolist()[k],
                                                   soft_prob_ab.tolist()[k], "backward"]

                lab_p_id = torch.argmax(soft_prob_ab[k], dim=-1).tolist()

                if self.args.hard:
                    # 对称性
                    lab_p_rev_id = torch.argmax(soft_prob_ba[k], dim=-1).tolist()
                    if lab_p != self.rev_lab[lab_p_rev]:
                        if lab_p == "VAGUE" and lab_p_rev == "AFTER" and \
                                info_[-2][lab_p_id] < info_[-1][lab_p_rev_id]:
                            lab_p = self.rev_lab[lab_p_rev]
                            lab_p_id = self.lab2id[lab_p]
                        if lab_p == "VAGUE" and lab_p_rev == "BEFORE" and \
                                info_[-2][lab_p_id] < info_[-1][lab_p_rev_id] and \
                                info_[-1][lab_p_rev_id] > 0.98:  # 0.98 -> VAGUE: 0.69595
                            lab_p = self.rev_lab[lab_p_rev]
                            lab_p_id = self.lab2id[lab_p]
                        if lab_p == "VAGUE" and lab_p_rev == "INCLUDES" and \
                                info_[-2][lab_p_id] < info_[-1][lab_p_rev_id]:
                            lab_p = self.rev_lab[lab_p_rev]
                            lab_p_id = self.lab2id[lab_p]
                    pred.append(lab_p_id)
                else:
                    lab_p_id = torch.argmax(soft_prob_ab[k], dim=-1).tolist()
                    pred.append(lab_p_id)
        prob = torch.cat(prob, dim=0).detach().cpu().numpy()
        labels = [idx for key, idx in self.lab2id.items() if key != 'SIMULTANEOUS']
        target_names = [key for key, idx in self.lab2id.items() if key != 'SIMULTANEOUS']
        acc = accuracy_score(y_true=np.array(gold), y_pred=np.array(pred))
        P, R, F1, s = precision_recall_fscore_support(y_true=np.array(gold), y_pred=np.array(pred),
                                                      average='micro', labels=labels)
        report = classification_report(y_true=np.array(gold), y_pred=np.array(pred), labels=labels,
                                       target_names=target_names, digits=5)
        exp_logger.info(f"{name}-ETR".center(20, "="))
        exp_logger.info("  Acc: {0:.9f}".format(round(acc, 5)))
        exp_logger.info("  P: {0:.9f}".format(round(P, 5)))
        exp_logger.info("  R: {0:.9f}".format(round(R, 5)))
        exp_logger.info("  F1: {0:.9f}".format(round(F1, 5)))
        exp_logger.info("  metrics report:")
        exp_logger.info("\n" + report)

        ss = confusion_matrix(y_true=[self.id2lab[idx] for idx in gold], y_pred=[self.id2lab[idx] for idx in pred],
                              labels=[key for key, idx in self.lab2id.items()])
        exp_logger.info("\n" + "confusion matrix")
        for item in ss.tolist():
            exp_logger.info(item)

        y_true = np.array(gold)
        one_hot = np.eye(len(self.lab2id))[y_true]
        auc_score = roc_auc_score(y_true=one_hot.ravel(), y_score=np.array(prob).ravel())
        exp_logger.info(f"\nAUC score:{auc_score}")

        return F1

    def prediction(self):
        self.PIPERModel.to(self.device)
        if self.args.stage == "backbone":
            model_param = torch.load(
                f"experiments/{self.args.task_name}/best-backbone-{args.encoder}-{args.feat_pair}-model.pt")
            self.PIPERModel.load_state_dict(model_param)  # 66.05
        elif self.args.stage == "symmetry":
            model_param = torch.load(
                # f"experiments/{self.args.task_name}/best-symmetry-{args.encoder}-{args.feat_pair}-model.pt")
                f"experiments/{self.args.task_name}/best-symmetry-{args.encoder}-{args.feat_pair}-0.0-0.78-model.pt")
            self.PIPERModel.load_state_dict(model_param)  # 66.80
        elif self.args.stage == "transitivity":
            model_param = torch.load(
                f"experiments/{self.args.task_name}/best-transitivity-{args.encoder}-{args.feat_pair}-model.pt")
            self.PIPERModel.load_state_dict(model_param)  # 67.44
        elif self.args.stage == "sym+trans":
            model_param = torch.load(
                f"experiments/{self.args.task_name}/best-sym+trans-{args.encoder}-{args.feat_pair}-model.pt")
            self.PIPERModel.load_state_dict(model_param)  # 67.44
        if self.args.stage not in ["transitivity", "sym+trans"]:
            self.evaluate(batch_data=self.test_batch_data, model=self.PIPERModel, name="test")
        else:
            self.evaluate_modification(batch_data=self.test_batch_data, model=self.PIPERModel, name="test")


if __name__ == '__main__':
    task_names = ["TB-Dense", "MATRES"]
    name = "TB-Dense"
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task_name', type=str, default=name,
                        choices=task_names)

    parser.add_argument('--output_path', type=str, default="experiments")
    parser.add_argument('--train_path', type=str, default=f"../sources/{name}/train.json")
    # parser.add_argument('--train_path', type=str, default=f"../sources/{name}/update_train.json")
    parser.add_argument('--devel_path', type=str, default=f"../sources/{name}/dev.json")
    parser.add_argument('--test_path', type=str, default=f"../sources/{name}/test.json")
    parser.add_argument('--embed_path', type=str, default="../sources/glove.6B.100d.txt")
    parser.add_argument('--bert_path', type=str, default="../bert/bert-base-uncased")
    # parser.add_argument('--bert_path', type=str, default="../bert/bert-large-uncased")

    # ["sigmoid", "tanh", "ReLU", "LeakyReLU", "RReLU", "softsign", "softplus", "GELU"]

    parser.add_argument('--encoder', type=str, default="BiGRU", choices=["BiGRU", "BiLSTM"])
    parser.add_argument('--feat_pair', type=str, default="GRU", choices=["GRU", "LSTM"])
    parser.add_argument('--activation_ru', type=str, default="LeakyReLU")
    # parser.add_argument('--activation_ru', type=str, default="sigmoid")
    # parser.add_argument('--activation_x', type=str, default="ReLU")
    parser.add_argument('--activation_x', type=str, default="sigmoid")
    parser.add_argument('--hard', type=bool, default=False)
    parser.add_argument('--stage', type=str, default="backbone",
                        choices=["backbone", "symmetry", "transitivity", "sym+trans", ])
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--drop_rate', type=float, default=0.5)
    parser.add_argument('--feat_list', type=list, default=["times", "avg", "pair_encode", "concat", "minus"])

    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--devel_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--warmup_proportion', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--tag_dim', type=int, default=50)
    parser.add_argument('--lab_dim', type=int, default=50)
    parser.add_argument('--dep_dim', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=2)

    parser.add_argument('--dis_wt', type=float, default=0.05)
    parser.add_argument('--sym_wt', type=float, default=1.0)  # symmetry: 1.0
    parser.add_argument('--trans_wt', type=float, default=0.78)

    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_devel', type=bool, default=True)
    parser.add_argument('--do_test', type=bool, default=True)

    args = parser.parse_args()

    os.makedirs('./log', exist_ok=True)
    log = Log(f'./log/{args.task_name}-{args.stage}-{args.encoder}-{args.feat_pair}' + ".log")
    exp_logger = log.getLog()

    params = [(key, value) for key, value in args.__dict__.items() if not key.startswith('__')]
    exp_logger.info("Hyper-parameter information".center(60, '+'))
    for key, value in params:
        exp_logger.info(f'{key}: {value}')

    Pipeline = PIPERMain(args=args, logger=exp_logger)
    Pipeline.run_train()
    # Pipeline.prediction()
