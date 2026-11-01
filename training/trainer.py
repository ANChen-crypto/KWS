import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import MelSpecDataset, CTCMelSpecDataset, CTCMelSpecNpzDataset, CTCMelSpecNoisyNpzDataset, keyword_ids2class, SSLMelSpecNpzDataset
from modeling import LSTM, ConformerForKWS, ConformerForCTCKWS, TDT, CNN, SSLCNN, ConformerForCTCKWSPretrain, NTCGraph
from criterion import KWSCriterion, CTCKWSCriterion, beam_search_decode, TDTKWSCriterion, CTCKWSCriterionPretrain, SSLKWSCriterion

import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

from modeling.conformer import ConformerForCTCKWSPretrain


with open("training/train_cfg.yaml", "r", encoding="utf-8") as f:
    spec = yaml.safe_load(f)

MODEL_WEIGHT_PATH = {
    "cnn_ctc": "./weights/cnn_ctc.pth",
    "cnn_ssl": "./weights/cnn_ssl.pth",
    "lstm": "./weights/lstm.pth",
    "conformer": "./weights/conformer_ce_not_best.pth",
    "conformer_ctc": "./weights/conformer_ctc_0.05.pth",
    "conformer_ctc_pretrain": "./weights/conformer_ctc_pretrain.pth",
    "tdt": "./weights/tdt.pth",
}
MODEL_FACTORY = {
    "cnn_ctc": lambda: CNN(input_dim=35),
    "cnn_ssl": lambda: SSLCNN(input_dim=35),
    "lstm": lambda: LSTM(),
    "conformer": lambda: ConformerForKWS(
        num_classes=spec["num_classes"],
        input_dim = 40,
        encoder_dim = 64,
        num_layers = 3,
        num_attention_heads = 8,
        feed_forward_expansion_factor = 4,
        conv_expansion_factor = 2,
        input_dropout_p = 0.1,
        feed_forward_dropout_p = 0.1,
        attention_dropout_p = 0.1,
        conv_dropout_p = 0.1,
        conv_kernel_size = 31,
        half_step_residual = True,
    ),
    "conformer_ctc": lambda: ConformerForCTCKWS(
        input_dim = 35,
        encoder_dim = 64,
        num_layers = 3,
        num_attention_heads = 4,
        feed_forward_expansion_factor = 4,
        conv_expansion_factor = 2,
        input_dropout_p = 0.1,
        feed_forward_dropout_p = 0.1,
        attention_dropout_p = 0.1,
        conv_dropout_p = 0.1,
        conv_kernel_size = 31,
        half_step_residual = True,
    ),
    "conformer_ctc_pretrain": lambda: ConformerForCTCKWSPretrain(
        input_dim = 40,
        encoder_dim = 64,
        num_layers = 3,
        num_attention_heads = 4,
        feed_forward_expansion_factor = 4,
        conv_expansion_factor = 2,
        input_dropout_p = 0.1,
        feed_forward_dropout_p = 0.1,
        attention_dropout_p = 0.1,
        conv_dropout_p = 0.1,
        conv_kernel_size = 31,
        half_step_residual = True,
    ),
    "tdt": lambda: TDT(
        vocab_size = 37,
        max_duration = 5,
        max_target_length = 8,  
        input_dim = 40,
        num_layers = 4,
    ),
}
CRITERION_FACTORY = {
    "cnn_ctc": lambda: CTCKWSCriterion(),
    "cnn_ssl": lambda: SSLKWSCriterion(),
    "lstm": lambda: KWSCriterion(spec["num_classes"], loss_weight=spec["loss_weight"]),
    "conformer_ctc_pretrain": lambda: CTCKWSCriterionPretrain(),
    "conformer_ctc": lambda: CTCKWSCriterion(),
}

class Trainer(object):
    def __init__(
        self,
        mode,
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.input_type = spec["input_type"]
        self.num_classes = spec["num_classes"]

        self.epoch = spec["epoch"]
        self.model = MODEL_FACTORY[self.input_type]()
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        self.opt = optim.AdamW(self.model.parameters(), lr=spec["learning_rate"], weight_decay=0.01)
        self.criterion = CRITERION_FACTORY[self.input_type]()

        self.stamp_every = spec["stamp_every"]
        self.save_every = spec["save_every"]

        self.weight_path = MODEL_WEIGHT_PATH[self.input_type]

    def _build_loader(self, mode):

            if mode == 'train':
                if self.input_type in ["lstm", "cnn"]:
                    self.train_ds = MelSpecDataset(True)
                elif self.input_type in ["cnn_ssl"]:
                    self.train_ds = SSLMelSpecNpzDataset(True)
                else:
                    self.train_ds = CTCMelSpecNpzDataset(True)
                shuffle = True
                batch_size = spec["batch_size"]
                self.train_collator = self.train_ds.collator
                self.train_dl = DataLoader(
                    self.train_ds,
                    collate_fn=self.train_collator,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=6,
                    pin_memory=False,
                    persistent_workers=False,
                    prefetch_factor=1
                )
                print("==========================================")
                print(f"Train size: {len(self.train_dl)}")
                print("==========================================")

            if self.input_type in ["lstm", "cnn"]:
                self.test_ds = MelSpecDataset(False)
            elif self.input_type in ["cnn_ssl"]:
                self.test_ds = SSLMelSpecNpzDataset(False)
            else:
                self.test_ds = CTCMelSpecNpzDataset(False)
            shuffle = False
            batch_size = spec["batch_size"]

            self.test_collator = self.test_ds.collator
            self.test_dl = DataLoader(
                self.test_ds,
                collate_fn=self.test_collator,
                batch_size=batch_size if mode == "train" else 1,
                shuffle=shuffle,
                num_workers=0,
            )
            print("==========================================")
            print(f"Test size: {len(self.test_dl)}")
            print("==========================================")

    def train(self, weight_path):
        self._build_loader("train")
        best_val_loss = 1e10
        train_loss_history = []
        test_loss_history = []
        acc_history = []

        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        if weight_path:
            ckpt = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"], strict=True)
            self.opt.load_state_dict(ckpt["optimizer"])
            best_val_loss = ckpt["best_val_loss"]
            print(f"ckpt best loss", best_val_loss)

        for epoch in range(self.epoch):
            self.model.train()
            total_training_loss = 0
            total_testing_loss = 0
            pbar = tqdm(self.train_dl, desc="Train")
            for step, sample in enumerate(pbar):
                results = self.criterion(self.model, sample, self.device)

                train_loss = results["loss"]
                self.opt.zero_grad()
                train_loss.backward()
                self.opt.step()
                torch.cuda.empty_cache()

                total_training_loss += results["loss_register"]

                if (step + 1) % 10 == 0:
                    pbar.set_postfix(step=step, loss=f"{(total_training_loss / (step + 1)):.4f}")

                if (step + 1) % 10000 == 0:
                    total_correct = 0
                    total_val_loss = 0
                    total_sample_size = 0
                    with torch.no_grad():
                        for step, sample in enumerate(self.test_dl):
                            results = self.criterion(self.model.eval(), sample, self.device)
        
                            total_val_loss += results["loss_register"]
                            total_correct += results["correct"]
                            total_sample_size += results["sample_size"]
                    self.criterion.train()

                    if total_val_loss / len(self.test_dl) < best_val_loss:
                        ckpt = {
                            "model": self.model.state_dict(),
                            "best_val_loss": total_val_loss / len(self.test_dl),
                            "optimizer": self.opt.state_dict(),
                            "epoch": epoch,
                            "accuracy": total_correct / total_sample_size
                        }
                        torch.save(ckpt, self.weight_path)
                        best_val_loss = total_val_loss / len(self.test_dl)
                    torch.save(ckpt, "./weights/cnn_ctc_not_best.pth")
                    print(f"Epoch: {epoch + 1} / {self.epoch}\
                          Val loss: {total_val_loss / len(self.test_dl):.4f}\
                          Acc: {total_correct / total_sample_size:.4f}")

            total_correct = 0
            total_sample_size = 0
            self.criterion.eval()
            with torch.no_grad():
                for step, sample in enumerate(self.test_dl):
                    results = self.criterion(self.model.eval(), sample, self.device)

                    total_testing_loss += results["loss_register"]
                    total_correct += results["correct"]
                    total_sample_size += results["sample_size"]
            self.criterion.train()

            if (epoch + 1) % self.stamp_every == 0:
                acc_history.append(total_correct / total_sample_size)
                train_loss_history.append(total_training_loss / len(self.train_dl))
                test_loss_history.append(total_testing_loss / len(self.test_dl))
                print(f"Epoch: {epoch + 1} / {self.epoch}\
      Train loss: {total_training_loss / len(self.train_dl):.4f}\
      Test loss: {total_testing_loss / len(self.test_dl):.4f}")
                    
        plt.plot(train_loss_history)
        plt.plot(test_loss_history)
        plt.show()

    def train_ssl(self, weight_path):
        self._build_loader("train")
        best_val_loss = 1e10
        train_loss_history = []
        test_loss_history = []
        acc_history = []

        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        if weight_path:
            ckpt = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"], strict=True)
            self.opt.load_state_dict(ckpt["optimizer"])
            best_val_loss = ckpt["best_val_loss"]
            print(f"ckpt best loss", best_val_loss)

        for epoch in range(self.epoch):
            self.model.train()
            total_training_loss = 0
            total_testing_loss = 0
            pbar = tqdm(self.train_dl, desc="Train")
            for step, sample in enumerate(pbar):
                results = self.criterion(self.model, sample, self.device)

                train_loss = results["loss"]
                self.opt.zero_grad()
                train_loss.backward()
                self.opt.step()
                torch.cuda.empty_cache()

                total_training_loss += results["loss_register"]

                if (step + 1) % 10 == 0:
                    pbar.set_postfix(step=step, loss=f"{(total_training_loss / (step + 1)):.4f}")

                if (step + 1) % 40000 == 0:
                    total_correct = 0
                    total_val_loss = 0
                    total_sample_size = 0
                    with torch.no_grad():
                        for step, sample in enumerate(self.test_dl):
                            results = self.criterion(self.model.eval(), sample, self.device)
        
                            total_val_loss += results["loss_register"]
                    self.criterion.train()

                    if total_val_loss / len(self.test_dl) < best_val_loss:
                        ckpt = {
                            "model": self.model.state_dict(),
                            "best_val_loss": total_val_loss / len(self.test_dl),
                            "optimizer": self.opt.state_dict(),
                            "epoch": epoch,
                        }
                        torch.save(ckpt, self.weight_path)
                        best_val_loss = total_val_loss / len(self.test_dl)
                    torch.save(ckpt, "./weights/cnn_ctc_not_best.pth")
                    print(f"Epoch: {epoch + 1} / {self.epoch}\
                          Val loss: {total_val_loss / len(self.test_dl):.4f}")

            total_correct = 0
            total_sample_size = 0
            self.criterion.eval()
            with torch.no_grad():
                for step, sample in enumerate(self.test_dl):
                    results = self.criterion(self.model.eval(), sample, self.device)

                    total_testing_loss += results["loss_register"]
            self.criterion.train()

            if (epoch + 1) % self.stamp_every == 0:
                train_loss_history.append(total_training_loss / len(self.train_dl))
                test_loss_history.append(total_testing_loss / len(self.test_dl))
                print(f"Epoch: {epoch + 1} / {self.epoch}\
      Train loss: {total_training_loss / len(self.train_dl):.4f}\
      Test loss: {total_testing_loss / len(self.test_dl):.4f}")
                    
        plt.plot(train_loss_history)
        plt.plot(test_loss_history)
        plt.show()


    def test(self):
        self._build_loader("test")
        ckpt = torch.load(self.weight_path, map_location="cpu")
        print("epoch", ckpt["epoch"])
        print("best loss", ckpt["best_val_loss"])
        self.model.load_state_dict(torch.load(self.weight_path, map_location="cpu")["model"])
        self.model = self.model.to(self.device)
        self.model.eval()
        self.criterion = self.criterion.to(self.device)
        self.criterion.eval()
        
        cm = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int32)
        pbar = tqdm(self.test_dl, desc="Test")

        with torch.no_grad():
            for step, sample in enumerate(pbar):
                results = self.criterion(self.model, sample, self.device)
                for j, (t, p) in enumerate(zip(results["max_indices"], results["classes"])):
                    cm[int(t), int(p)] += 1

                    # if (not t == p):
                    #     idx = 0
                    #     while True:
                    #         error_path = f"./error files/class_{p}_{idx:03d}.wav"
                    #         if not os.path.exists(error_path):
                    #             sf.write(error_path, sample["audio"][j].squeeze(), 16000)
                    #             break
                    #         idx += 1
        
        plt.figure(figsize=(6,5))
        plt.imshow(cm, origin="lower", cmap="Blues")
        plt.ylabel("Predict")
        plt.xlabel("Ground truth")
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i,
                    f"{cm[i, j]}",
                    ha="center", va="center",
                    color="black"
                )
        # plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
        #            ["unk", "help_hok", "help_zh", "help_en", "need_hok", "need_zh", "need_en", "no_need_hok", "no_need_zh", "no_need_en", "fell_hok", "fell_zh", "fell_en"])
        # plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
        #            ["unk", "help_hok", "help_zh", "help_en", "need_hok", "need_zh", "need_en", "no_need_hok", "no_need_zh", "no_need_en", "fell_hok", "fell_zh", "fell_en"])
        plt.xticks([0, 1, 2, 3, 4], 
                    ["help_hok", "need_hok", "no_need_hok", "fell_hok", "noise"])
        plt.yticks([0, 1, 2, 3, 4], 
                    ["help_hok", "need_hok", "no_need_hok", "fell_hok", "noise"])
        plt.show()

    def test_ctc(self, weight_path):
        self._build_loader("test")

        self.weight_path = weight_path if weight_path is not None else self.weight_path

        ckpt = torch.load(self.weight_path, map_location="cpu")
        print("weight path: ", self.weight_path)
        print("best loss", ckpt["best_val_loss"])
        print("epoch", ckpt["epoch"])
        self.model.load_state_dict(ckpt["model"])
        self.model = self.model.to(self.device)
        self.model.eval()
        self.criterion = self.criterion.to(self.device)
        self.criterion.eval()
        
        pbar = tqdm(self.test_dl, desc="Test")
        undetected = 0

        all_scores = []
        all_gt = []
        all_pred = []
        with torch.no_grad():
            for step, sample in enumerate(pbar):
                results = self.criterion(self.model, sample, self.device)
                for i, (gt_p, pred_p, score) in enumerate(zip(sample["labels"], results["phonemes"], results["score"])):

                    gt_p2class = keyword_ids2class[tuple(gt_p)]
                    try:
                        pred_p2class = keyword_ids2class[tuple(pred_p)]
                    except Exception as e:
                        undetected += 1
                        pred_p2class = 4

                    all_gt.append(gt_p2class)
                    all_pred.append(pred_p2class)
                    all_scores.append(float(score))

        all_gt = np.asarray(all_gt, dtype=np.int32)
        all_pred = np.asarray(all_pred, dtype=np.int32)
        all_scores = np.asarray(all_scores, dtype=np.float32)

        thresholds = np.linspace(-6, 0, 151)

        num_classes = 5
        unk_id = 4

        results_by_threshold = []

        positive = all_gt != unk_id
        negative = all_gt == unk_id

        for threshold in thresholds:

            pred = np.where(all_scores >= threshold, all_pred, unk_id)

            n_negative = np.sum(negative)
            false_alarm = np.sum(negative & (pred != unk_id))

            far = (
                false_alarm / n_negative
                if n_negative else np.nan
            )

            n_positive = np.sum(positive)
            false_rejection = np.sum(positive & (pred != all_gt))

            frr = (
                false_rejection / n_positive
                if n_positive else np.nan
            )

            keyword_confusion = np.sum(
                positive & (pred != unk_id) & (pred != all_gt)
            )

            cm = confusion_matrix(
                all_gt,
                pred,
                labels=np.arange(num_classes)
            )

            results_by_threshold.append({
                "threshold": threshold,
                "far": far,
                "frr": frr,
                "false_alarm": false_alarm,
                "false_rejection": false_rejection,
                "keyword_confusion": keyword_confusion,
                "cm": cm,
            })

        os.makedirs("cm_ctc", exist_ok=True)
        for name in os.listdir("cm_ntc"):
            path = os.path.join("cm_ntc", name)
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
        class_names = ["help", "fall", "need", "no_need", "unk"]

        for r in results_by_threshold:
            cm = r["cm"]
            threshold = r["threshold"]
            far = r["far"]
            frr = r["frr"]

            fig, ax = plt.subplots(figsize=(8, 6))

            im = ax.imshow(cm, origin="lower", cmap="Blues")

            ax.set_title(
                f"CTC Confusion Matrix\n"
                f"Threshold={threshold:.3f}, "
                f"FAR={far:.2%}, FRR={frr:.2%}"
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Ground Truth")

            ticks = np.arange(num_classes)
            ax.set_xticks(ticks, class_names, rotation=30)
            ax.set_yticks(ticks, class_names)

            fig.colorbar(im, ax=ax)

            for i in range(num_classes):
                for j in range(num_classes):
                    ax.text(
                        j, i, str(cm[i, j]),
                        ha="center",
                        va="center",
                        color="black"
                    )

            fig.tight_layout()
            fig.savefig(
                f"cm_ctc/threshold_{threshold:.3f}_FAR_{far:.4f}_FRR_{frr:.4f}.png",
                dpi=200,
                bbox_inches="tight"
            )
            plt.close(fig)

        far_values = np.array([r["far"] for r in results_by_threshold])
        frr_values = np.array([r["frr"] for r in results_by_threshold])

        np.savez("ctc_frr_far_curve.npz", far=far_values, frr=frr_values)

        plt.figure(figsize=(7, 5))
        plt.plot(far_values * 100, frr_values * 100, marker=".")
        plt.xlabel("FAR (%)")
        plt.ylabel("FRR (%)")
        plt.title("FAR vs FRR")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def test_ntc(self, weight_path):
            self._build_loader("test")

            self.weight_path = weight_path if weight_path is not None else self.weight_path
    
            ckpt = torch.load(self.weight_path, map_location="cpu")
            print("weight path: ", self.weight_path)
            print("best loss", ckpt["best_val_loss"])
            print("epoch", ckpt["epoch"])
            self.model.load_state_dict(ckpt["model"])
            self.model.eval()
            
            pbar = tqdm(self.test_dl, desc="Test")
            cm = np.zeros((5, 5), dtype=np.int32)

            self.ntc_graphs = []
            for keyword_tokens_tuple, keyword_id in keyword_ids2class.items():
                if list(keyword_tokens_tuple) not in [
                    [20, 17, 22, 12, 24, 11],
                    [22, 5, 13],
                    [27, 30, 14],
                    [10, 31, 26, 32, 29, 24, 1],
                ]:
                    continue
                ntc_graph = NTCGraph(keyword_id, 0, -4, -4)
                ntc_graph.graph_add_keyword(list(keyword_tokens_tuple))
                self.ntc_graphs.append(ntc_graph)

            print("\n****************************************")
            print(f"{len(self.ntc_graphs)} NTC graphs build")
            print("\n****************************************")

            total = 0

            all_scores = []
            all_labels = []

            with torch.no_grad():
                for step, sample in enumerate(pbar):
                    out = self.model(sample["x"])["logits"]
                    out = out.squeeze(1).numpy()
                    gt_p = sample["labels"][0]
                    gt_p2class = keyword_ids2class[tuple(gt_p)]
                    scores = []
                    for g in self.ntc_graphs:
                        g_result = g.graph_decode(out, threshold=-1e30)
                        if g_result["accepted"]:
                            scores.append(g_result["best_score"])
                        else:
                            scores.append(float("-inf"))

                    all_scores.append(scores)
                    all_labels.append(gt_p2class)

                    total += 1

            all_scores = np.array(all_scores, dtype=np.float32)
            all_labels = np.array(all_labels)

            max_scores = np.max(all_scores, axis=1)     # b,
            best_ids = np.argmax(all_scores, axis=1)    # b,

            positive = all_labels != 4
            negative = all_labels == 4

            results = []
            thresholds = np.linspace(-6, 0, 151)
            os.makedirs("cm_ntc", exist_ok=True)
            for name in os.listdir("cm_ntc"):
                path = os.path.join("cm_ntc", name)
                if os.path.isfile(path) or os.path.islink(path):
                    os.remove(path)
            for th in thresholds:
                accepted = max_scores >= th
                pred = np.where(accepted, best_ids, 4)

                far = (
                    np.sum(negative & accepted) / np.sum(negative)
                )

                frr = (
                    np.sum(positive & (pred != all_labels)) / np.sum(positive)
                )

                confusion_count = np.sum(
                    positive & (pred != 4) & (pred != all_labels)
                )

                cm = confusion_matrix(
                    all_labels,
                    pred,
                    labels=np.arange(5)
                )

                plt.figure(figsize=(8, 6))
                plt.imshow(cm, origin="lower", cmap="Blues")
                plt.title(
                    f"Confusion Matrix\n"
                    f"Threshold={th:.3f}, "
                    f"FAR={far:.2%}, FRR={frr:.2%}"
                )
                plt.xlabel("Predict")
                plt.ylabel("Ground Truth")

                class_names = ["help", "fall", "need", "no_need", "unk"]
                ticks = np.arange(5)

                plt.xticks(ticks, class_names, rotation=30)
                plt.yticks(ticks, class_names)
                plt.colorbar()

                for i in range(5):
                    for j in range(5):
                        plt.text(
                            j, i, str(cm[i, j]),
                            ha="center",
                            va="center",
                            color="black"
                        )

                plt.tight_layout()

                plt.savefig(
                    f"cm_ntc/confusion_threshold_{th:.3f}_FAR_{far:.4f}_FRR_{frr:.4f}.png",
                    dpi=200,
                    bbox_inches="tight"
                )
                plt.close()

                results.append([th, far, frr])

            for target_far in [0.01, 0.05, 0.1]:
                valid = [r for r in results if r[1] <= target_far]
                
                if not valid:
                    continue
                
                best = min(valid, key=lambda r: r[2])
                
                print(
                    f"Target FAR <= {target_far:.0%}: "
                    f"threshold={best[0]:.3f}, "
                    f"actual FAR={best[1]:.2%}, "
                    f"FRR={best[2]:.2%}"
                )

            far_values = np.array([r[1] for r in results])
            frr_values = np.array([r[2] for r in results])

            np.savez("ntc_frr_far_curve.npz", far=far_values, frr=frr_values)

            plt.figure(figsize=(7, 5))
            plt.plot(far_values * 100, frr_values * 100, marker=".")
            plt.xlabel("FAR (%)")
            plt.ylabel("FRR (%)")
            plt.title("FAR vs FRR")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def test_tdt(self):

        print("best loss", torch.load(self.weight_path, map_location="cpu")["best_val_loss"])
        # print("time", torch.load(self.weight_path, map_location="cuda:0")["time"])
        state = torch.load(self.weight_path, map_location="cpu")["model"]
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.criterion = self.criterion.to(self.device)
        self.criterion.eval()
        
        pbar = tqdm(self.test_dl, desc="Test")
        cm = np.zeros((self.num_classes+1, self.num_classes+1), dtype=np.int32)
        undetected = 0
        with torch.no_grad():
            for step, sample in enumerate(pbar):

                gt_p = sample["labels"][0]

                encoder_out, _ = self.model.encoder(sample["x"].to(self.device))

                T = encoder_out.shape[1]

                t = 0
                u = 0

                output = []
                tokens = []
                total_score = 0.0
                count = 0
                path = torch.zeros(T, self.model.vocab_size, device=self.device)

                while t < T:
                    
                    tokens_embedding = self.model.predictor(torch.tensor(tokens, device=self.device).unsqueeze(0))
                    token_logits, dur_logits = self.model.joint(encoder_out[:, t].unsqueeze(1), tokens_embedding)

                    label_prob = torch.log_softmax(token_logits[0, 0, -1], dim=-1)                  # (B, T, U, V)
                    duration_prob = torch.log_softmax(dur_logits[0, 0, -1], dim=-1)                 # (B, T, U, D)

                    best_score = torch.finfo(torch.float32).min
                    best_token = None
                    best_l = None

                    for k in range(label_prob.size(-1)):
                        for n, l in enumerate(self.model.durations):

                            if l <= 0:
                                continue

                            score = (label_prob[..., k] + duration_prob[..., n]).item()
                            
                            if score > best_score:
                                best_score = score
                                best_token = k
                                best_l = l.item()

                    for dt in range(best_l):
                        if t+dt < T:
                            path[t, best_token] = 1   
                    
                    if best_token == self.model.blank:
                        t = t + best_l
                    else:
                        output.append(best_token)
                        tokens.append(best_token)
                        t = t + best_l
                        u = u + 1
                        total_score += best_score
                        count += 1

                plt.imshow(path.cpu().T)
                plt.show()

                gt_p2class = keyword_ids2class[tuple(gt_p)]

                if count > 0:
                    avg_logprob = total_score / count
                else:
                    avg_logprob = -1e9

                if avg_logprob > -2.5:
                    try:
                        pred_p2class = keyword_ids2class[tuple(tokens)]
                        cm[pred_p2class, gt_p2class] += 1
                    except:
                        cm[-1, gt_p2class] += 1
                        undetected += 1
                else:
                    cm[-1, gt_p2class] += 1
                    undetected += 1

            plt.imshow(cm, origin="lower", cmap="Blues")
            plt.title(f"Undetect: {undetected}")
            plt.ylabel("Predict")
            plt.xlabel("Ground truth")
            plt.colorbar()
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(
                        j, i,
                        f"{cm[i, j]}",
                        ha="center", va="center",
                        color="black"
                    )
            plt.xticks([0, 1, 2, 3, 4, 5, 6], ["help", "fell", "need", "no need", "noise", "unk", "undetected"])
            plt.yticks([0, 1, 2, 3, 4, 5, 6], ["help", "fell", "need", "no need", "noise", "unk", "undetected"])
            plt.show()

    def test_ctc_noisy(self, wav_path):
        self.test_ds = CTCMelSpecNoisyNpzDataset(wav_path)

        self.test_collator = self.test_ds.collator
        self.test_dl = DataLoader(
            self.test_ds,
            collate_fn=self.test_collator,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        ckpt = torch.load(r"weights\conformer_ctc.pth", map_location="cpu")
        print("weight path: ", self.weight_path)
        print("best loss", ckpt["best_val_loss"])
        # print("time", ckpt["time"])
        print("epoch", ckpt["epoch"])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.criterion = self.criterion.to(self.device)
        self.criterion.eval()
        
        pbar = tqdm(self.test_dl, desc="Test")
        cm = np.zeros((self.num_classes+1, self.num_classes+1), dtype=np.int32)
        undetected = 0
        count = 0
        with torch.no_grad():
            for step, sample in enumerate(pbar):
                out = self.model(sample["x"])
                labels = beam_search_decode(out["logits"], beam_size=2)[0][0]
                print(labels)
                if labels == [20, 17, 22, 12, 24, 11]:
                    count += 1
        
        # plt.imshow(cm, origin="lower", cmap="Blues")
        # plt.title(f"Undetect: {undetected}")
        # plt.ylabel("Predict")
        # plt.xlabel("Ground truth")
        # plt.colorbar()
        # for i in range(cm.shape[0]):
        #     for j in range(cm.shape[1]):
        #         plt.text(
        #             j, i,
        #             f"{cm[i, j]}",
        #             ha="center", va="center",
        #             color="black"
        #         )
        # plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
        #             ["unk", "help_hok", "help_zh", "help_en", "need_hok", "need_zh", "need_en", "no_need_hok", "no_need_zh", "no_need_en", "fell_hok", "fell_zh", "fell_en"])
        # plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
        #             ["unk", "help_hok", "help_zh", "help_en", "need_hok", "need_zh", "need_en", "no_need_hok", "no_need_zh", "no_need_en", "fell_hok", "fell_zh", "fell_en"])
        # plt.show()

        print(count)

    def test_streaming(self, wav_path, weight_path=None, score_threshold=-6.0):

        self.test_ds = CTCMelSpecNoisyNpzDataset(wav_path)
        self.test_dl = DataLoader(
            self.test_ds,
            collate_fn=self.test_ds.collator,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        self.weight_path = weight_path if weight_path is not None else self.weight_path
        ckpt = torch.load(self.weight_path, map_location="cpu")
        print("weight path: ", self.weight_path)
        print("best loss", ckpt["best_val_loss"])
        print("epoch", ckpt["epoch"])
        self.model.load_state_dict(ckpt["model"])
        self.model = self.model.to(self.device)
        self.model.eval()

        class_names = {
            0: "help",
            1: "fall",
            2: "need",
            3: "no need",
            4: "noise",
        }
        class_colors = {
            0: "tab:blue",
            1: "tab:orange",
            2: "tab:green",
            3: "tab:red",
            4: "lightgray",
        }
        predicted_classes = []

        with torch.no_grad():
            for sample in tqdm(self.test_dl, desc="Streaming"):
                output = self.model(sample["x"].to(self.device))
                predictions, scores = beam_search_decode(
                    output["logits"], beam_size=1
                )

                predicted_class = keyword_ids2class.get(
                    tuple(predictions[0]), 4
                )
                if scores[0] < score_threshold:
                    predicted_class = 4
                predicted_classes.append(predicted_class)

        audio_duration = len(self.test_ds.wav) / self.test_ds.sampling_rate
        step_duration = int(self.test_ds.sampling_rate * 0.2) / self.test_ds.sampling_rate
        window_duration = self.test_ds.segs.shape[1] / self.test_ds.sampling_rate
        spans_by_class = {class_id: [] for class_id in class_names}
        spans_by_class[4].append((0, min(window_duration, audio_duration)))
        run_start = 0

        for step in range(1, len(predicted_classes) + 1):
            if (
                step == len(predicted_classes)
                or predicted_classes[step] != predicted_classes[run_start]
            ):
                class_id = predicted_classes[run_start]
                start = window_duration + run_start * step_duration
                end = min(window_duration + step * step_duration, audio_duration)
                if end > start:
                    spans_by_class[class_id].append((start, end - start))
                run_start = step

        fig, ax = plt.subplots(figsize=(16, 5))
        for class_id, spans in spans_by_class.items():
            if spans:
                ax.broken_barh(
                    spans,
                    (class_id - 0.4, 0.8),
                    facecolors=class_colors[class_id],
                    label=class_names[class_id],
                )

        ax.set_xlim(0, audio_duration)
        ax.set_ylim(-0.5, len(class_names) - 0.5)
        ax.set_yticks(
            list(class_names),
            [class_names[class_id] for class_id in class_names],
        )
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Predictions")
        ax.set_title(
            f"Streaming timeline(Trigger threshold:{score_threshold:.1f})"
        )
        ax.grid(axis="x", alpha=0.3)
        ax.legend(loc="upper right")
        fig.tight_layout()
        os.makedirs("streaming_timeline", exist_ok=True)
        output_path = (
            f"streaming_timeline/streaming_timeline-th_{score_threshold}"
            f"-id_{ckpt['best_val_loss']:.4f}.png"
        )
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Streaming timeline saved to {output_path}")
        plt.show()
        plt.close(fig)

    def compare_curve(self):
        ctc_frr_far_curve_cnn1 = np.load("ctc_frr_far_curve-cnn-0.1186.npz")
        ctc_frr_far_curve_cnn2 = np.load("ctc_frr_far_curve-cnn-0.1196.npz")
        ctc_frr_far_curve_cnn3 = np.load("ctc_frr_far_curve-cnn-0.1124.npz")
        ctc_frr_far_curve_cnn4 = np.load("ctc_frr_far_curve-cnn-0.0921.npz")
        plt.plot(ctc_frr_far_curve_cnn1["far"], ctc_frr_far_curve_cnn1["frr"], label="CNN-CTC 0.1186 FRR@FAR", linewidth=2, marker=".")
        plt.plot(ctc_frr_far_curve_cnn2["far"], ctc_frr_far_curve_cnn2["frr"], label="CNN-CTC 0.1196 FRR@FAR", linewidth=2, marker=".")
        plt.plot(ctc_frr_far_curve_cnn3["far"], ctc_frr_far_curve_cnn3["frr"], label="CNN-CTC 0.1124 FRR@FAR", linewidth=2, marker=".")
        plt.plot(ctc_frr_far_curve_cnn4["far"], ctc_frr_far_curve_cnn4["frr"], label="CNN-CTC 0.0921 FRR@FAR", linewidth=2, marker=".")
        plt.xlabel("FAR")
        plt.ylabel("FRR")
        plt.grid(True)
        plt.legend()
        plt.show()