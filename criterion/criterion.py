import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from criterion import dict_to_device, beam_search_decode

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaCosLoss(nn.Module):
    def __init__(self, embedding_dim, num_classes, eps=1e-7):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)

        self.num_classes = num_classes
        self.eps = eps

        # Initial adaptive scale
        self.register_buffer(
            "s",
            torch.tensor(
                math.sqrt(2.0) *
                math.log(num_classes - 1)
            )
        )

        self.new_s = None

    def forward(self, embeddings, labels):
        """
        embeddings: [N, D]
        labels:     [N], integer class labels in [0, C-1]
        """
        # Normalize embeddings and class weights
        embeddings = F.normalize(
            embeddings, p=2, dim=1, eps=self.eps
        )
        weight = F.normalize(
            self.weight, p=2, dim=1, eps=self.eps
        )

        # Cosine similarities: [N, C]
        cosine = F.linear(embeddings, weight)
        cosine = cosine.clamp(-1.0 + self.eps,
                              1.0 - self.eps)

        # Calculate loss using the current scale
        logits = self.s * cosine
        loss = F.cross_entropy(logits, labels)

        # Update adaptive scale during training only
        if self.training:
            with torch.no_grad():
                batch_size = labels.numel()

                # Angles of the ground-truth classes
                target_cos = cosine.gather(
                    1, labels.view(-1, 1)
                ).squeeze(1)

                theta = torch.acos(target_cos)

                # Median target angle
                B_avg = torch.median(theta).item()

                # Sum exp(s * cosine) over non-target classes
                exp_logits = torch.exp(
                    self.s * cosine
                )

                target_mask = F.one_hot(
                    labels,
                    num_classes=self.num_classes
                ).bool()

                exp_logits = exp_logits.masked_fill(
                    target_mask, 0.0
                )

                B_avg_exp = exp_logits.sum(dim=1).mean()

                # Original AdaCos adaptive scaling rule
                if B_avg_exp.item() > 0:
                    self.new_s = (
                        torch.log(B_avg_exp)
                        / math.cos(min(
                            math.pi / 4.0, B_avg
                        ))
                    )

                    # if torch.isfinite(new_s):
                    #     self.s.copy_(
                    #         new_s.clamp(min=1.0, max=100.0)
                    #     )
        return loss
    def update_s(self):
        if self.new_s is not None:
            with torch.no_grad():
                self.s.copy_(self.new_s.detach())
            self.new_s = None

class FocalLoss(nn.Module):
    def __init__(self, num_class, reduction="mean", alpha=None, gamma=2):
        super().__init__()

        if alpha is None:
            self.register_buffer("alpha", torch.ones(num_class))
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float))

        self.gamma = gamma
        self.num_class = num_class
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        targets = targets.long()
        pt = probs.gather(1, targets.unsqueeze(-1)).squeeze(1)  
        log_pt = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(1)

        alpha_t = self.alpha[targets].to(inputs.device)

        batch_loss = -alpha_t * ((1 - pt) ** self.gamma) * log_pt
        if self.reduction == "mean":
            return batch_loss.mean()
        else:
            return batch_loss.sum()


class KWSCriterion(_Loss):
    def __init__(self, num_class, use_focalloss=False, loss_weight=[1, 1, 1, 1, 1], reduce="none"):
        super().__init__()

        self.loss_fn = FocalLoss(num_class=num_class, alpha=loss_weight, reduction=reduce) if use_focalloss \
            else nn.CrossEntropyLoss(weight=torch.tensor(loss_weight, dtype=torch.float32), reduction=reduce)
       
    def forward(self, model, sample, device):
        sample = dict_to_device(sample, device)

        results = model(**sample)
        
        loss = self.loss_fn(
            results["logits"],
            sample["classes"],
        )

        loss_register = loss.mean().item()

        with torch.no_grad():
            loss_mean = loss.mean().clamp_min(1e-8)
            dynamic_weights = (loss / loss_mean).clamp(0.5, 2)
        loss = (loss * dynamic_weights * sample["importance"]).mean()

        sample_size = results["logits"].size(0)
        max_indices = torch.argmax(results["logits"], dim=1)
        correct = (max_indices == sample["classes"]).sum()

        results = {
            "loss": loss,
            "loss_register": loss_register,
            "sample_size": sample_size,
            "correct": correct,
            "max_indices": max_indices,
            "classes": sample["classes"],
        }

        return results

class CTCKWSCriterion(_Loss):
    def __init__(self, reduce="none"):
        super().__init__()

        self.loss_fn = nn.CTCLoss(blank=0, reduction=reduce, zero_infinity=True)

        self.kw1 = torch.tensor([28])
        self.kw2 = torch.tensor([33])

    def forward(self, model, sample, device, reduce=True):
        sample = dict_to_device(sample, device)
        model = model.cuda(device)

        results = model(**sample)

        loss = 0

        joint_class = torch.zeros(results["logits"].shape[1], dtype=torch.long, device=device)
        joint_class[(sample["target_lengths"] == 1)] = 1
        loss += F.cross_entropy(
            input=results["binary_logits"],
            target=joint_class,
        )

        losses = self.loss_fn(
            log_probs=results["logits"],
            targets=sample["targets"],
            input_lengths= torch.full(size=(results["logits"].shape[1],), fill_value=results["logits"].shape[0], dtype=torch.long),
            target_lengths=sample["target_lengths"],
        )

        weights = losses.detach() / (losses.detach().mean() + 1e-8)
        weights = torch.clamp(weights, 0.5, 2.0)

        ctc_loss = (sample["importance"] * losses * weights).mean()
        loss += ctc_loss

        sample_size = results["logits"].size(1)
        correct = 0

        labels = None
        score = None
        if not self.training:
            res, scored = beam_search_decode(results["logits"], beam_size=1)
            labels = res
            score = scored
            for i, (label, gt) in enumerate(zip(labels, sample["labels"])):
                # if not sample["labels"][i] == [33] and not sample["labels"][i] ==  [28]:
                #     for i in range(results["logits"].shape[1]):
                #         plt.title(f'{sample["labels"][i]}')
                #         plt.imshow(results["logits"][:, i].detach().cpu().T)
                #         plt.show()
                if label == gt:
                    correct += 1
            results["correct"] = correct
            results["phonemes"] = labels
        results = {
            "term": None,
            "loss": loss,
            "loss_register": loss,
            "logits": results["logits"],
            "sample_size": sample_size,
            "correct": correct,
            "phonemes": labels,
            "score": score
        }

        return results

class CTCKWSCriterionPretrain(_Loss):

    def __init__(self):
        super().__init__()

        self.loss_fn = nn.MSELoss(
            reduction="none"
        )

        self.loss_fn_decoder = nn.CrossEntropyLoss(
            ignore_index=37,
            reduction="none"
        )

    def forward(self, model, sample, device, reduce=True):

        sample = dict_to_device(sample, device)

        results = model(**sample)

        # --------------------------------
        # Mel reconstruction loss
        # --------------------------------

        mel_loss = self.loss_fn(
            results["mel"],
            sample["x"],
        )

        # (B, Mel, T)
        # ↓
        # (B,)

        mel_loss = mel_loss.mean(dim=(1, 2))

        mel_loss = (
            sample["importance"] * mel_loss
        ).mean()

        # --------------------------------
        # Decoder loss
        # --------------------------------

        B, S = sample["decoder_target"].shape

        token_logits = results["token_logits"]

        # (B, S, V)
        # ↓
        # (B*S, V)

        decoder_loss = self.loss_fn_decoder(
            token_logits.reshape(-1, token_logits.shape[-1]),
            sample["decoder_target"].reshape(-1),
        )

        # (B*S,)
        # ↓
        # (B,S)

        decoder_loss = decoder_loss.reshape(B, S)

        # importance:
        # (B,)
        #
        # importance.unsqueeze(1):
        # (B,1)

        decoder_loss = (
            decoder_loss
            * sample["importance"].unsqueeze(1)
        )

        decoder_loss = decoder_loss.mean()

        # --------------------------------
        # Total
        # --------------------------------

        loss = mel_loss + decoder_loss

        return {
            "loss": loss,
        }

class SSLKWSCriterion(_Loss):

    def __init__(self):
        super().__init__()

        self.loss_fn = nn.MSELoss(
            reduction="none"
        )

    def forward(self, model, sample, device, reduce=True):

        sample = dict_to_device(sample, device)

        results = model(**sample)

        # --------------------------------
        # Mel reconstruction loss
        # --------------------------------

        mel_loss = self.loss_fn(
            results["out"],
            sample["targets"],
        )

        # (B, Mel, T)
        # ↓
        # (B,)

        mel_loss = mel_loss.mean(dim=(1, 2))

        mel_loss = (
            sample["importance"] * mel_loss
        ).mean()

        loss = mel_loss

        return {
            "loss": loss,
            "loss_register": loss
        }
    
class TDTKWSCriterion(_Loss):
    def __init__(self, reduce="mean"):
        super().__init__()

        self.loss_fn = nn.CTCLoss(blank=0, reduction=reduce, zero_infinity=True)

    def forward(self, model, sample, device, reduce=True):
        sample = dict_to_device(sample, device)

        results = model(x=sample["x"], tokens=sample["targets"], target_lengths=sample["target_lengths"])

        correct = 0
        
        results = {
            "loss": results["loss"],
            "alpha": results["alpha"],
            "correct": correct,
        }

        return results
