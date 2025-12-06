from dataclasses import dataclass, field

import torch
import torch.nn as nn

from transformers import WavLMForSequenceClassification


@ dataclass
class KMSModelConfig:
    num_classes: int = field(
        default=5, metadata={"help": "number of classes for kws model"}
    )
    freeze_backbone: bool = field(
        default=True, metadata={"help": "freeze WavLM's parameters"}
    )

class KWSModel(nn.Module):
    def __init__(self, cfg=KMSModelConfig()):
        super().__init__()

        self.wavlm = WavLMForSequenceClassification.from_pretrained(
            "microsoft/wavlm-base",
            num_labels=cfg.num_classes,
            problem_type="multi_label_classification"
        )
        if cfg.freeze_backbone:
            self.wavlm.freeze_base_model()

    def forward(self, x, classes):

        x = self.wavlm(x)

        results = {
            "logits": x.logits,                 # (B, C)
            "classes": classes                  # (B,)
        }

        return results
