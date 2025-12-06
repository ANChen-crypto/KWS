import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from criterion import dict_to_device


class KWSCriterion(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, model, sample, device, reduce=True):
        sample = dict_to_device(sample, device)
        results = model(**sample)

        reduction = "mean" if reduce else "sum"
        loss = F.cross_entropy(results["logits"], results["classes"], reduction=reduction)

        sample_size = results["logits"].size(0)

        results = {
            "loss": loss,
            "sample_size": sample_size,
        }

        return results