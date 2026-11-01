from dataclasses import dataclass, field

import torch
import torch.nn as nn


@ dataclass
class LSTMModelConfig:
    num_classes: int = field(
        default=5, metadata={"help": "number of classes for kws model"}
    )
    hidden_dim: int = field(
        default=32, metadata={"help": "dimension of model"}
    )
    proj_size: int = field(
        default=32, metadata={"help": "out dimension of model"}
    )
    num_layers: int = field(
        default=2, metadata={"help": "dimension of model"}
    )
    num_classes: int = field(
        default=5, metadata={"help": "number of classes"}
    )
    dropout: int = field(
        default=0.1, metadata={"help": "number of classes"}
    )
    bidirectional: bool = field(
        default=False, metadata={"help": "bi-directional"}
    )

class LSTM(nn.Module):
    def __init__(self, cfg=LSTMModelConfig()):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout
        )

        h = cfg.hidden_dim * 2 if cfg.bidirectional else cfg.hidden_dim
        self.out_linear = nn.Linear(h, cfg.num_classes)
        self.out_binary_linear = nn.Linear(h, 2)

    def forward(self, x, **kwargs):
        
        out, (hn, cn) = self.lstm(x.transpose(1,2))
        out1 = self.out_linear(out)
        out_binary = self.out_binary_linear(out)

        results = {
            "logits": out1[:, -1],                 # (B, C)
            "out_binary": out_binary
        }

        return results

class LSTMQ(nn.Module):
    def __init__(self, cfg=LSTMModelConfig()):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=cfg.hidden_dim,
            # num_layers=cfg.num_layers,
            batch_first=False,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout
        )

        h = cfg.hidden_dim * 2 if cfg.bidirectional else cfg.hidden_dim
        self.out_linear = nn.Linear(h, cfg.num_classes)

    def forward(self, x, **kwargs):
        
        out, _ = self.lstm(x)
        out1 = self.out_linear(out)

        return out1
    
if __name__ == "__main__":
    x = torch.randn((1, 197, 40))
    model = LSTM()
    print(model(x).shape)