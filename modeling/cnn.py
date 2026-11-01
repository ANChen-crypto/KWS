
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.init as init
import torch.nn.functional as F

import math


class Conv1dAttention(nn.Module):
    def __init__(self, feat=64, in_dim=40) -> None:
        super(Conv1dAttention, self).__init__()
        self.q_conv = nn.Conv1d(feat, feat, 1, stride=1)
        self.k_conv = nn.Conv1d(feat, feat, 1, stride=1)
        self.v_conv = nn.Conv1d(feat, feat, 1, stride=1)
        self.out_proj = nn.Linear(in_dim, in_dim)
        self.sqrt_dim = math.sqrt(in_dim)

    def forward(self, x):
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        content_score = torch.matmul(q, k.transpose(1, 2)) / self.sqrt_dim
        content_score = F.softmax(content_score, dim=-1)
        context = torch.matmul(content_score, v)
        return self.out_proj(context)

class Conv1dSubampling(nn.Module):
    def __init__(self, input_dim=35, feat=64, dropout=0.2) -> None:
        super(Conv1dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(input_dim, feat, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(feat),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            )
      
    def forward(self, inputs: Tensor):
        
        outputs = self.sequential(inputs)

        outputs = outputs[:, :, 2:-2]  # b,64,21

        return outputs
    
class CNN(nn.Module):
    def __init__(
            self,
            input_dim=35,
            num_layers=10,
            feat=64,
            out_class=37,
            kernel_size=3,
            dropout=0.2
        ):
        super(CNN, self).__init__()

        self.num_layers = num_layers

        self.subsampling = Conv1dSubampling(input_dim, feat)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feat, feat*2, kernel_size, stride=1, padding=(kernel_size-1) // 2),
                nn.BatchNorm1d(feat*2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(feat*2, feat, kernel_size, stride=1, padding=(kernel_size-1) // 2),
                nn.BatchNorm1d(feat),
                nn.ReLU(),
                nn.Dropout(dropout),
                # Conv1dAttention(),
            ) for _ in range(num_layers)
        ])
        self.out_layer = nn.Sequential(
            nn.Conv1d(feat, out_class, 1, stride=1, padding=0),
        )
        self.out_layer_binary = nn.Sequential(
            nn.Conv1d(feat, 1, 1, stride=1, padding=0),
            nn.Linear(21, 2)
        )

    def forward(self, x, **kwargs):
        x = self.subsampling(x)
        for i, layer in enumerate(self.layers):
            x = x + layer(x)
        out = self.out_layer(x)     # b, c, t
        out = F.log_softmax(out, dim=1)
        out_binary = self.out_layer_binary(x)
        result = {
            "logits": out.permute(2,0,1),   # t, b, c
            "binary_logits": out_binary.squeeze(1)
        }
        return result

class SSLCNN(nn.Module):
    def __init__(
            self,
            input_dim=35,
            num_layers=10,
            feat=64,
            kernel_size=3,
            dropout=0.3
        ):
        super(SSLCNN, self).__init__()

        self.num_layers = num_layers

        self.subsampling = Conv1dSubampling(input_dim, feat)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feat, feat*2, kernel_size, stride=1, padding=(kernel_size-1) // 2),
                nn.BatchNorm1d(feat*2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(feat*2, feat, kernel_size, stride=1, padding=(kernel_size-1) // 2),
                nn.BatchNorm1d(feat),
                nn.ReLU(),
                nn.Dropout(dropout),
                # Conv1dAttention(),
            ) for _ in range(num_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Conv1d(feat, input_dim, 1, stride=1, padding=0),
            nn.Linear(21, 197)
        )

    def forward(self, x, **kwargs):
        x = self.subsampling(x)
        for i, layer in enumerate(self.layers):
            x = x + layer(x)
        out = self.decoder(x)
        result = {
            "out": out,
        }
        return result
    
class CNNOnnx(nn.Module):
    def __init__(
            self,
            input_dim=35,
            num_layers=10,
            feat=64,
            out_class=37,
            kernel_size=3,
            dropout=0.2
        ):
        super(CNNOnnx, self).__init__()

        self.num_layers = num_layers
        
        self.subsampling = Conv1dSubampling(input_dim, feat)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feat, feat*2, kernel_size, stride=1, padding=(kernel_size-1) // 2),
                nn.BatchNorm1d(feat*2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(feat*2, feat, kernel_size, stride=1, padding=(kernel_size-1) // 2),
                nn.BatchNorm1d(feat),
                nn.ReLU(),
                nn.Dropout(dropout),
                # Conv1dAttention(),
            ) for _ in range(num_layers)
        ])
        self.out_layer = nn.Sequential(
            nn.Conv1d(feat, out_class, 1, stride=1, padding=0),
        ) 

    def forward(self, inputs, **kwargs):
        inputs = self.subsampling(inputs)
        for i, layer in enumerate(self.layers):
            inputs = inputs + layer(inputs)
        output = self.out_layer(inputs)     # b, c, t
        output = F.log_softmax(output, dim=1)
        output = output.permute(0,2,1)        # b, c, t
        # output = output.permute(2,0,1)      # t, b, c
        return output
    
if __name__ == "__main__":
    x = torch.randn(1, 35, 197)
    model = CNNOnnx()
    print(model(x))

    num_params = sum(p.numel() for p in model.parameters())
    print(num_params)