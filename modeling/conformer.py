
import torch
import torch.nn as nn
import math
import numpy as np
from torch import Tensor
from typing import Tuple
from typing import Optional
import torch.nn.init as init
import torch.nn.functional as F
from modeling.transformer_decoder import TransformerDecoder

    
class DepthwiseConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode='zeros',
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)
        
class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class PointwiseConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length) -> Tensor:
        return self.pe[:, :length]

class RelativeMultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        #self.query_proj = weight_sharing(d_model)
        #self.key_proj = weight_sharing(d_model)
        #self.value_proj = weight_sharing(d_model)
        
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        # self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        # self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        # torch.nn.init.xavier_uniform_(self.u_bias)
        # torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            # mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)
        L = value.size(1)

        query = self.query_proj(query).view(batch_size, L, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, L, self.num_heads, self.d_head).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, L, self.num_heads, self.d_head).transpose(1, 2)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, L, self.num_heads, self.d_head)

        content_score = torch.matmul((query).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query).transpose(1, 2), pos_embedding.transpose(1,2).transpose(2,3))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        # if mask is not None:
        #     mask = mask.unsqueeze(1)
        #     score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score


class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.batch_norm = nn.BatchNorm1d(21)
        # self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor):
        batch_size, seq_length, _ = inputs.size()
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        inputs = self.batch_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding)

        return self.dropout(outputs)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()

class ConformerConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.sequential = nn.Sequential(
            # nn.BatchNorm1d(21),
            Transpose(shape=(1, 2)),
            PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
            # PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
            # GLU(dim=1),
            nn.ReLU(),
            DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            # nn.BatchNorm1d(in_channels),  # not accepted for NPU
            # ManualBatchNorm1d(in_channels),
            # Swish(),
            nn.ReLU(),
            PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs).transpose(1, 2)

class FeedForwardModule(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            # nn.BatchNorm1d(21),
            nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            # Swish(),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)


class ResidualConnectionModule(nn.Module):
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor) -> Tensor:
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)

class ConformerBlock(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ):
        super(ConformerBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            # nn.LayerNorm(encoder_dim),
            # nn.BatchNorm1d(21),
        )
        
        self.sequential[1].module.register_forward_hook(self._hook)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs), self.attn_out
        
    def _hook(self, module, input, output):
        self.attn_out = output


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)
        
class Conv1dSubampling(nn.Module):
    def __init__(self, input_dim) -> None:
        super(Conv1dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),

            nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            
            nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
        )
      
    def forward(self, inputs: Tensor):
        
        outputs = self.sequential(inputs)

        outputs = outputs.transpose(1, 2)
        outputs = outputs[:, 2:-2]

        return outputs

class ConformerEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_layers: int = 17,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ):
        super(ConformerEncoder, self).__init__()
        self.conv_subsample = Conv1dSubampling(input_dim)
        self.input_projection = nn.Sequential(
            Linear(input_dim, encoder_dim),
            nn.Dropout(p=input_dropout_p),
        )
        self.layers = nn.ModuleList([ConformerBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        ) for _ in range(num_layers)])

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel() for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: Tensor):
        
        outputs = self.conv_subsample(inputs)
        
        outputs = self.input_projection(outputs)
        
        KV = []
        for i, layer in enumerate(self.layers):
            outputs, attn_out = layer(outputs)
            KV.append(attn_out)
        KV = torch.stack(KV, dim=1)

        return outputs, KV

class ConformerForKWS(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: tuple = (40, 13),
        encoder_dim: int = 512,
        num_layers: int = 17,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
    ):
        super(ConformerForKWS, self).__init__()

        spec_dim = input_dim

        self.spec_encoder = ConformerEncoder(
            input_dim = spec_dim,
            encoder_dim = encoder_dim,
            num_layers = num_layers,
            num_attention_heads = num_attention_heads,
            feed_forward_expansion_factor = feed_forward_expansion_factor,
            conv_expansion_factor = conv_expansion_factor,
            input_dropout_p = input_dropout_p,
            feed_forward_dropout_p = feed_forward_dropout_p,
            attention_dropout_p = attention_dropout_p,
            conv_dropout_p = conv_dropout_p,
            conv_kernel_size = conv_kernel_size,
            half_step_residual = half_step_residual,
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, num_classes),
        )
        
    def forward(self, x, classes=None, **kwargs):

        x, _ = self.spec_encoder(x)
        x = x.permute(0, 2, 1)

        embedding = self.adaptive_pool(x).squeeze(-1)
        x = self.head(embedding)

        results = {
            "logits": x,
            "classes": classes
        }
        return results
    
class ConformerForCTCKWS(nn.Module):
    def __init__(
        self,
        input_dim: tuple = 40,
        encoder_dim: int = 64,
        num_layers: int = 3,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
    ):
        super(ConformerForCTCKWS, self).__init__()

        self.spec_encoder = ConformerEncoder(
            input_dim = input_dim,
            encoder_dim = encoder_dim,
            num_layers = num_layers,
            num_attention_heads = num_attention_heads,
            feed_forward_expansion_factor = feed_forward_expansion_factor,
            conv_expansion_factor = conv_expansion_factor,
            input_dropout_p = input_dropout_p,
            feed_forward_dropout_p = feed_forward_dropout_p,
            attention_dropout_p = attention_dropout_p,
            conv_dropout_p = conv_dropout_p,
            conv_kernel_size = conv_kernel_size,
            half_step_residual = half_step_residual,
        )

        self.head = nn.Linear(encoder_dim, 37)
        self.binary_head = nn.Sequential(
            nn.Conv1d(in_channels=encoder_dim, out_channels=1, kernel_size=1),
            nn.Linear(21, 2),
        )
        
    def forward(self, x, **kwargs):

        x, kv = self.spec_encoder(x)
        embed = torch.mean(x, dim=1)
        binary = self.binary_head(x.transpose(1, 2)).squeeze(1)
        x = self.head(x)

        p_blank = F.softmax(x, dim=-1)[..., 0]
        x = F.log_softmax(x, dim=-1)

        results = {
            "logits": x.transpose(0, 1),
            "binary_logits": binary,
            "p_blank": p_blank,
            "embed": embed
        }
        return results

class ConformerForCTCKWSPretrain(nn.Module):
    def __init__(
        self,
        input_dim: tuple = 40,
        encoder_dim: int = 64,
        num_layers: int = 3,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
    ):
        super(ConformerForCTCKWSPretrain, self).__init__()

        self.spec_encoder = ConformerEncoder(
            input_dim = input_dim,
            encoder_dim = encoder_dim,
            num_layers = num_layers,
            num_attention_heads = num_attention_heads,
            feed_forward_expansion_factor = feed_forward_expansion_factor,
            conv_expansion_factor = conv_expansion_factor,
            input_dropout_p = input_dropout_p,
            feed_forward_dropout_p = feed_forward_dropout_p,
            attention_dropout_p = attention_dropout_p,
            conv_dropout_p = conv_dropout_p,
            conv_kernel_size = conv_kernel_size,
            half_step_residual = half_step_residual,
        )

        self.transformer_decoder = TransformerDecoder(
            vocab_size=39,
            d_model=encoder_dim,
            nhead=num_attention_heads,
            num_layers=num_layers,
            dim_feedforward=feed_forward_expansion_factor * encoder_dim,
            dropout=feed_forward_dropout_p,
            max_len=21,
            pad_id=0,
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=21, out_channels=input_dim, kernel_size=1),
            nn.Linear(encoder_dim, 197),
        )

    def forward(self, x, **kwargs):

        x, kv = self.spec_encoder(x)

        decoder_input = kwargs.get("decoder_input", None)
        token_logits = self.transformer_decoder(tgt=decoder_input, memory=kv)

        mel = self.decoder(x)

        results = {
            "token_logits": token_logits,
            "mel": mel
        }
        return results
