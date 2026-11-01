import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size=39,
        d_model=64,
        nhead=8,
        num_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=16,
        pad_id=0,
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_id = pad_id
        self.num_layers = num_layers
        self.pad_id = 0

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id
        )

        self.pos_embedding = nn.Embedding(
            max_len,
            d_model
        )

        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(
        self,
        tgt,
        memory,
    ):
        """
        tgt:
            (B, T)

        memory:
            (B, num_layers from encoder, S, d_model)

        return:
            logits:
            (B, T, vocab_size)
        """

        B, T = tgt.shape

        x = self.token_embedding(tgt)

        pos = torch.arange(
            T,
            device=tgt.device
        ).unsqueeze(0)

        x = x + self.pos_embedding(pos)

        causal_mask = torch.triu(
            torch.ones(
                T,
                T,
                device=tgt.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        tgt_key_padding_mask = (tgt == self.pad_id)

        for i in range(self.num_layers):

            memory_i = memory[:, i, :, :]

            x = self.decoder_layers[i](
                tgt=x,
                memory=memory_i,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

        logits = self.fc(x)

        return logits

if __name__ == "__main__":
    model = TransformerDecoder()
    tgt = torch.randint(0, 39, (2, 10))
    memory = torch.randn(2, 3, 21, 64)
    logits = model(tgt, memory)
    print(logits.shape)