import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    #  https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """

        x = x + self.pe[: x.size(0)]

        return self.dropout(x)


class TokenEmbedding(nn.Module):
    #  https://pytorch.org/tutorials/beginner/translation_transformer.html
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class S2S(pl.LightningModule):
    def __init__(
        self,
        out_vocab_size,
        channels=256,
        dropout=0.1,
    ):
        super().__init__()

        self.dropout = dropout
        self.out_vocab_size = out_vocab_size

        embeddings = TokenEmbedding(
            vocab_size=self.out_vocab_size, emb_size=channels
        )

        pos_encoder = PositionalEncoding(
            d_model=channels, dropout=dropout)

        self.encodeEmbed = nn.Sequential(
            embeddings,
            pos_encoder
        )

        self.transformer = torch.nn.Transformer(
            d_model=channels,
            nhead=4,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=dropout,
        )

        self.linear = Linear(channels, out_vocab_size)

        self.do = nn.Dropout(p=self.dropout)

    def init_weights(self) -> None:
        init_range = 0.1
        self.embeddings.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-init_range, init_range)

    def encode_src(self, src):
        src_pad_mask = ~src["attention_mask"].bool()
        src = src["input_ids"]
        src = src.permute(1, 0)

        src = self.embeddings(src)

        src = self.pos_encoder(src)

        src = self.transformer.encoder(src, src_key_padding_mask=src_pad_mask)

        src = self.pos_encoder(src)

        return src

    def preprocessInput(self, input):
        return input["input_ids"].permute(1, 0), ~input["attention_mask"].bool()

    def targetMask(self, length, device):
        return torch.triu(
            torch.ones(length, length, device=device) * float("-inf"), diagonal=1
        )

    def forward(self, x):
        src, trg = x

        src, src_pad_mask = self.preprocessInput(src)
        trg, trg_pad_mask = self.preprocessInput(trg)
        out_sequence_len = trg.size(0)
        trg_mask = self.targetMask(out_sequence_len, trg.device)

        src = self.encodeEmbed(src)
        trg = self.encodeEmbed(trg)

        src = self.transformer.encoder(src, src_key_padding_mask=src_pad_mask)

        out = self.transformer.decoder(
            tgt=trg, memory=src, tgt_mask=trg_mask, tgt_key_padding_mask=trg_pad_mask
        )

        out = out.permute(1, 0, 2)
        out = self.linear(out)

        return out
