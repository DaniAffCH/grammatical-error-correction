import torch
from torch import nn, Tensor
import math


class PositionalEncoding(nn.Module):
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class S2S(nn.Module):

    def __init__(self, ntoken: int, d_model: int):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, 0.1)
        self.embedding = TokenEmbedding(ntoken, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=4,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )

        self.head = nn.Linear(d_model, ntoken)
        self.sm = nn.Softmax(dim=0)

    def forward(self, input: Tensor, target: Tensor, input_mask: Tensor, target_mask: Tensor) -> Tensor:
        input_mask = input_mask.bool()
        target_mask = target_mask.bool()

        src = self.embedding(input)
        src = self.pos_encoder(src)

        target = self.embedding(target)
        target = self.pos_encoder(target)

        out_sequence_len = target.size(1)

        trmask = torch.triu(torch.ones(
            out_sequence_len, out_sequence_len, device=self.device) * float("-inf"), diagonal=1)

        encoded = self.transformer.encoder(
            src, src_key_padding_mask=input_mask)

        decoded = self.transformer.decoder(
            target, memory=encoded, tgt_mask=trmask, tgt_key_padding_mask=target_mask)

        out = self.head(decoded)

        return self.sm(out)
