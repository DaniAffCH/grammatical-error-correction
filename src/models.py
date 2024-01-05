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


class CopyAttention(nn.Module):
    def __init__(self, embedding_size) -> None:
        super().__init__()
        self.copyAtt = nn.MultiheadAttention(embedding_size, 4, 0.1)
        self.copyAlphaLinear = nn.Linear(embedding_size, 1)

    def forward(self, encoderHidden: torch.Tensor, src_tokens: torch.Tensor, encoderPaddingMask: torch.Tensor, decoderHidden: torch.Tensor, outDistribution: torch.Tensor):
        # src, src_tokens, src_pad_mask, out, outDistribution
        src_tokens = src_tokens.permute(1, 0)

        x_copy, copy_attn = self.copyAtt(
            query=decoderHidden,
            key=encoderHidden,
            value=encoderHidden,
            key_padding_mask=encoderPaddingMask,
            need_weights=True,
        )

        x_copy = x_copy.transpose(0, 1)

        copy_alpha = torch.sigmoid(self.copyAlphaLinear(x_copy))

        compositeDistribution = copy_alpha * outDistribution
        copyDistribution = (1-copy_alpha) * copy_attn

        extendedOutDistribution = torch.zeros(outDistribution.size(
            0), outDistribution.size(1), src_tokens.size(1)).float()

        if src_tokens.device.type == 'cuda':
            extendedOutDistribution = extendedOutDistribution.cuda()

        outDistribution = torch.cat(
            [outDistribution, extendedOutDistribution], dim=-1)

        src_tokens = src_tokens.unsqueeze(
            1).repeat(1, outDistribution.size(1), 1)

        compositeDistribution.scatter_add_(-1, src_tokens, copyDistribution)

        return compositeDistribution


class S2S(pl.LightningModule):
    def __init__(
        self,
        tok,
        out_vocab_size,
        pad_idx,
        embedding_size=256,
        dropout=0.1,
        lr=1e-4,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.pad_idx = pad_idx
        self.dropout = dropout
        self.out_vocab_size = out_vocab_size

        self.embeddings = TokenEmbedding(
            vocab_size=self.out_vocab_size, emb_size=embedding_size
        )

        self.copying = CopyAttention(embedding_size)

        self.pos_encoder = PositionalEncoding(
            d_model=embedding_size, dropout=dropout)

        self.encodeEmbed = nn.Sequential(
            self.embeddings,
            self.pos_encoder
        )

        self.transformer = torch.nn.Transformer(
            d_model=embedding_size,
            nhead=4,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=dropout,
        )

        self.linear = Linear(embedding_size, out_vocab_size)

        self.do = nn.Dropout(p=self.dropout)
        self.tok = tok

    def init_weights(self) -> None:
        self.embeddings.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def preprocessInput(self, input):
        return input["input_ids"].permute(1, 0), ~input["attention_mask"].bool()

    def generateTargetMask(self, length, device):
        return torch.triu(
            torch.ones(length, length, device=device) * float("-inf"), diagonal=1
        )

    def maskedAccuracy(self, y, y_hat, pad_idx):
        mask = y != pad_idx

        return (torch.masked_select(y, mask) ==
                torch.masked_select(y_hat, mask)).double().mean()

    def forward(self, src, trg):

        src_tokens, src_pad_mask = self.preprocessInput(src)
        trg, trg_pad_mask = self.preprocessInput(trg)

        # encoder
        src = self.encodeEmbed(src_tokens)
        src = self.transformer.encoder(src, src_key_padding_mask=src_pad_mask)

        src = self.pos_encoder(src)

        # decoder
        trg = self.encodeEmbed(trg)
        trg_mask = self.generateTargetMask(trg.size(0), trg.device)

        out = self.transformer.decoder(
            tgt=trg, memory=src, tgt_mask=trg_mask, tgt_key_padding_mask=trg_pad_mask
        )

        # head
        outDistribution = out.permute(1, 0, 2)
        outDistribution = self.linear(outDistribution)

        # copy
        finalDistribution = self.copying(src, src_tokens,
                                         src_pad_mask, out, outDistribution)

        return finalDistribution

    def training_step(self, batch, batch_idx):
        y_hat, trg_out = self.feedforward(batch, batch_idx)

        y_hat = y_hat.view(-1, y_hat.size(2))
        y = trg_out.contiguous().view(-1)

        loss = F.cross_entropy(y_hat, y, ignore_index=self.pad_idx)

        self.log("training_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        y_hat, trg_out = self.feedforward(batch, batch_idx)

        y_hat = y_hat.view(-1, y_hat.size(2))
        y = trg_out.contiguous().view(-1)

        loss = F.cross_entropy(y_hat, y, ignore_index=self.pad_idx)

        _, predicted = torch.max(y_hat, 1)
        acc = self.maskedAccuracy(y, predicted, pad_idx=self.pad_idx)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)

        return loss

    def feedforward(self, batch, batch_idx):
        src, trg = batch

        trg_in = {
            "input_ids": trg["input_ids"][:, :-1],
            "attention_mask": trg["attention_mask"][:, :-1]
        }

        trg_out = trg["input_ids"][:, 1:]

        y_hat = self(src, trg_in)

        return y_hat, trg_out

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, 1000)
        return {"optimizer": opt, "lr_scheduler": sched}
