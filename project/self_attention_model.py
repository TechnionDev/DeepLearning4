import project.attention as attention
from abc import ABC

import torch
from torch import nn
import torchtext.data
import torchtext.datasets
import math


# noinspection PyAbstractClass
class Sublayer(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.attn = attention.MultiplicativeAttention(dim, dim_out)
        self.channel_mod = nn.Conv1d(in_channels=dim, out_channels=dim_out, kernel_size=1)
        self.layer_norm = nn.LayerNorm(dim_out)

    def forward(self, x):
        x_add = x.transpose(1, 2)
        return self.layer_norm(self.attn(x) + self.channel_mod(x_add).transpose(1, 2))

    @property
    def last_dot_scores(self):
        return self.attn.last_dot_scores


# noinspection PyAbstractClass
class AttentionModel(nn.Module):
    def __init__(self, embedding, embedding_dim, dropout=0.7, attention_layer_count=1, output_dim=5, pe_dropout=0.1):
        super().__init__()
        assert (attention_layer_count >= 1)
        self.attention_layer_count = attention_layer_count
        self.layers = [
            embedding,
            attention.PositionalEncoding(embedding_dim, dropout=pe_dropout),
            Sublayer(embedding_dim, embedding_dim*2),
        ]
        for i in range(attention_layer_count - 1):
            self.layers += [
                nn.Linear(embedding_dim*2, embedding_dim*2, bias=False),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                Sublayer(embedding_dim*2, embedding_dim*2),
            ]
        # self.layers += [nn.Linear(embedding_dim*2, embedding_dim*2, bias=False)]
        self.layers = nn.Sequential(*self.layers)
        self.fc = nn.Linear(embedding_dim*2, output_dim, bias=False)

    def forward(self, x):
        out = self.layers(x)
        out = torch.mean(out, dim=1)
        out = torch.nn.functional.log_softmax(self.fc(out), dim=-1)
        return out

    @property
    def last_dot_scores(self):
        return self.layers[6].last_dot_scores


# noinspection PyAbstractClass
class MultiheadSublayer(nn.Module):
    def __init__(self, dim, num_heads, dropout, with_norm):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        if with_norm:
            self.layer_norm = nn.LayerNorm(dim)
        else:
            self.layer_norm = lambda x: x

    def forward(self, x):
        return self.layer_norm(self.attn(x, x, x)[0] + x)


# noinspection PyAbstractClass
class MultiheadAttentionModel(nn.Module):
    def __init__(self, embedding, embedding_dim, dropout=0.7, attention_layer_count=1, output_dim=5, pe_dropout=0.1, with_norm=True, num_heads=2):
        super().__init__()
        assert (attention_layer_count >= 1)
        self.attention_layer_count = attention_layer_count
        self.layers = [
            embedding,
            attention.PositionalEncoding(embedding_dim, dropout=pe_dropout),
            MultiheadSublayer(dim=embedding_dim, num_heads=num_heads, dropout=dropout, with_norm=with_norm),
        ]
        for i in range(attention_layer_count - 1):
            self.layers += [
                nn.Linear(embedding_dim , embedding_dim * 2, bias=True),
                nn.ReLU(),
                MultiheadSublayer(embedding_dim , num_heads=num_heads, dropout=dropout, with_norm=with_norm),
            ]
        self.layers = nn.Sequential(*self.layers)
        self.fc = nn.Linear(embedding_dim, output_dim, bias=False)

    def forward(self, x):
        out = self.layers(x)
        out = torch.mean(out, dim=1)
        out = torch.nn.functional.log_softmax(self.fc(out), dim=-1)
        return out