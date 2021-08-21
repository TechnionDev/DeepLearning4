import project.attention as attention
from abc import ABC

import torch
from torch import nn
import torchtext.data
import torchtext.datasets
import math


class AttentionModel(nn.Module):
    def __init__(self, embedding, embedding_dim, output_dim=3):
        super(AttentionModel, self).__init__()
        self.embedding = embedding
        self.pos_enc = attention.PositionalEncoding(embedding_dim, dropout=0.4)
        # self.attn1 = nn.MultiheadAttention(embedding_dim, 1)
        # self.attn2 = nn.MultiheadAttention(embedding_dim, 1)
        # self.attn3 = nn.MultiheadAttention(embedding_dim, 1)

        # attn_output, attn_output_weights = multihead_attn(query, key, value)

        self.attn1 = attention.MultiplicativeAttention(embedding_dim, embedding_dim)
        self.bnorm_1 = nn.LayerNorm(embedding_dim)
        self.attn2 = attention.MultiplicativeAttention(embedding_dim, embedding_dim)
        self.bnorm_2 = nn.LayerNorm(embedding_dim)
        self.attn3 = attention.MultiplicativeAttention(embedding_dim, embedding_dim)
        self.bnorm_3 = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim, bias=False)

    def forward(self, x):
        x_embedded = self.embedding(x)
        # print(f"shape of embedded is {x_embedded.shape}")
        encoded_x = self.pos_enc(x_embedded)
        encoded_x = encoded_x + self.attn1(x_embedded)
        encoded_x = self.bnorm_1(encoded_x)
        encoded_x = encoded_x + self.attn2(x_embedded)
        encoded_x = self.bnorm_2(encoded_x)
        encoded_x = encoded_x + self.attn3(x_embedded)
        encoded_x = self.bnorm_3(encoded_x)
        # print(f"shape of x is {encoded_x.shape}")
        encoded_x = torch.mean(encoded_x, dim=1)
        # print(f"shape of x is {encoded_x.shape}")

        out = torch.nn.functional.log_softmax(self.fc(encoded_x), dim=-1)
        return out
