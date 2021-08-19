from abc import ABC

import torch
from torch import nn
import torchtext.data
import torchtext.datasets
import math
import numpy as np


class MultiplicativeAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.wv = torch.nn.Parameter(
            torch.FloatTensor(d_model, d_k).uniform_(-0.1, 0.1))

        self.wk = torch.nn.Parameter(
            torch.FloatTensor(d_model, d_k).uniform_(-0.1, 0.1))
        self.wq = torch.nn.Parameter(
            torch.FloatTensor(d_model, d_k).uniform_(-0.1, 0.1))
        self.d_model = d_model
        self.d_k = d_k
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv
        dot_scores = q @ k.transpose(1, 2) / np.sqrt(self.d_k)
        return self.softmax(dot_scores) @ v


class PositionalEncoding(nn.Module, ABC):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# pos_enc = PositionalEncoding(42, 0.1, 100)
# print(pos_enc)
# vec = torch.rand(20, 32, 42)
# enc = pos_enc(vec).shape
# print(f"enc shape is {enc}")
