import project.attention as attention
from abc import ABC

import torch
from torch import nn
import torchtext.data
import torchtext.datasets
import math

class Sublayer(nn.Module):
    def __init__(self, dim,dim_out):
        super().__init__()
        self.attn = attention.MultiplicativeAttention(dim, dim_out)
        self.channel_mod = nn.Conv1d(in_channels=dim,out_channels=dim_out,kernel_size=1)
        self.layer_norm = nn.LayerNorm(dim_out)
    def forward(self,x):
        x_add = x.transpose(1,2)
        return self.layer_norm(self.attn(x)+self.channel_mod(x_add).transpose(1,2))

class AttentionModel(nn.Module):
    def __init__(self, embedding, embedding_dim,dropout=0.7,attention_layer_count=1, output_dim=5):
        super(AttentionModel, self).__init__()
        assert(attention_layer_count>=1)
        self.layers = [
            embedding,
            attention.PositionalEncoding(embedding_dim),
            Sublayer(embedding_dim,embedding_dim*2),
        ]
        for i in range(attention_layer_count-1):
            self.layers+= [
                nn.Linear(embedding_dim*2,embedding_dim*2,bias=True),
                nn.ReLU(),
                nn.Dropout(p=0.7),            
                Sublayer(embedding_dim*2,embedding_dim*2),
        ]
        self.layers = nn.Sequential(*self.layers)
        self.fc = nn.Linear(embedding_dim*2, output_dim, bias=False)

    def forward(self, x):
        out = self.layers(x)
        out = torch.mean(out, dim=1)
        out = torch.nn.functional.log_softmax(self.fc(out), dim=-1)
        return out
