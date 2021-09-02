# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch
from torch import nn
import torchtext.data
import torchtext.datasets


# noinspection PyAbstractClass
class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, embedding, bidirectional=False, device='cpu', dropout=0.1):
        super().__init__()
        self.embedding = embedding
        self.embedding.to(device)
        self.num_layers = num_layers

        self.lstm_net = torch.nn.LSTM(batch_first=True, input_size=embedding_dim, hidden_size=hidden_dim,
                                      num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.lstm_net.to(device)
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_layer = nn.Linear(output_dim, 5, bias=False)
        self.output_layer.to(device)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.log_softmax.to(device)
        self.device = device

    def forward(self, x, hidden_state=None):
        #         print(x)
        x_embedded = self.embedding(x).to(device=self.device)

        if hidden_state is None:
            out, hidden_state = self.lstm_net(x_embedded)
        else:
            out, hidden_state = self.lstm_net(x_embedded, hidden_state)
        #         print(f"shape of out is {out.shape}, shape of hidden is {hidden_state[0].shape}")
        out = out[:, -1, :]
        out = out.view(out.shape[0], -1)
        out = self.log_softmax(self.output_layer(out))
        return out, hidden_state
