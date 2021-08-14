# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch
from torch import nn
import pickle
import glove_parser


class SimplePredictionModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, embedding, bidirectional=True):
        super().__init__()
        self.embedding = embedding
        self.lstm_net = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                                      num_layers=num_layers, bidirectional=bidirectional)
        # self.hidden = None

    def forward(self, x, hidden_state=None):
        x_embedded = self.embedding(x)
        if hidden_state is None:
            out, hidden_state = self.lstm_net(x_embedded)
        else:
            out, hidden_state = self.lstm_net(x_embedded, hidden_state)
        return out, hidden_state


# Press the green button in the gutter to run the script.
# model = SimplePredictionModel(50, 50, 3)
# glove_parser.read_glove_dim(glove_parser.GloveDimSize.FIFTY)
t = torch.load("embedding_parsed/glove.6b.100d.pt")
embedding = nn.Embedding.from_pretrained(t)
# print(embedding.embedding_dim)
model = SimplePredictionModel(embedding_dim=embedding.embedding_dim, hidden_dim=200, num_layers=2, embedding=embedding)
output, hidden_state = model(torch.LongTensor([[1, 6, 7, 8, 4, 2], [1, 6, 7, 8, 10, 2], [1, 6, 7, 8, 5, 2]]))
output2, hidden_state = model(torch.LongTensor([[1, 6, 7, 8, 4, 2], [1, 6, 7, 8, 10, 2], [1, 6, 7, 8, 5, 2]]),
                              hidden_state)
output3, hidden_state = model(torch.LongTensor([[1, 6, 7, 8, 4, 2], [1, 6, 7, 8, 10, 2], [1, 6, 7, 8, 5, 2]]),
                              hidden_state)
output4, hidden_state = model(torch.LongTensor([[1, 6, 7, 8, 4, 2], [1, 6, 7, 8, 10, 2], [1, 6, 7, 8, 5, 2]]),
                              hidden_state)

print(f"output shape is {output4.shape}")
print(f"hn shape is {hidden_state[0].shape}")
print(f"cn shape is {hidden_state[1].shape}")

# input = torch.LongTensor([1, 4, 6])
# print(embedding(input).shape)
# print(t)
# with open("embedding_parsed/glove.6B.50d_word_to_idx.pkl", "rb") as f:
#     print(pickle.load(f))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
