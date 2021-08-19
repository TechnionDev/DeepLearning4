# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch
from torch import nn
import torchtext.data
import torchtext.datasets


class SimplePredictionModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, embedding, bidirectional=True, device='cpu'):
        super().__init__()
        self.embedding = embedding
        self.num_layers = num_layers

        self.lstm_net = torch.nn.LSTM(batch_first=True, input_size=embedding_dim, hidden_size=hidden_dim,
                                      num_layers=num_layers, bidirectional=bidirectional)
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_layer = nn.Linear(output_dim, 3, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.device = device
        # self.hidden = None

    def forward(self, x, hidden_state=None):
        #         print(x)
        x_embedded = self.embedding(x).to(device=self.device)
        #         x_embedded = x_embedded.transpose(0, 1)
        # print(x_embedded.shape)
        # return Falclass SimplePredictionModel(nn.Module):
        #     def __init__(self, embedding_dim, hidden_dim, num_layers, embedding, bidirectional=True,device='cpu'):
        #         super().__init__()
        #         self.embedding = embedding
        #         self.num_layers = num_layers
        #
        #         self.lstm_net = torch.nn.LSTM(batch_first=True,input_size=embedding_dim, hidden_size=hidden_dim,
        #                                       num_layers=num_layers, bidirectional=bidirectional)
        #         output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        #         self.output_layer = nn.Linear(output_dim, 3, bias=False)
        #         self.log_softmax = nn.LogSoftmax(dim=1)
        #         self.device = device
        #         # self.hidden = None
        #
        #     def forward(self, x, hidden_state=None):
        # #         print(x)
        #         x_embedded = self.embedding(x).to(device=self.device)
        # #         x_embedded = x_embedded.transpose(0, 1)
        #         # print(x_embedded.shape)
        #         # return False
        #         # print(x_embedded.shape)
        #         if hidden_state is None:
        #             out, hidden_state = self.lstm_net(x_embedded)
        #         else:
        #             out, hidden_state = self.lstm_net(x_embedded, hidden_state)
        # #         print(f"shape of out is {out.shape}, shape of hidden is {hidden_state[0].shape}")
        #         out = out[:, -1, :]
        #         out = out.view(out.shape[0], -1)
        #         out = self.log_softmax(self.output_layer(out))
        #         return out, hidden_statese
        # print(x_embedded.shape)
        if hidden_state is None:
            out, hidden_state = self.lstm_net(x_embedded)
        else:
            out, hidden_state = self.lstm_net(x_embedded, hidden_state)
        #         print(f"shape of out is {out.shape}, shape of hidden is {hidden_state[0].shape}")
        out = out[:, -1, :]
        out = out.view(out.shape[0], -1)
        out = self.log_softmax(self.output_layer(out))
        return out, hidden_state


def load_data():
    # torchtext Field objects parse text (e.g. a review) and create a tensor representation

    # This Field object will be used for tokenizing the movie reviews text
    # For this application, tokens ~= words
    review_parser = torchtext.data.Field(
        sequential=True, use_vocab=True, lower=True,
        init_token='<sos>', eos_token='<eos>', dtype=torch.long,
        tokenize='spacy', tokenizer_language='en_core_web_sm'
    )

    # This Field object converts the text labels into numeric values (0,1,2)
    label_parser = torchtext.data.Field(
        is_target=True, sequential=False, unk_token=None, use_vocab=True
    )
    # Load SST, tokenize the samples and labels
    # ds_X are Dataset objects which will use the parsers to return tensors
    ds_train, ds_valid, ds_test = torchtext.datasets.SST.splits(
        review_parser, label_parser, root="project/data"
    )
    review_parser.build_vocab(ds_train, vectors="glove.6B.100d")
    label_parser.build_vocab(ds_train)
    # print(f"review parser dict is {review_parser.vocab.vectors}")
    return ds_train, ds_valid, ds_test, review_parser.vocab.vectors
