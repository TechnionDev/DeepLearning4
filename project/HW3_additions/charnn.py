import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator
import unittest

def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    y = y / temperature
    softmax = nn.Softmax(dim=dim)
    result = softmax(y)
    # ========================
    return result

class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        num_batches = len(self.dataset) // self.batch_size
        for j in range(num_batches):
            for i in range(self.batch_size):
                yield j + i * num_batches

    def __len__(self):
        return len(self.dataset)

class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model for all layers.
        #  To implement the affine transforms you can use either nn.Linear
        #  modules (recommended) or create W and b tensor pairs directly.
        #  Create these modules or tensors and save them per-layer in
        #  the layer_params list.
        #  Important note: You must register the created parameters so
        #  they are returned from our module's parameters() function.
        #  Usually this happens automatically when we assign a
        #  module/tensor as an attribute in our module, but now we need
        #  to do it manually since we're not assigning attributes. So:
        #    - If you use nn.Linear modules, call self.add_module() on them
        #      to register each of their parameters as part of your model.
        #    - If you use tensors directly, wrap them in nn.Parameter() and
        #      then call self.register_parameter() on them. Also make
        #      sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======
        self.dropout = dropout

        for i in range(n_layers):
            self.layer_params += [{}]
            if i == 0:
                input_dim = in_dim
            else:
                input_dim = h_dim

            self.layer_params[i]['z_in'] = nn.Linear(input_dim, h_dim, bias=False)
            self.layer_params[i]['z_hidden'] = nn.Linear(h_dim, h_dim, bias=True)
            self.layer_params[i]['r_in'] = nn.Linear(input_dim, h_dim, bias=False)
            self.layer_params[i]['r_hidden'] = nn.Linear(h_dim, h_dim, bias=True)
            self.layer_params[i]['g_in'] = nn.Linear(input_dim, h_dim, bias=False)
            self.layer_params[i]['g_hidden'] = nn.Linear(h_dim, h_dim, bias=True)
            self.add_module(f'z_input{i}', self.layer_params[i]['z_in'])
            self.add_module(f'z_hidden{i}', self.layer_params[i]['z_hidden'])
            self.add_module(f'r_input{i}', self.layer_params[i]['r_in'])
            self.add_module(f'r_hidden{i}', self.layer_params[i]['r_hidden'])
            self.add_module(f'g_input{i}', self.layer_params[i]['g_in'])
            self.add_module(f'g_hidden{i}', self.layer_params[i]['g_hidden'])
            in_dim = out_dim
        self.output_layer = nn.Linear(h_dim, out_dim, bias=True)
        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])
        layer_input = input
        layer_output = None

        # TODO:
        #  Implement the model's forward pass.
        #  You'll need to go layer-by-layer from bottom to top (see diagram).
        #  Tip: You can use torch.stack() to combine multiple tensors into a
        #  single tensor in a differentiable manner.
        # ====== YOUR CODE: ======
        layer_output = torch.zeros((input.shape[0], input.shape[1], self.out_dim))

        for j in range(input.shape[1]):
            x = input[:, j]
            for i, layer in enumerate(self.layer_params):
                if i != 0:
                    x = torch.dropout(x, self.dropout, self.training)
                update_gate_out = torch.sigmoid(layer['z_in'](x) + layer['z_hidden'](layer_states[i]))
                reset_gate_out = torch.sigmoid(layer['r_in'](x) + layer['r_hidden'](layer_states[i]))
                h_dot_r = layer_states[i] * reset_gate_out
                candidate_hidden_out = torch.tanh(layer['g_in'](x) + layer['g_hidden'](h_dot_r))
                layer_states[i] = update_gate_out * layer_states[i] + (1 - update_gate_out) * candidate_hidden_out
                x = layer_states[i]
            layer_output[:, j, :] = self.output_layer(x)
        layer_states = torch.cat(layer_states)
        hidden_state = layer_states.reshape((input.shape[0], len(self.layer_params), self.h_dim))
        # ========================

        return layer_output, hidden_state


