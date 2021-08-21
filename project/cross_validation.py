import itertools

import torch.optim as optim
from sklearn.model_selection import KFold
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test
import torch.optim as optim
import torch
import torchtext
import torch.nn as nn
import project.model as model
import project.self_attention_model as attn_model
import numpy as np
from project.config import lstm_hyper_params
from project.HW3_additions.training import LSTMTrainer, AttentionTrainer
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, SubsetRandomSampler, ConcatDataset


def hp_fitting():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on a {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}')
    ds_train, ds_valid, ds_test, embedding_tensor = model.load_data()
    embedding = nn.Embedding.from_pretrained(embedding_tensor)
    batch_size = 32

    # BucketIterator creates batches with samples of similar length
    # to minimize the number of <pad> tokens in the batch.
    dl_train, dl_valid, dl_test = torchtext.data.BucketIterator.splits(
        (ds_train, ds_valid, ds_test), batch_size=batch_size,
        shuffle=True, device=device)

    loss_fn = nn.NLLLoss()

    perf = {}
    hp = [
        [0.001, 0.005, 0.01, 0.05, 0.1],  # lr
        [0],  # dropout
        [50, 100, 150, 200, 250],  # hidden dim
    ]

    hp_combinations = list(itertools.product(*hp))

    for comb in hp_combinations:
        lr, dropout, hidden_dim = comb

        learning_model = attn_model.AttentionModel(embedding=embedding, embedding_dim=embedding.embedding_dim)
        learning_model.to(device)
        optimizer = optim.Adam(learning_model.parameters(), lr=lr)

        trainer = AttentionTrainer(learning_model, loss_fn, optimizer, device)
        results = trainer.fit(dl_train, dl_test, 10)

        perf[comb] = results


print('Starting')
hp_fitting()
