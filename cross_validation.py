import itertools
import pickle
from datetime import date
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
import warnings
warnings.filterwarnings('ignore')

def hp_fitting(num_epochs=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(f'Running on a {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}')
    ds_train, ds_valid, ds_test, embedding_tensor = model.load_data()
    embedding = nn.Embedding.from_pretrained(embedding_tensor)
    batch_sizes = [16, 32, 64]

    loss_fn = nn.NLLLoss()

    perf = {}

    for batch_size in batch_sizes:
        # BucketIterator creates batches with samples of similar length
        # to minimize the number of <pad> tokens in the batch.
        dl_train, dl_valid, dl_test = torchtext.data.BucketIterator.splits(
            (ds_train, ds_valid, ds_test), batch_size=batch_size,
            shuffle=True, device=device)

        hp = [
            [0.001, 0.005, 0.01, 0.05, 0.1],  # lr
            [0, 0.4, 0.6, 0.7, 0.8, 0.9],  # dropout
            [1, 2, 3],  # layer count
        ]

        hp_combinations = list(itertools.product(*hp))

        for comb in hp_combinations:
            lr, dropout, layer_count = comb
            print(f'Running Attention with batch_size={batch_size} lr={lr} dropout={dropout} layer_count={layer_count}')

            learning_model = attn_model.AttentionModel(embedding=embedding, embedding_dim=embedding.embedding_dim, dropout=dropout, attention_layer_count=layer_count)
            learning_model.to(device)
            optimizer = optim.Adam(learning_model.parameters(), lr=lr)

            trainer = AttentionTrainer(learning_model, loss_fn, optimizer, device)
            results = trainer.fit(dl_train, dl_test, num_epochs)

            perf[(f'Attention_bs_{batch_size}', *comb)] = results

            today = date.today()
            with open(f'output_{today}.tmp', 'wb') as output_file:
                pickle.dump(perf, output_file)

        # hp.append([50, 100, 150, 200, 250])  # hidden dim
        # hp_combinations = list(itertools.product(*hp))
        #
        # for comb in hp_combinations:
        #     lr, dropout, layer_count, hidden_dim = comb
        #     print(f'Running LSTM with batch_size={batch_size} lr={lr} dropout={dropout} layer_count={layer_count}')
        # 
        #     learning_model = model.SimplePredictionModel(embedding=embedding, embedding_dim=embedding.embedding_dim, dropout=dropout, num_layers=layer_count, hidden_dim=hidden_dim,
        #                                                  device=device)
        #     learning_model.train()
        #     learning_model.to(device=device)
        #     print(f'data device {dl_train.device}')
        #     print(f'model device {learning_model.device}')
        #     optimizer = optim.Adam(learning_model.parameters(), lr=lr)
        #
        #     trainer = LSTMTrainer(learning_model, loss_fn, optimizer, device=device)
        #     results = trainer.fit(dl_train, dl_test, num_epochs)
        # 
        #     perf[(f'LSTM_bs_{batch_size}', *comb)] = results
        # 
        #     today = date.today()
        #     with open(f'output_{today}.tmp', 'wb') as output_file:
        #         pickle.dump(perf, output_file)

    return perf


if __name__ == '__main__':
    perf = hp_fitting()

    today = date.today()
    with open(f'output_{today}.final', 'wb') as output_file:
        pickle.dump(perf, output_file)
