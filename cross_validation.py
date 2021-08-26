import itertools
import pickle
from datetime import date
import torch.optim as optim
import torch
import torchtext
import torch.nn as nn
import project.model as model
import project.self_attention_model as attn_model
from project.HW3_additions.training import LSTMTrainer, AttentionTrainer
import warnings

warnings.filterwarnings('ignore')


def save_to_file(perf, num_epochs, do_model, seed):
    with open(f'{(do_model + "_") if do_model else ""}seed{seed}_ne{num_epochs}_output_{date.today()}.tmp', 'wb') as output_file:
        pickle.dump(perf, output_file)


def hp_fitting(num_epochs=20, do_model=None, seed=679):
    early_stopping = 999
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(f'Running on a {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}')
    ds_train, ds_valid, ds_test, embedding_tensor = model.load_data()
    embedding = nn.Embedding.from_pretrained(embedding_tensor)
    batch_sizes = [64, 32]

    loss_fn = nn.NLLLoss()

    perf = {}

    for batch_size in batch_sizes:
        # BucketIterator creates batches with samples of similar length
        # to minimize the number of <pad> tokens in the batch.
        torch.manual_seed(seed)
        dl_train, dl_valid, dl_test = torchtext.data.BucketIterator.splits(
            (ds_train, ds_valid, ds_test), batch_size=batch_size,
            shuffle=True, device=device)

        hp = [
            [0.1, 0.01, 0.007, 0.005, 0.001],  # lr
            [0.9, 0.7, 0.5, 0.3],  # dropout
            [3, 2, 4],  # layer count
            [0.1, 0.5, 0.7, 0.9]  # pe dropout
        ]

        hp_combinations = list(itertools.product(*hp))

        if do_model is None or do_model == 'attention':
            for comb in hp_combinations:
                torch.manual_seed(seed)
                lr, dropout, layer_count, pe_dropout = comb
                print(f'Running Attention with batch_size={batch_size} lr={lr} dropout={dropout} layer_count={layer_count}')

                learning_model = attn_model.AttentionModel(embedding=embedding, embedding_dim=embedding.embedding_dim, dropout=dropout, attention_layer_count=layer_count,
                                                           pe_dropout=pe_dropout)
                learning_model.to(device)
                optimizer = optim.Adam(learning_model.parameters(), lr=lr)

                trainer = AttentionTrainer(learning_model, loss_fn, optimizer, device)
                results = trainer.fit(dl_train, dl_test, num_epochs, early_stopping=early_stopping)

                perf[(f'Attention_bs_{batch_size}', *comb)] = results

                save_to_file(perf, num_epochs, do_model, seed)

        if do_model is None or do_model == 'multihead':
            # hp.append([])

            for comb in hp_combinations:
                torch.manual_seed(seed)
                lr, dropout, layer_count, pe_dropout = comb
                print(f'Running Attention with batch_size={batch_size} lr={lr} dropout={dropout} layer_count={layer_count}')

                learning_model = attn_model.MultiheadAttentionModel(embedding=embedding, embedding_dim=embedding.embedding_dim, dropout=dropout, attention_layer_count=layer_count,
                                                                    pe_dropout=pe_dropout, with_norm=True, num_heads=2)
                learning_model.to(device)
                optimizer = optim.Adam(learning_model.parameters(), lr=lr)

                trainer = AttentionTrainer(learning_model, loss_fn, optimizer, device)
                results = trainer.fit(dl_train, dl_test, num_epochs, early_stopping=early_stopping)

                perf[(f'Attention_bs_{batch_size}', *comb)] = results

                save_to_file(perf, num_epochs, do_model, seed)

        if do_model is None or do_model == 'lstm':
            hp = hp[:-1]
            hp.append([150, 200, 250])  # hidden dim
            hp_combinations = list(itertools.product(*hp))
            print(f'Num of combinations: {len(hp_combinations)}')
            for i, comb in enumerate(hp_combinations):
                print(f'Running combinations {i}/{len(hp_combinations)}')
                torch.manual_seed(seed)
                lr, dropout, layer_count, hidden_dim = comb
                print(f'Running LSTM with batch_size={batch_size} lr={lr} dropout={dropout} layer_count={layer_count}')

                learning_model = model.LSTMModel(embedding=embedding, embedding_dim=embedding.embedding_dim, dropout=dropout, num_layers=layer_count, hidden_dim=hidden_dim,
                                                 device=device)
                print(f'LSTM model: {learning_model}')
                learning_model.train()
                learning_model.to(device=device)
                print(f'data device {dl_train.device}')
                print(f'model device {learning_model.device}')
                optimizer = optim.Adam(learning_model.parameters(), lr=lr)

                trainer = LSTMTrainer(learning_model, loss_fn, optimizer, device=device)
                results = trainer.fit(dl_train, dl_test, num_epochs, early_stopping=early_stopping)

                perf[(f'LSTM_bs_{batch_size}', *comb)] = results

                save_to_file(perf, num_epochs, do_model, seed)

    return perf


def main():
    perf = hp_fitting(do_model='lstm')

    today = date.today()
    with open(f'output_{today}.final', 'wb') as output_file:
        pickle.dump(perf, output_file)


if __name__ == '__main__':
    main()
