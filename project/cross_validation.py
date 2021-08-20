import torch.optim as optim
from ray import tune
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test
from project.HW3_additions import training
from project.HW3_additions.training import LSTMTrainer
import torch.optim as optim
import torch
import torchtext
import torch.nn as nn
import project.model as model
import project.glove_parser as glove
import numpy as np
from project.config import lstm_hyper_params



def get_trainer_mnist(model_trainer, ds_train, ds_valid, ds_test, embedding_tensor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(config):
        config.update(lstm_hyper_params)
        print(f'Config is: {config}')
        embedding = nn.Embedding.from_pretrained(embedding_tensor)
        learning_model = model.SimplePredictionModel(embedding_dim=embedding.embedding_dim,
                                                     hidden_dim=config['hidden_dim'],
                                                     num_layers=config['num_layers'],
                                                     embedding=embedding,
                                                     device=device)

        loss = nn.NLLLoss()

        dl_train, dl_valid, dl_test = torchtext.data.BucketIterator.splits((ds_train, ds_valid, ds_test),
                                                                           batch_size=config['batch_size'],
                                                                           shuffle=True,
                                                                           device='cpu')

        optimizer = optim.Adam(learning_model.parameters(), lr=config['lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=config['factor'], patience=config['patience'], verbose=True
        )

        trainer = model(learning_model, loss, optimizer, device)
        for epoch in range(20):
            epoch_result = trainer.train_epoch(dl_train, verbose=True)

            # Every X epochs, we'll generate a sequence starting from the first char in the first sequence
            # to visualize how/if/what the model is learning.
            if epoch == 0 or model_trainer(epoch + 1) % 25 == 0:
                avg_loss = np.mean(epoch_result.losses)
                accuracy = np.mean(epoch_result.accuracy)
                print(f'\nEpoch #{epoch + 1}: Avg. loss = {avg_loss:.3f}, Accuracy = {accuracy:.2f}%')

    return train


def cross_validate():
    ds_train, ds_valid, ds_test, embedding_tensor = model.load_data()

    analysis = tune.run(get_trainer_mnist(LSTMTrainer, ds_train, ds_valid, ds_test, embedding_tensor),
                        config={'lr': tune.grid_search([0.001, 0.01, 0.1])})

    print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))

    # Get a dataframe for analyzing trial results.
    return analysis.dataframe()
