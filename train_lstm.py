import torch.optim as optim
import torch
import torchtext
import torch.nn as nn
import project.data_loader as data_loader
import project.lstm_model as lstm_model
import project.config as config
from project.HW3_additions.training import LSTMTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_model():

    ds_train, ds_valid, ds_test, embedding_tensor, _,_ ,_ = data_loader.load_data()
    embedding = nn.Embedding.from_pretrained(embedding_tensor)
    loss_fn = nn.NLLLoss()
    batch_size = config.lstm_hyper_params["batch_size"]
    dl_train, _, dl_test = torchtext.data.BucketIterator.splits(
    (ds_train, ds_valid, ds_test), batch_size=batch_size,
    shuffle=True, device=device)
    learning_model = lstm_model.LSTMModel(embedding=embedding,
                                            embedding_dim=embedding.embedding_dim,
                                            dropout=config.lstm_hyper_params["dropout"],
                                            num_layers=config.lstm_hyper_params["num_layers"], 
                                            hidden_dim=config.lstm_hyper_params["hidden_dim"],
                                            device=device)
    print(f'LSTM model: {learning_model}')
    learning_model.train()
    learning_model.to(device=device)
    optimizer = optim.Adam(learning_model.parameters(), lr=config.lstm_hyper_params["lr"])
    trainer = LSTMTrainer(learning_model, loss_fn, optimizer, device=device)
    trainer.fit(dl_train, dl_test, config.lstm_hyper_params["num_epochs"])

if __name__=="__main__":
    train_model()

