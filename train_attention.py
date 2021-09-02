# from cross_validation import main
import project.config as config
from datetime import date
import torch.optim as optim
import torch
import torchtext
import torch.nn as nn
import project.data_loader as data_loader
import project.self_attention_model as attn_model
from project.HW3_additions.training import AttentionTrainer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_model():
    ds_train, ds_valid, ds_test, embedding_tensor, _,_ ,_ = data_loader.load_data()
    embedding = nn.Embedding.from_pretrained(embedding_tensor)
    loss_fn = nn.NLLLoss()
    batch_size = config.self_attention_params["batch_size"]
    dl_train, _, dl_test = torchtext.data.BucketIterator.splits(
    (ds_train, ds_valid, ds_test), batch_size=batch_size,
    shuffle=True, device=device)
    learning_model = attn_model.AttentionModel(embedding=embedding,
                                               embedding_dim=embedding.embedding_dim,
                                               dropout=config.self_attention_params["dropout"],
                                               attention_layer_count=config.self_attention_params["num_layers"],
                                                           pe_dropout=config.self_attention_params["pe_dropout"])
    print(f'Attention model: {learning_model}')

    learning_model.to(device)
    optimizer = optim.Adam(learning_model.parameters(), lr=config.self_attention_params["lr"])

    trainer = AttentionTrainer(learning_model, loss_fn, optimizer, device)
    trainer.fit(dl_train, dl_test, config.self_attention_params["num_epochs"])

if __name__=="__main__":
    train_model()