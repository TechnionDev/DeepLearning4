lstm_hyper_params = dict(
    num_layers=2,
    lr = 0.0003,
    num_epochs = 70,
    bidirectional = False,
    dropout = 0.7,
    hidden_dim=100,
    batch_size=64,
)

self_attention_params = dict(
    num_layers = 2,
    lr = 0.0001,
    batch_size = 64,
    num_epochs = 70,
    dropout = 0.7,
    pe_dropout = 0.1
)