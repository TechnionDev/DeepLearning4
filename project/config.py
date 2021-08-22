# from ray import tune


lstm_hyper_params = dict(
    num_layers=4,
    hidden_dim=150,
    batch_size=32,
#     lr=0.001,
    factor=0.1,
    patience=0.1,
    mode='max',
)
