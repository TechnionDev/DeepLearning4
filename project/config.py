lstm_hyper_params = dict(
    num_layers=[1, 2, 4, 6, 8],
    hidden_dim=[50, 100, 150, 200, 300],
    batch_size=[8, 16, 32, 64, 128],
    lr=[0.001, 0.005, 0.01, 0.03, 0.06, 0.1],
    factor=[0.1],
    patience=[0.1],
    mode=['max'],
)
