from ray import tune


lstm_hyper_params = dict(
    num_layers=tune.choice([1, 2, 4, 6, 8]),
    hidden_dim=tune.choice([50, 100, 150, 200, 300]),
    batch_size=tune.choice([8, 16, 32, 64, 128]),
    lr=tune.grid_search([0.001, 0.005, 0.01, 0.03, 0.06, 0.1]),
    factor=tune.choice([0.1]),
    patience=tune.choice([0.1]),
    mode=tune.choice(['max']),
)
