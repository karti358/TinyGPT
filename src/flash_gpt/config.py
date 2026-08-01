"""Training and model hyperparameters."""

from ml_collections import ConfigDict


def get_config(vocab_size: int = 50257) -> ConfigDict:
    config = ConfigDict()

    data = ConfigDict()
    data.batch_size = 120
    data.vocab_size = vocab_size

    model = ConfigDict()
    model.seed = 2635276
    model.max_len = 257
    model.d = 256
    model.dropout_rate = 0.1
    model.num_heads = 8
    model.num_resids = 4
    model.head_dim = model.d // model.num_heads
    model.ff_expan = 4
    model.ln_epsilon = 1e-6
    model.q_chunk_size = 64
    model.k_chunk_size = 64

    training = ConfigDict()
    training.lr = 1e-3
    training.epochs = 1
    training.save_and_sample_every = 1000

    sampling = ConfigDict()
    sampling.temperature = 0.95
    sampling.top_k = 10
    sampling.batch_size = 16

    config.data = data
    config.model = model
    config.training = training
    config.sampling = sampling
    return config
