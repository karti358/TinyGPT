"""Pytest fixtures: CPU mesh + flash_gpt runtime."""

import os

import pytest

# Must be set before JAX import.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp

from flash_gpt.config import get_config
from flash_gpt.runtime import init_from_config


@pytest.fixture(scope="session")
def flash_runtime():
    cfg = get_config(vocab_size=128)
    cfg.model.max_len = 65
    cfg.model.q_chunk_size = 32
    cfg.model.k_chunk_size = 32
    cfg.sampling.batch_size = 2
    init_from_config(cfg)
    return cfg


@pytest.fixture
def attn_inputs(flash_runtime):
    del flash_runtime
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    bh, t, d = 8, 32, 32
    q = jax.random.normal(k1, (bh, t, d), dtype=jnp.float32)
    k = jax.random.normal(k2, (bh, t, d), dtype=jnp.float32)
    v = jax.random.normal(k3, (bh, t, d), dtype=jnp.float32)
    return q, k, v
