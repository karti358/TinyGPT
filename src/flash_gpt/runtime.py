"""Shared runtime state: mesh, config, initializers."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P

config = None
devices = None
MESH = None
vmap_devices = None
VMAP_MESH = None
NEG_INF = None
tokenizer = None

kernel_init_fn = None
embed_init_fn = None
scale_init_fn = None
bias_init_fn = None


def setup_mesh() -> None:
    global devices, MESH, vmap_devices, VMAP_MESH

    devices = np.array(jax.devices())
    if devices.shape[0] == 2:
        devices = devices.reshape(2, 1)
    elif devices.shape[0] == 8:
        devices = devices.reshape(4, 2)
    else:
        devices = devices[:, np.newaxis]

    MESH = jax.sharding.Mesh(devices=devices, axis_names=("x", "y"))
    vmap_devices = np.array(jax.devices())
    VMAP_MESH = jax.sharding.Mesh(devices=vmap_devices, axis_names=("x",))


def setup_initializers() -> None:
    global kernel_init_fn, embed_init_fn, scale_init_fn, bias_init_fn
    kernel_init_fn = nnx.initializers.xavier_uniform()
    embed_init_fn = nnx.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)
    scale_init_fn = nnx.initializers.ones_init()
    bias_init_fn = nnx.initializers.zeros_init()


def init_from_config(cfg, tok=None) -> None:
    """Initialize global mesh + config used by model and attention modules."""
    global config, NEG_INF, tokenizer
    config = cfg
    tokenizer = tok
    setup_mesh()
    setup_initializers()
    NEG_INF = jnp.finfo(jnp.float32).min


def partition_spec_for_batch(batch_dim: int, seq_dim: int | None = None) -> P:
  """Build a 2D batch/sequence partition spec for the training mesh."""
  x = "x" if batch_dim % devices.shape[0] == 0 else None
  y = "y" if seq_dim is not None and seq_dim % devices.shape[1] == 0 else None
  return P(x, y)
