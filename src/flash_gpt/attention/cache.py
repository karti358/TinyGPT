"""KV cache for autoregressive inference."""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P

from flash_gpt import runtime as rt


class KVCache(nnx.Module):
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        d: int,
    ):
        self.k = nnx.Variable(
            jnp.zeros((batch_size * num_heads, max_seq_len, d), dtype=jnp.float32),
            sharding_names=(
                "x" if (batch_size * num_heads) % rt.devices.shape[0] == 0 else None,
                None,
                None,
            ),
            mesh=rt.VMAP_MESH,
        )
        self.v = nnx.Variable(
            jnp.zeros((batch_size * num_heads, max_seq_len, d), dtype=jnp.float32),
            sharding_names=(
                "x" if batch_size % rt.devices.shape[0] == 0 else None,
                None,
                None,
            ),
            mesh=rt.VMAP_MESH,
        )
        self.pos = nnx.Variable(jnp.array(0, dtype=jnp.int32))

    def reset(self):
        self.k[...] = jax.device_put(
            jnp.zeros_like(self.k[...], dtype=jnp.float32),
            jax.sharding.NamedSharding(
                rt.MESH,
                P(
                    "x" if self.k[...].shape[0] % rt.devices.shape[0] == 0 else None,
                    None,
                    None,
                ),
            ),
        )
        self.v[...] = jax.device_put(
            jnp.zeros_like(self.v[...], dtype=jnp.float32),
            jax.sharding.NamedSharding(
                rt.MESH,
                P(
                    "x" if self.v[...].shape[0] % rt.devices.shape[0] == 0 else None,
                    None,
                    None,
                ),
            ),
        )
        self.pos[...] = jnp.array(0, dtype=jnp.int32)
