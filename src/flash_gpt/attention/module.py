"""Flash attention module with training and inference paths."""

import math

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P

from flash_gpt import runtime as rt
from flash_gpt.attention.cache import KVCache
from flash_gpt.attention.flash import flash_attention


class FlashAttention(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.Whqkv = nnx.Linear(
            in_features=rt.config.model.d,
            out_features=3 * rt.config.model.d,
            use_bias=False,
            kernel_init=nnx.with_partitioning(
                rt.kernel_init_fn,
                (
                    "x" if rt.config.model.d % rt.devices.shape[0] == 0 else None,
                    "y" if (3 * rt.config.model.d) % rt.devices.shape[1] == 0 else None,
                ),
                mesh=rt.MESH,
            ),
            bias_init=None,
            rngs=rngs,
        )

        self.Wp = nnx.Linear(
            in_features=rt.config.model.d,
            out_features=rt.config.model.d,
            use_bias=False,
            kernel_init=nnx.with_partitioning(
                rt.kernel_init_fn,
                (
                    "x" if rt.config.model.d % rt.devices.shape[0] == 0 else None,
                    "y" if rt.config.model.d % rt.devices.shape[1] == 0 else None,
                ),
                mesh=rt.MESH,
            ),
            bias_init=None,
            rngs=rngs,
        )

        self.cache = KVCache(
            batch_size=rt.config.sampling.batch_size,
            num_heads=rt.config.model.num_heads,
            max_seq_len=rt.config.model.max_len,
            d=rt.config.model.head_dim,
        )

    def _project_qkv(self, x: jnp.ndarray):
        B, T = x.shape[:2]
        D = rt.config.model.head_dim
        H = rt.config.model.num_heads

        hqkvd = self.Whqkv(x)
        qkv = jnp.permute_dims(hqkvd.reshape(B, T, 3, H, D), axes=(0, 3, 1, 2, 4))

        q = jax.lax.with_sharding_constraint(
            jax.lax.dynamic_index_in_dim(qkv, 0, axis=3, keepdims=False).reshape(B * H, T, D),
            jax.sharding.NamedSharding(
                rt.VMAP_MESH,
                P("x" if B % rt.devices.shape[0] == 0 else None, None, None),
            ),
        )
        k = jax.lax.with_sharding_constraint(
            jax.lax.dynamic_index_in_dim(qkv, 1, axis=3, keepdims=False).reshape(B * H, T, D),
            jax.sharding.NamedSharding(
                rt.VMAP_MESH,
                P("x" if B % rt.devices.shape[0] == 0 else None, None, None),
            ),
        )
        v = jax.lax.with_sharding_constraint(
            jax.lax.dynamic_index_in_dim(qkv, 2, axis=3, keepdims=False).reshape(B * H, T, D),
            jax.sharding.NamedSharding(
                rt.VMAP_MESH,
                P("x" if B % rt.devices.shape[0] == 0 else None, None, None),
            ),
        )
        return B, T, H, D, q, k, v

    def __call__(self, x: jnp.ndarray):
        B, T, H, D, q, k, v = self._project_qkv(x)
        out = flash_attention(q, k, v).reshape(B, H, T, D)
        out = jnp.permute_dims(out, (0, 2, 1, 3)).reshape(B, T, H * D)
        return self.Wp(out)

    def prepare(self, x: jnp.ndarray):
        B, T, H, D, q, k, v = self._project_qkv(x)
        self.cache.reset()
        self.cache.k[...] = jax.lax.dynamic_update_slice_in_dim(self.cache.k[...], k, 0, axis=1)
        self.cache.v[...] = jax.lax.dynamic_update_slice_in_dim(self.cache.v[...], v, 0, axis=1)

        out = flash_attention(q, k, v).reshape(B, H, T, D)
        out = jnp.permute_dims(out, axes=(0, 2, 1, 3)).reshape(B, T, H * D)
        out = self.Wp(out)
        self.cache.pos[...] = T
        return out

    def inference(self, x: jnp.ndarray):
        B, T = x.shape[:2]
        assert T == 1, "Inference x should have only one timestamp"

        D = rt.config.model.head_dim
        H = rt.config.model.num_heads
        current_pos = self.cache.pos[...]

        hqkvd_new = self.Whqkv(x)
        qkv_new = jnp.permute_dims(hqkvd_new.reshape(B, T, 3, H, D), axes=(0, 3, 1, 2, 4))

        q_new = jax.lax.with_sharding_constraint(
            jax.lax.dynamic_index_in_dim(qkv_new, 0, axis=3, keepdims=False).reshape(B * H, T, D),
            jax.sharding.NamedSharding(
                rt.VMAP_MESH,
                P("x" if B % rt.devices.shape[0] == 0 else None, None, None),
            ),
        )
        k_new = jax.lax.with_sharding_constraint(
            jax.lax.dynamic_index_in_dim(qkv_new, 1, axis=3, keepdims=False).reshape(B * H, T, D),
            jax.sharding.NamedSharding(
                rt.VMAP_MESH,
                P("x" if B % rt.devices.shape[0] == 0 else None, None, None),
            ),
        )
        v_new = jax.lax.with_sharding_constraint(
            jax.lax.dynamic_index_in_dim(qkv_new, 2, axis=3, keepdims=False).reshape(B * H, T, D),
            jax.sharding.NamedSharding(
                rt.VMAP_MESH,
                P("x" if B % rt.devices.shape[0] == 0 else None, None, None),
            ),
        )

        self.cache.k[...] = jax.lax.dynamic_update_slice_in_dim(self.cache.k[...], k_new, current_pos, axis=1)
        self.cache.v[...] = jax.lax.dynamic_update_slice_in_dim(self.cache.v[...], v_new, current_pos, axis=1)
        self.cache.pos[...] = current_pos + 1

        k_full = self.cache.k[...]
        v_full = self.cache.v[...]
        k_T_full = jnp.transpose(k_full, axes=(0, 2, 1))

        scores = (q_new @ k_T_full) / math.sqrt(D)
        mask = jnp.arange(k_full.shape[1]) < (current_pos + 1)
        scores = jnp.where(mask[None, None, :], scores, rt.NEG_INF)

        attn = jax.nn.softmax(scores, axis=-1)
        out = (attn @ v_full).reshape(B, H, T, D)
        out = jnp.permute_dims(out, axes=(0, 2, 1, 3)).reshape(B, T, H * D)
        return self.Wp(out)
