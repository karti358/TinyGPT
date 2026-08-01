"""KV-cache inference path: mask and position updates."""

import jax
import jax.numpy as jnp
from flax import nnx

from flash_gpt import runtime as rt
from flash_gpt.attention.module import FlashAttention


def test_inference_cache_mask(flash_runtime):
    del flash_runtime
    rngs = nnx.Rngs(0)
    attn = FlashAttention(rngs=rngs)

    b, prefix_len, d_model = rt.config.sampling.batch_size, 5, rt.config.model.d
    x = jax.random.normal(jax.random.key(1), (b, prefix_len, d_model), dtype=jnp.float32)

    attn.prepare(x)
    assert int(attn.cache.pos[...]) == prefix_len

    token = jax.random.normal(jax.random.key(2), (b, 1, d_model), dtype=jnp.float32)
    out1 = attn.inference(token)
    assert out1.shape == (b, 1, d_model)
    assert int(attn.cache.pos[...]) == prefix_len + 1

    out2 = attn.inference(token)
    assert int(attn.cache.pos[...]) == prefix_len + 2
    assert jnp.all(jnp.isfinite(out2))
