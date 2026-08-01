"""Flash attention forward/backward vs reference implementation."""

import numpy as np
import jax
import jax.numpy as jnp

from flash_gpt import runtime as rt
from flash_gpt.attention.flash import flash_attention
from flash_gpt.attention.kernel import flash_forward


def naive_causal_attention(q, k, v):
    bh, t, d = q.shape
    scores = (q @ jnp.transpose(k, (0, 2, 1))) / jnp.sqrt(d)
    mask = jnp.tril(jnp.ones((t, t), dtype=jnp.bool_))
    scores = jnp.where(mask[None, ...], scores, rt.NEG_INF)
    attn = jax.nn.softmax(scores, axis=-1)
    return attn @ v


def test_flash_forward_matches_reference(attn_inputs):
    q, k, v = attn_inputs
    flash_out = flash_attention(q, k, v)
    ref_out = naive_causal_attention(q, k, v)
    assert flash_out.shape == ref_out.shape
    np.testing.assert_allclose(flash_out, ref_out, rtol=1e-4, atol=1e-4)


def test_flash_backward_gradcheck(attn_inputs):
    q, k, v = attn_inputs

    def loss_fn(q, k, v):
        return flash_attention(q, k, v).sum()

    jax.check_grads(loss_fn, (q, k, v), order=1, rtol=1e-3, atol=1e-3, eps=1e-3)


def test_flash_forward_returns_aux(attn_inputs):
    q, k, v = attn_inputs
    o, res = flash_forward(q, k, v)
    assert o.shape == q.shape
    assert len(res) == 6
