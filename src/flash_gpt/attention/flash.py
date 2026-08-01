"""Custom VJP wrapper around Pallas flash attention."""

import jax

from flash_gpt.attention.kernel import flash_backward, flash_forward


@jax.custom_vjp
def flash_attention(q, k, v):
    o, res = flash_forward(q, k, v)
    return o


flash_attention.defvjp(flash_forward, flash_backward)
