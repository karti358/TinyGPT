"""Autoregressive text sampling."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from flash_gpt import runtime as rt
from flash_gpt.model.gpt import TinyGPT


@nnx.jit
def sample_token(logits):
    logits = logits / (rt.config.sampling.temperature + 1e-8)
    keys = jax.random.split(jax.random.PRNGKey(42), logits.shape[0])
    logits, indices = jax.lax.top_k(logits, k=rt.config.sampling.top_k)
    logits = nnx.softmax(logits)
    tokens = [
        jax.random.choice(keys[i], indices[i, 0], p=logits[i, 0]) for i in range(logits.shape[0])
    ]
    return jnp.array(tokens)[:, None]


@nnx.jit
def prepare(model, x):
    return model.prepare(x)


@nnx.jit
def sample_step(model, x):
    return model.inference(x)


@nnx.jit(static_argnums=(1,))
def prepare_initial(batch, length=20):
    return jax.lax.dynamic_slice_in_dim(batch[0], 0, length, axis=1)


def prompt_to_batch(token_ids: list[int], batch_size: int | None = None):
    """Build a (inputs, targets) batch from token ids (replicated across batch dim)."""
    if batch_size is None:
        batch_size = rt.config.sampling.batch_size

    max_len = rt.config.model.max_len
    encoding = list(token_ids)[:max_len]
    encoding = encoding + [0] * (max_len - len(encoding))

    inputs = jnp.array(encoding[:-1], dtype=jnp.int32)
    targets = jnp.array(encoding[1:], dtype=jnp.int32)
    inputs = jnp.stack([inputs] * batch_size)
    targets = jnp.stack([targets] * batch_size)
    return inputs, targets


def sample(
    model: TinyGPT,
    batch,
    seed: int = 42,
    *,
    prefix_len: int | None = None,
    max_new_tokens: int | None = None,
):
    model.eval()
    rng = jax.random.PRNGKey(seed)

    if prefix_len is None:
        prefix_len = 20
    prefix_len = min(prefix_len, rt.config.model.max_len - 1)

    length = prefix_len
    t_max = rt.config.model.max_len

    x = prepare_initial(batch, length=length)
    logits = prepare(model, x)
    last_logit = jax.lax.dynamic_index_in_dim(logits, logits.shape[1] - 1, axis=1, keepdims=True)

    rng, subrng = jax.random.split(rng)
    first_gen_token = sample_token(last_logit)
    length += 1

    n_generate = t_max - length
    if max_new_tokens is not None:
        n_generate = min(n_generate, max_new_tokens)

    rng, loop_rng = jax.random.split(rng)
    scan_rngs = jax.random.split(loop_rng, n_generate)

    generated = []
    current_token = first_gen_token
    for lrngs in scan_rngs:
        logit = sample_step(model, current_token)
        current_token = sample_token(logit)
        generated.append(current_token)
        length += 1

    model.train()
    return jnp.concatenate([x, first_gen_token] + generated, axis=1)


def sample_from_prompt(
    model: TinyGPT,
    prompt: str,
    seed: int = 42,
    *,
    prefix_len: int | None = None,
    max_new_tokens: int | None = None,
) -> jnp.ndarray:
    """Tokenize prompt, run generation, return token ids for the first batch row."""
    if rt.tokenizer is None:
        raise RuntimeError("tokenizer not set; call init_from_config(config, tokenizer=...)")

    token_ids = rt.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    if prefix_len is None:
        prefix_len = min(len(token_ids), rt.config.model.max_len - 1)

    batch = prompt_to_batch(token_ids)
    out = sample(
        model,
        batch,
        seed=seed,
        prefix_len=prefix_len,
        max_new_tokens=max_new_tokens,
    )
    return out[0]


def decode_tokens(token_ids) -> str:
    if rt.tokenizer is None:
        raise RuntimeError("tokenizer not set; call init_from_config(config, tokenizer=...)")
    return rt.tokenizer.decode(list(token_ids))
