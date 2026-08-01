"""Token and position embeddings."""

import jax.numpy as jnp
from flax import nnx

from flash_gpt import runtime as rt


class Embeddings(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.emb_inputs = nnx.Embed(
            num_embeddings=rt.config.data.vocab_size,
            features=rt.config.model.d,
            embedding_init=nnx.with_partitioning(
                rt.embed_init_fn,
                ("x" if rt.config.data.vocab_size % rt.devices.shape[0] == 0 else None, None),
                mesh=rt.MESH,
            ),
            rngs=rngs,
        )

        self.emb_pos = nnx.Embed(
            num_embeddings=rt.config.model.max_len,
            features=rt.config.model.d,
            embedding_init=nnx.with_partitioning(
                rt.embed_init_fn,
                ("x" if rt.config.model.max_len % rt.devices.shape[0] == 0 else None, None),
                mesh=rt.MESH,
            ),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray, position_offset=0):
        token_emb = self.emb_inputs(x)
        positions = position_offset + jnp.arange(0, x.shape[1])[None, :]
        pos_emb = self.emb_pos(positions)
        return token_emb + pos_emb
