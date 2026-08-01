"""Transformer residual block."""

import jax.numpy as jnp
from flax import nnx

from flash_gpt import runtime as rt
from flash_gpt.attention.module import FlashAttention


class ResidualBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.attention_layer = FlashAttention(rngs=rngs)

        self.dropout_1 = nnx.Dropout(rate=rt.config.model.dropout_rate, rngs=rngs)
        self.dropout_2 = nnx.Dropout(rate=rt.config.model.dropout_rate, rngs=rngs)

        self.dense_1 = nnx.Linear(
            in_features=rt.config.model.d,
            out_features=rt.config.model.d,
            use_bias=True,
            kernel_init=nnx.with_partitioning(
                rt.kernel_init_fn,
                (
                    "x" if rt.config.model.d % rt.devices.shape[0] == 0 else None,
                    "y" if rt.config.model.d % rt.devices.shape[1] == 0 else None,
                ),
                mesh=rt.MESH,
            ),
            bias_init=nnx.with_partitioning(
                rt.bias_init_fn,
                ("x" if rt.config.model.d % rt.devices.shape[0] == 0 else None,),
                mesh=rt.MESH,
            ),
            rngs=rngs,
        )

        self.dense_2 = nnx.Linear(
            in_features=rt.config.model.d,
            out_features=rt.config.model.d,
            use_bias=True,
            kernel_init=nnx.with_partitioning(
                rt.kernel_init_fn,
                (
                    "x" if rt.config.model.d % rt.devices.shape[0] == 0 else None,
                    "y" if rt.config.model.d % rt.devices.shape[1] == 0 else None,
                ),
                mesh=rt.MESH,
            ),
            bias_init=nnx.with_partitioning(
                rt.bias_init_fn,
                ("x" if rt.config.model.d % rt.devices.shape[0] == 0 else None,),
                mesh=rt.MESH,
            ),
            rngs=rngs,
        )

        self.layernorm_1 = nnx.LayerNorm(
            num_features=rt.config.model.d,
            epsilon=rt.config.model.ln_epsilon,
            use_bias=True,
            use_scale=True,
            scale_init=nnx.with_partitioning(
                rt.scale_init_fn,
                ("x" if rt.config.model.d % rt.devices.shape[0] == 0 else None,),
                mesh=rt.MESH,
            ),
            bias_init=nnx.with_partitioning(
                rt.bias_init_fn,
                ("x" if rt.config.model.d % rt.devices.shape[0] == 0 else None,),
                mesh=rt.MESH,
            ),
            rngs=rngs,
        )

        self.layernorm_2 = nnx.LayerNorm(
            num_features=rt.config.model.d,
            epsilon=rt.config.model.ln_epsilon,
            use_bias=True,
            use_scale=True,
            scale_init=nnx.with_partitioning(
                rt.scale_init_fn,
                ("x" if rt.config.model.d % rt.devices.shape[0] == 0 else None,),
                mesh=rt.MESH,
            ),
            bias_init=nnx.with_partitioning(
                rt.bias_init_fn,
                ("x" if rt.config.model.d % rt.devices.shape[0] == 0 else None,),
                mesh=rt.MESH,
            ),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray):
        out = self.attention_layer(x)
        out = self.dropout_1(out)
        out = self.layernorm_1(x + out)

        ffn_out = self.dense_1(out)
        ffn_out = nnx.relu(ffn_out)
        ffn_out = self.dense_2(ffn_out)
        ffn_out = self.dropout_2(ffn_out)
        return self.layernorm_2(out + ffn_out)

    def prepare(self, x: jnp.ndarray):
        out = self.attention_layer.prepare(x)
        out = self.dropout_1(out)
        out = self.layernorm_1(x + out)

        ffn_out = self.dense_1(out)
        ffn_out = nnx.relu(ffn_out)
        ffn_out = self.dense_2(ffn_out)
        ffn_out = self.dropout_2(ffn_out)
        return self.layernorm_2(out + ffn_out)

    def inference(self, x: jnp.ndarray):
        out = self.attention_layer.inference(x)
        out = self.dropout_1(out)
        out = self.layernorm_1(x + out)

        ffn_out = self.dense_1(out)
        ffn_out = nnx.relu(ffn_out)
        ffn_out = self.dense_2(ffn_out)
        ffn_out = self.dropout_2(ffn_out)
        return self.layernorm_2(out + ffn_out)
