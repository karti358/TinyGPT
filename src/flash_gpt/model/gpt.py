"""TinyGPT language model."""

from flax import nnx

from flash_gpt import runtime as rt
from flash_gpt.model.blocks import ResidualBlock
from flash_gpt.model.embeddings import Embeddings


class TinyGPT(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.embeddings = Embeddings(rngs=rngs)

        @nnx.split_rngs(splits=rt.config.model.num_resids)
        @nnx.vmap(in_axes=(0,), out_axes=0, transform_metadata={nnx.PARTITION_NAME: None})
        def create_block(rngs):
            return ResidualBlock(rngs)

        self.resblocks = create_block(rngs)

        self.Wout = nnx.Linear(
            in_features=rt.config.model.d,
            out_features=rt.config.data.vocab_size,
            use_bias=False,
            kernel_init=nnx.with_partitioning(
                rt.kernel_init_fn,
                (
                    "x" if rt.config.model.d % rt.devices.shape[0] == 0 else None,
                    "y" if rt.config.data.vocab_size % rt.devices.shape[1] == 0 else None,
                ),
                mesh=rt.MESH,
            ),
            bias_init=nnx.with_partitioning(
                rt.bias_init_fn,
                ("x" if rt.config.data.vocab_size % rt.devices.shape[0] == 0 else None,),
                mesh=rt.MESH,
            ),
            rngs=rngs,
        )

    def __call__(self, x):
        out = self.embeddings(x)

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(x, model):
            return model(x)

        out = forward(out, self.resblocks)
        return self.Wout(out)

    def prepare(self, x):
        out = self.embeddings(x)

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(x, model):
            return model.prepare(x)

        out = forward(out, self.resblocks)
        return self.Wout(out)

    def inference(self, x):
        out = self.embeddings(x, position_offset=self.resblocks.attention_layer.cache.pos[...][0])

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(x, model):
            return model.inference(x)

        out = forward(out, self.resblocks)
        return self.Wout(out)
