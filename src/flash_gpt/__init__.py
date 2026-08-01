"""JAX Flash miniGPT — custom Pallas attention + Flax NNX GPT."""

from flash_gpt.config import get_config
from flash_gpt.runtime import init_from_config

__all__ = ["get_config", "init_from_config"]
