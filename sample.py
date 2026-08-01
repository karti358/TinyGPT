#!/usr/bin/env python3
"""Sample text from a trained TinyGPT checkpoint.

Usage:
    python sample.py "There was a small home..."
    python sample.py "Once upon a time" --checkpoint checkpoints/1234567890_state
"""

from __future__ import annotations

import argparse
import os
import sys

# Default for local CPU dev (match training notebook).
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

# Allow running before `pip install -e .`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import jax
import tiktoken
from flax import nnx

from flash_gpt import runtime as rt
from flash_gpt.checkpoints import find_latest_checkpoint, load_checkpoint
from flash_gpt.config import get_config
from flash_gpt.model.gpt import TinyGPT
from flash_gpt.runtime import init_from_config
from flash_gpt.sample import decode_tokens, sample_from_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a TinyGPT checkpoint")
    parser.add_argument("prompt", help='Text prompt, e.g. "There was a small home..."')
    parser.add_argument(
        "--checkpoint",
        "-c",
        default=os.environ.get("FLASH_GPT_CHECKPOINT"),
        help="Orbax checkpoint directory (default: latest under ./checkpoints or FLASH_GPT_CHECKPOINT)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=None, help="Override config sampling.temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Override config sampling.top_k")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Cap number of tokens to generate (default: fill to max_len)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Sampling batch size / KV-cache batch dim (default: 16, match training)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        try:
            checkpoint = str(find_latest_checkpoint("checkpoints"))
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            print("Pass --checkpoint PATH or set FLASH_GPT_CHECKPOINT.", file=sys.stderr)
            sys.exit(1)

    tok = tiktoken.get_encoding("gpt2")
    cfg = get_config(vocab_size=tok.n_vocab)
    if args.batch_size is not None:
        cfg.sampling.batch_size = args.batch_size
    if args.temperature is not None:
        cfg.sampling.temperature = args.temperature
    if args.top_k is not None:
        cfg.sampling.top_k = args.top_k

    init_from_config(cfg, tok)

    model = TinyGPT(rngs=nnx.Rngs(cfg.model.seed))
    model = load_checkpoint(model, checkpoint, mesh=rt.MESH)

    token_ids = jax.device_get(
        sample_from_prompt(
            model,
            args.prompt,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )
    )
    print(decode_tokens(token_ids))


if __name__ == "__main__":
    main()
