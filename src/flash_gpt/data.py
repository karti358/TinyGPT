"""TinyStories data pipeline."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import grain
from grain import MapDataset
from jax.sharding import PartitionSpec as P

from flash_gpt import runtime as rt


class DataSource(grain.sources.RandomAccessDataSource):
    def __init__(self, stories):
        self.stories = stories

    def __getitem__(self, idx):
        return self.stories[idx]

    def __len__(self):
        return len(self.stories)


def load_story(story: str):
    if rt.tokenizer is None:
        raise RuntimeError("tokenizer not set; call init_from_config(config, tokenizer=...)")

    encoding = rt.tokenizer.encode(story, allowed_special={"<|endoftext|>"})[: rt.config.model.max_len]
    encoding = encoding + [0] * (rt.config.model.max_len - len(encoding))

    inputs = jnp.array(encoding[:-1], dtype=jnp.int32)
    targets = jnp.array(encoding[1:], dtype=jnp.int32)
    return inputs, targets


def shard_stories(x):
    sharded_inputs = jax.device_put(
        x[0],
        jax.sharding.NamedSharding(
            rt.MESH,
            P(
                "x" if x[0].shape[0] % rt.devices.shape[0] == 0 else None,
                "y" if x[0].shape[1] % rt.devices.shape[1] == 0 else None,
            ),
        ),
    )

    sharded_targets = jax.device_put(
        x[1],
        jax.sharding.NamedSharding(
            rt.MESH,
            P(
                "x" if x[1].shape[0] % rt.devices.shape[0] == 0 else None,
                "y" if x[1].shape[1] % rt.devices.shape[1] == 0 else None,
            ),
        ),
    )
    return sharded_inputs, sharded_targets


def build_datasets(train_stories, val_stories, *, shuffle_seed: int = 677456347):
    train_source = DataSource(train_stories)
    train_dataset = (
        MapDataset.source(train_source)
        .shuffle(seed=shuffle_seed)
        .map(load_story)
        .batch(batch_size=rt.config.data.batch_size, drop_remainder=True)
        .map(shard_stories)
    )

    val_source = DataSource(val_stories)
    val_dataset = (
        MapDataset.source(val_source)
        .shuffle(seed=shuffle_seed)
        .map(load_story)
        .batch(batch_size=rt.config.sampling.batch_size, drop_remainder=True)
        .map(shard_stories)
    )
    return train_dataset, val_dataset
