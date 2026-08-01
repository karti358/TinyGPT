"""Training loop, loss, and optimizer setup."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

import jax
import numpy as np
import optax
import tqdm
from flax import nnx

from flash_gpt import runtime as rt
from flash_gpt.checkpoints import ManageCheckpoints
from flash_gpt.model.gpt import TinyGPT
from flash_gpt.sample import sample


def loss_fn(model: TinyGPT, batch: Sequence[jax.Array]):
    logits = model(batch[0])
    return optax.softmax_cross_entropy_with_integer_labels(logits, batch[1]).mean()


grad_loss_fn = nnx.value_and_grad(loss_fn)


@nnx.jit
def train_step(model: TinyGPT, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    loss, grads = grad_loss_fn(model, batch)
    optimizer.update(model, grads)
    metrics.update(loss=loss)


@nnx.jit
def eval_step(model: TinyGPT, metrics: nnx.MultiMetric, batch):
    loss = loss_fn(model, batch)
    metrics.update(loss=loss)


def create_sharded_model():
    model = TinyGPT(rngs=nnx.Rngs(rt.config.model.seed))
    opt = optax.adam(learning_rate=rt.config.training.lr)
    optimizer = nnx.Optimizer(model, opt, wrt=nnx.Param)

    model_state = nnx.state(model, nnx.Param, nnx.Variable)
    param_shardings = nnx.get_named_sharding(model_state[0], rt.MESH)
    var_shardings = nnx.get_named_sharding(model_state[1], rt.MESH)

    param_sharded_state = jax.lax.with_sharding_constraint(model_state[0], param_shardings)
    var_sharded_state = jax.lax.with_sharding_constraint(model_state[1], var_shardings)
    nnx.update(model, param_sharded_state, var_sharded_state)

    optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
    optimizer_shardings = nnx.get_named_sharding(optimizer_state, rt.MESH)
    optimizer_sharded_state = jax.lax.with_sharding_constraint(optimizer_state, optimizer_shardings)
    nnx.update(optimizer, optimizer_sharded_state)
    return model, optimizer


def train(
    epochs: int,
    model: TinyGPT,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    train_dataset,
    val_dataset,
    train_metrics: dict,
    val_metrics: dict,
    sample_dir: str,
    checkpoint_manager: ManageCheckpoints | None = None,
    *,
    run: Any = None,
    tokenizer=None,
):
    table = None
    if run is not None:
        import wandb

        table = wandb.Table(columns=["actual", "completions"], log_mode="MUTABLE")

    train_key = jax.random.key(rt.config.model.seed)

    for epoch in range(1, epochs + 1):
        model.train()
        with tqdm.tqdm(train_dataset, unit="batch") as train_epochs:
            for index, batch in enumerate(train_epochs):
                step = len(train_dataset) * (epoch - 1) + index
                train_epochs.set_description(f"Epoch {epoch} | Training | ")

                train_step(model=model, optimizer=optimizer, metrics=metrics, batch=batch)

                batch_metrics = metrics.compute()
                for metric, value in batch_metrics.items():
                    train_metrics[metric].append(value)
                metrics.reset()

                if run is not None:
                    run.log({"train_loss": batch_metrics["loss"]})
                train_epochs.set_postfix(**{f"{k}": v for k, v in batch_metrics.items()})

                if (step + 1) % rt.config.training.save_and_sample_every == 0 and checkpoint_manager:
                    checkpoint_dir = checkpoint_manager.save_model(model)
                    max_retries = 30
                    while not os.path.isdir(checkpoint_dir) and max_retries > 0:
                        print("Waiting for Checkpoint manager")
                        time.sleep(15)
                        max_retries -= 1

                    if run is not None and os.path.isdir(checkpoint_dir):
                        import wandb

                        artifact = wandb.Artifact(name=f"model_{step + 1}", type="jax_flash_minigpt")
                        artifact.add_dir(checkpoint_dir)
                        run.log_artifact(artifact)
                        print(f"Saved checkpoint at {checkpoint_dir}")

                    if tokenizer is not None and table is not None:
                        train_key, subkey = jax.random.split(train_key)
                        index = jax.random.randint(subkey, shape=(), minval=0, maxval=len(val_dataset))
                        sample_batch = val_dataset[index]
                        sample_arrs = jax.block_until_ready(sample(model, sample_batch))
                        host_arrs = jax.device_get(sample_arrs)
                        for i in range(rt.config.sampling.batch_size):
                            table.add_data(
                                tokenizer.decode(sample_batch[0][i, ...]),
                                tokenizer.decode(host_arrs[i, ...]),
                            )
                        run.log({"completions_table": table})

        if checkpoint_manager:
            checkpoint_dir = checkpoint_manager.save_model(model)

        model.eval()
        with tqdm.tqdm(val_dataset[:2000], unit="batch") as val_epochs:
            for batch in val_epochs:
                val_epochs.set_description(f"Epoch {epoch} | Validation | ")
                eval_step(model=model, metrics=metrics, batch=batch)
                batch_metrics = metrics.compute()
                for metric, value in batch_metrics.items():
                    val_metrics[metric].append(value)
                metrics.reset()
                if run is not None:
                    run.log({"val_loss": batch_metrics["loss"]})
                val_epochs.set_postfix(**{f"{k}": v for k, v in batch_metrics.items()})

        print(f"\nEpoch {epoch} Summary:")
        for metric in train_metrics.keys():
            train_avg = np.mean(train_metrics[metric][-len(train_dataset) :])
            val_avg = np.mean(val_metrics[metric][-len(val_dataset) :])
            print(f"Model {metric} | Train: {train_avg:.4f} | Val: {val_avg:.4f}")

    return train_metrics, val_metrics
