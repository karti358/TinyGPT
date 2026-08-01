"""Orbax checkpoint helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import jax
import orbax.checkpoint as ocp
from flax import nnx


def load_checkpoint(
    model: nnx.Module,
    checkpoint_path: str | Path,
    mesh: jax.sharding.Mesh | None = None,
) -> nnx.Module:
    """Restore model state from an Orbax checkpoint directory."""
    checkpoint_path = Path(checkpoint_path)
    checkpointer = ocp.StandardCheckpointer()

    abstract_model = nnx.eval_shape(lambda: model)
    graphdef, abstract_state = nnx.split(abstract_model)

    if mesh:
        abstract_state = jax.tree.map(
            lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s),
            abstract_state,
            nnx.get_named_sharding(abstract_state, mesh),
        )

    restored_state = checkpointer.restore(checkpoint_path, abstract_state)
    return nnx.merge(graphdef, restored_state)


def find_latest_checkpoint(base_dir: str | Path = "checkpoints") -> Path:
    """Return the most recently modified checkpoint under base_dir."""
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {base_dir}")

    candidates = [p for p in base_dir.rglob("*") if p.is_dir() and (p / "_CHECKPOINT_METADATA").exists()]
    if not candidates:
        candidates = [p for p in base_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {base_dir}")
    
    candidates = [Path.cwd() / p for p in candidates]
    return max(candidates, key=lambda p: p.stat().st_mtime)


class ManageCheckpoints:
    def __init__(self, base_dir: str):
        self.ckpt_dir = ocp.test_utils.erase_and_create_empty(base_dir)
        self.checkpointer = ocp.StandardCheckpointer()

    def save_model(self, model: nnx.Module) -> str:
        _, state = nnx.split(model)
        new_dir = self.ckpt_dir / f"{int(datetime.utcnow().timestamp())}_state"
        self.checkpointer.save(new_dir, state)
        return str(new_dir)

    def load_model(self, model: nnx.Module, checkpoint: str, mesh: jax.sharding.Mesh | None = None) -> nnx.Module:
        return load_checkpoint(model, self.ckpt_dir / checkpoint, mesh=mesh)
