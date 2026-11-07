"""FastSpeech 2 Checkpoint Management System.

Provides robust, atomic checkpoint saving and resumption tailored for:
    - Long-running local training
    - Free Google Colab environments (interruption resilience, Google Drive paths)
    - Full reproducibility (state_dicts, step/epoch counters, loss history, RNG states)
    - Cross-platform portability (CPU, CUDA, Apple Silicon MPS via map_location)
    - Safe recovery from DataParallel 'module.' prefix discrepancies
"""

from __future__ import annotations

import os
import random
import re
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _clean_model_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    target_is_parallel: bool = False,
) -> dict[str, torch.Tensor]:
    """Ensure state_dict keys match target model's DataParallel wrapping.

    If target is not wrapped in DataParallel, strips leading 'module.' prefixes.
    If target is wrapped in DataParallel, ensures leading 'module.' prefixes exist.
    """
    cleaned: dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        if target_is_parallel:
            new_key = key if key.startswith("module.") else f"module.{key}"
        else:
            new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = val
    return cleaned


def _capture_rng_state() -> dict[str, Any]:
    """Capture current RNG states across PyTorch, NumPy, and Python random."""
    rng: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        try:
            rng["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return rng


def _restore_rng_state(rng: Mapping[str, Any] | None) -> None:
    """Restore saved RNG states across PyTorch, NumPy, and Python random."""
    if not rng:
        return
    if "torch" in rng:
        try:
            torch.set_rng_state(rng["torch"])
        except Exception:
            pass
    if "numpy" in rng:
        try:
            np.random.set_state(rng["numpy"])
        except Exception:
            pass
    if "python" in rng:
        try:
            random.setstate(rng["python"])
        except Exception:
            pass
    if "torch_cuda" in rng and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        except Exception:
            pass


class CheckpointManager:
    """Manages saving, loading, listing, and pruning checkpoints."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_to_keep: int | None = None,
    ) -> None:
        """Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory where checkpoints are saved and loaded.
                Works seamlessly with local folders and Google Drive paths.
            max_to_keep: Optional maximum number of step checkpoints to retain.
                Older step checkpoints are pruned to save disk space.
                'checkpoint_latest.pth.tar' and 'checkpoint_best.pth.tar' are never pruned.
        """
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep

    @property
    def latest_path(self) -> Path:
        """Path to the latest checkpoint file."""
        return self.checkpoint_dir / "checkpoint_latest.pth.tar"

    @property
    def best_path(self) -> Path:
        """Path to the best validation checkpoint file."""
        return self.checkpoint_dir / "checkpoint_best.pth.tar"

    def step_path(self, step: int) -> Path:
        """Path to a specific step checkpoint file."""
        return self.checkpoint_dir / f"checkpoint_{step}.pth.tar"

    def find_all_step_checkpoints(self) -> list[tuple[int, Path]]:
        """List all step checkpoints in directory sorted by step ascending.

        Supports both 'checkpoint_{step}.pth.tar' and original '{step}.pth.tar'.
        """
        checkpoints: list[tuple[int, Path]] = []
        if not self.checkpoint_dir.exists():
            return checkpoints

        for file in self.checkpoint_dir.iterdir():
            if not file.is_file():
                continue
            name = file.name
            if name in ("checkpoint_latest.pth.tar", "checkpoint_best.pth.tar"):
                continue

            # Check for 'checkpoint_{step}.pth.tar'
            m1 = re.match(r"^checkpoint_(\d+)\.pth\.tar$", name)
            if m1:
                checkpoints.append((int(m1.group(1)), file))
                continue

            # Check for original FastSpeech 2 format '{step}.pth.tar'
            m2 = re.match(r"^(\d+)\.pth\.tar$", name)
            if m2:
                checkpoints.append((int(m2.group(1)), file))

        checkpoints.sort(key=lambda x: x[0])
        return checkpoints

    def save(
        self,
        model: nn.Module,
        optimizer: Any,
        epoch: int,
        global_step: int,
        best_val_loss: float = float("inf"),
        history: dict[str, Any] | None = None,
        is_best: bool = False,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Atomically save training state to checkpoint files.

        Writes first to a temporary file in the same directory, then replaces
        the target file. This prevents corrupted partial files on abrupt termination.

        Args:
            model: PyTorch model.
            optimizer: Optimizer or ScheduledOptim instance.
            epoch: Current epoch number.
            global_step: Current global training step.
            best_val_loss: Lowest validation loss achieved so far.
            history: Optional training/validation loss history.
            is_best: Whether this checkpoint achieved a new best validation loss.
            extra_metadata: Optional additional metadata dictionary.

        Returns:
            Path to the saved step checkpoint.
        """
        # Unwrap DataParallel if present so saved state_dict is universally usable
        raw_model = model.module if hasattr(model, "module") else model
        model_state = _clean_model_state_dict(
            raw_model.state_dict(), target_is_parallel=False
        )

        # Get optimizer state
        if hasattr(optimizer, "state_dict"):
            optimizer_state = optimizer.state_dict()
        else:
            optimizer_state = None

        checkpoint_data: dict[str, Any] = {
            "model": model_state,
            "optimizer": optimizer_state,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_val_loss": float(best_val_loss),
            "history": history or {},
            "rng_state": _capture_rng_state(),
            "timestamp": time.time(),
        }
        if extra_metadata:
            checkpoint_data["metadata"] = extra_metadata

        # Atomic write helper
        def _atomic_torch_save(target_path: Path, data: dict[str, Any]) -> None:
            temp_file = target_path.with_suffix(".tmp")
            try:
                torch.save(data, temp_file)
                os.replace(temp_file, target_path)
            finally:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except OSError:
                        pass

        # 1. Save periodic step checkpoint
        step_file = self.step_path(global_step)
        _atomic_torch_save(step_file, checkpoint_data)

        # 2. Always update latest checkpoint
        _atomic_torch_save(self.latest_path, checkpoint_data)

        # 3. If best validation loss, update best checkpoint
        if is_best:
            _atomic_torch_save(self.best_path, checkpoint_data)

        # 4. Prune older step checkpoints if max_to_keep is set
        if self.max_to_keep is not None and self.max_to_keep > 0:
            step_checkpoints = self.find_all_step_checkpoints()
            if len(step_checkpoints) > self.max_to_keep:
                to_delete = step_checkpoints[: -self.max_to_keep]
                for _, file_path in to_delete:
                    try:
                        file_path.unlink()
                    except OSError:
                        pass

        return step_file

    def resolve_checkpoint_path(
        self,
        checkpoint_target: str | Path | int | None = None,
    ) -> Path:
        """Resolve target step, path, or 'latest' to an existing checkpoint Path.

        Args:
            checkpoint_target: Step number (int), file path (str/Path), or None
                (which resolves to the latest available checkpoint).

        Returns:
            Existing Path to checkpoint.

        Raises:
            FileNotFoundError: If no matching checkpoint exists.
        """
        if checkpoint_target is None:
            if self.latest_path.exists():
                return self.latest_path
            step_checkpoints = self.find_all_step_checkpoints()
            if step_checkpoints:
                return step_checkpoints[-1][1]
            raise FileNotFoundError(
                f"No checkpoints found in directory '{self.checkpoint_dir}'."
            )

        if isinstance(checkpoint_target, int):
            # Check for 'checkpoint_{step}.pth.tar' first
            cand1 = self.step_path(checkpoint_target)
            if cand1.exists():
                return cand1
            # Check for original '{step}.pth.tar'
            cand2 = self.checkpoint_dir / f"{checkpoint_target}.pth.tar"
            if cand2.exists():
                return cand2
            raise FileNotFoundError(
                f"No checkpoint found for step {checkpoint_target} in '{self.checkpoint_dir}'."
            )

        target_path = Path(checkpoint_target).resolve()
        if not target_path.exists():
            # Try looking inside checkpoint_dir if relative
            cand3 = self.checkpoint_dir / checkpoint_target
            if cand3.exists():
                return cand3
            raise FileNotFoundError(f"Checkpoint file not found: '{checkpoint_target}'.")
        return target_path

    def load(
        self,
        checkpoint_target: str | Path | int | None = None,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """Load a checkpoint dictionary with device portability.

        Args:
            checkpoint_target: Step number, path, or None for latest.
            map_location: Device to map loaded tensors to (defaults to 'cpu').

        Returns:
            Loaded checkpoint dictionary.
        """
        path = self.resolve_checkpoint_path(checkpoint_target)
        if map_location is None:
            map_location = torch.device("cpu")

        load_kwargs: dict[str, Any] = {"map_location": map_location}
        # In PyTorch >= 2.4 / 2.6+, weights_only defaults to True. Full training
        # checkpoints contain metadata, optimizer dicts, and RNG states that require weights_only=False.
        try:
            import inspect

            if "weights_only" in inspect.signature(torch.load).parameters:
                load_kwargs["weights_only"] = False
        except Exception:
            pass

        try:
            checkpoint = torch.load(path, **load_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint file at '{path}': {e}"
            ) from e

        if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
            raise ValueError(
                f"Checkpoint at '{path}' is invalid: missing 'model' key."
            )

        return checkpoint

    def restore_into(
        self,
        model: nn.Module,
        optimizer: Any | None = None,
        checkpoint_target: str | Path | int | None = None,
        map_location: str | torch.device | None = None,
        strict_model: bool = True,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """Load checkpoint and restore state directly into model and optimizer.

        Args:
            model: Target PyTorch model.
            optimizer: Optional target optimizer or ScheduledOptim.
            checkpoint_target: Step number, path, or None for latest.
            map_location: Target device for loaded tensors.
            strict_model: Whether to strictly enforce matching keys in model.
            restore_rng: Whether to restore saved random-generator states.

        Returns:
            The loaded checkpoint dictionary for accessing epoch, step, and history.
        """
        checkpoint = self.load(checkpoint_target, map_location=map_location)

        # 1. Restore model weights
        target_is_parallel = hasattr(model, "module")
        cleaned_weights = _clean_model_state_dict(
            checkpoint["model"], target_is_parallel=target_is_parallel
        )
        model.load_state_dict(cleaned_weights, strict=strict_model)

        # 2. Restore optimizer and scheduler
        if optimizer is not None and "optimizer" in checkpoint:
            opt_data = checkpoint["optimizer"]
            if opt_data is not None:
                optimizer.load_state_dict(opt_data)

        # 3. Restore RNG state
        if restore_rng and "rng_state" in checkpoint:
            _restore_rng_state(checkpoint["rng_state"])

        return checkpoint


__all__ = [
    "CheckpointManager",
    "_clean_model_state_dict",
    "_capture_rng_state",
    "_restore_rng_state",
]
