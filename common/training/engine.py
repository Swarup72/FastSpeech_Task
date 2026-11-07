"""FastSpeech 2 Training Engine.

Provides an end-to-end, reusable training and validation pipeline:
    - Modular train_step and validate_step primitives
    - Full train_epoch and validate loops
    - Gradient accumulation and gradient clipping
    - Scheduled optimizer stepping and learning rate logging
    - Checkpoint saving and resumption with CheckpointManager
    - Cross-platform support for CPU, Apple Silicon MPS, and CUDA
    - Google Colab resilience (interruption recovery, drive path handling)
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from common.model.fastspeech2 import FastSpeech2
from common.model.loss import FastSpeech2Loss
from common.model.optimizer import ScheduledOptim
from common.training.checkpoint import CheckpointManager
from common.utils.tools import to_device


@dataclass
class TrainLosses:
    """Container for FastSpeech 2 multi-task losses and step metrics."""

    total: float
    mel: float
    postnet_mel: float
    pitch: float
    energy: float
    duration: float
    lr: float | None = None
    grad_norm: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """Convert loss container to dictionary."""
        return asdict(self)

    @classmethod
    def from_tuple(
        cls,
        losses_tuple: Sequence[torch.Tensor] | Sequence[float],
        lr: float | None = None,
        grad_norm: float | None = None,
    ) -> TrainLosses:
        """Create TrainLosses from a 6-element tuple of loss tensors or floats."""
        vals = [
            float(l.item()) if hasattr(l, "item") else float(l)
            for l in losses_tuple[:6]
        ]
        return cls(
            total=vals[0],
            mel=vals[1],
            postnet_mel=vals[2],
            pitch=vals[3],
            energy=vals[4],
            duration=vals[5],
            lr=lr,
            grad_norm=grad_norm,
        )


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve compute device prioritizing CUDA, then Apple Silicon MPS, then CPU."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """FastSpeech 2 Training Engine."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: FastSpeech2Loss | nn.Module,
        optimizer: ScheduledOptim | torch.optim.Optimizer,
        train_config: Any | None = None,
        device: str | torch.device | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        checkpoint_dir: str | Path | None = None,
        grad_clip_thresh: float = 1.0,
        grad_acc_step: int = 1,
    ) -> None:
        """Initialize the Trainer.

        Args:
            model: FastSpeech 2 model instance.
            loss_fn: FastSpeech2Loss instance.
            optimizer: ScheduledOptim or standard PyTorch optimizer.
            train_config: Optional TrainConfig or dictionary for default settings.
            device: Compute device ('cpu', 'mps', 'cuda', or torch.device).
            checkpoint_manager: Optional existing CheckpointManager instance.
            checkpoint_dir: Directory path for checkpoint management if checkpoint_manager is not given.
            grad_clip_thresh: Maximum gradient norm for clipping.
            grad_acc_step: Number of gradient accumulation steps.
        """
        self.device = _resolve_device(device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = optimizer

        # Extract settings from train_config if provided
        if train_config is not None:
            opt_cfg = (
                train_config.optimizer
                if hasattr(train_config, "optimizer")
                else train_config.get("optimizer", {})
                if isinstance(train_config, Mapping)
                else {}
            )
            self.grad_clip_thresh = float(
                opt_cfg.grad_clip_thresh
                if hasattr(opt_cfg, "grad_clip_thresh")
                else opt_cfg.get("grad_clip_thresh", grad_clip_thresh)
                if isinstance(opt_cfg, Mapping)
                else grad_clip_thresh
            )
            self.grad_acc_step = int(
                opt_cfg.grad_acc_step
                if hasattr(opt_cfg, "grad_acc_step")
                else opt_cfg.get("grad_acc_step", grad_acc_step)
                if isinstance(opt_cfg, Mapping)
                else grad_acc_step
            )
            if checkpoint_dir is None and hasattr(train_config, "path"):
                checkpoint_dir = getattr(train_config.path, "ckpt_path", None)
            elif (
                checkpoint_dir is None
                and isinstance(train_config, Mapping)
                and "path" in train_config
            ):
                checkpoint_dir = train_config["path"].get("ckpt_path")
        else:
            self.grad_clip_thresh = float(grad_clip_thresh)
            self.grad_acc_step = int(grad_acc_step)

        # Initialize CheckpointManager
        if checkpoint_manager is not None:
            self.checkpoint_manager = checkpoint_manager
        elif checkpoint_dir is not None:
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        else:
            self.checkpoint_manager = None

        # Training progress counters
        self.epoch: int = 1
        self.global_step: int = 0
        self.accumulation_counter: int = 0
        self.best_val_loss: float = float("inf")

        # History tracking
        self.history: dict[str, list[dict[str, Any]]] = {
            "train_loss": [],
            "val_loss": [],
        }

    def train_step(self, batch: Any) -> TrainLosses:
        """Execute a single training step on a batch.

        Handles:
            1. Device transfer
            2. Forward pass
            3. FastSpeech2Loss calculation
            4. Gradient accumulation scaling
            5. Backward pass
            6. Gradient clipping and optimizer step on accumulation boundaries
            7. Loss and metric tracking

        Args:
            batch: FastSpeech 2 batch (12-tuple or mapping).

        Returns:
            TrainLosses dataclass with all 6 loss terms, learning rate, and grad norm.
        """
        self.model.train()
        batch_dev = to_device(batch, self.device)

        # Forward pass
        if isinstance(batch_dev, (list, tuple)):
            predictions = self.model(*(batch_dev[2:]))
        elif isinstance(batch_dev, Mapping):
            predictions = self.model(**batch_dev)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch_dev).__name__}")

        # Compute multi-task losses
        losses = self.loss_fn(batch_dev, predictions)
        total_loss = losses[0]

        # Verify finite loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Encountered non-finite loss at step {self.global_step}: total_loss={total_loss.item()}"
            )

        # Scale loss for gradient accumulation and backpropagate
        scaled_loss = total_loss / self.grad_acc_step
        scaled_loss.backward()

        self.accumulation_counter += 1
        grad_norm_val: float | None = None

        # Optimizer step on accumulation boundary
        if self.accumulation_counter % self.grad_acc_step == 0:
            if self.grad_clip_thresh > 0.0:
                gn = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_thresh
                )
                grad_norm_val = float(gn.item()) if hasattr(gn, "item") else float(gn)

            if hasattr(self.optimizer, "step_and_update_lr"):
                self.optimizer.step_and_update_lr()
            else:
                self.optimizer.step()

            if hasattr(self.optimizer, "zero_grad"):
                self.optimizer.zero_grad()

            self.global_step += 1

        # Current learning rate
        current_lr: float | None = None
        if hasattr(self.optimizer, "get_lr"):
            current_lr = self.optimizer.get_lr()
        elif hasattr(self.optimizer, "param_groups") and self.optimizer.param_groups:
            current_lr = float(self.optimizer.param_groups[0].get("lr", 0.0))

        return TrainLosses.from_tuple(
            losses, lr=current_lr, grad_norm=grad_norm_val
        )

    def validate_step(self, batch: Any) -> tuple[TrainLosses, int]:
        """Execute a single evaluation step on a batch without gradients.

        Args:
            batch: FastSpeech 2 batch (12-tuple or mapping).

        Returns:
            Tuple of (TrainLosses, batch_sample_count).
        """
        self.model.eval()
        batch_dev = to_device(batch, self.device)

        with torch.no_grad():
            if isinstance(batch_dev, (list, tuple)):
                predictions = self.model(*(batch_dev[2:]))
                sample_count = len(batch_dev[0]) if len(batch_dev) > 0 else 1
            elif isinstance(batch_dev, Mapping):
                predictions = self.model(**batch_dev)
                sample_count = (
                    batch_dev["texts"].shape[0] if "texts" in batch_dev else 1
                )
            else:
                raise TypeError(f"Unsupported batch type: {type(batch_dev).__name__}")

            losses = self.loss_fn(batch_dev, predictions)

        return TrainLosses.from_tuple(losses), sample_count

    def train_epoch(
        self,
        dataloader: Iterable[Any],
        epoch: int | None = None,
        step_callback: Callable[[int, TrainLosses], None] | None = None,
    ) -> TrainLosses:
        """Run a complete training epoch across a DataLoader.

        Supports standard loaders and grouped loaders (where each item is a list of batches).

        Args:
            dataloader: Training dataloader or iterable of batches.
            epoch: Optional epoch number override.
            step_callback: Optional callback invoked after each step: fn(step, losses).

        Returns:
            Aggregated average TrainLosses across the epoch.
        """
        if epoch is not None:
            self.epoch = epoch

        self.model.train()
        loss_sums = [0.0] * 6
        step_count = 0

        for item in dataloader:
            # Handle grouped batches (from original FastSpeech 2 dataset collation)
            batches = item if isinstance(item, list) and len(item) > 0 and isinstance(item[0], (tuple, list)) else [item]

            for batch in batches:
                step_loss = self.train_step(batch)
                loss_sums[0] += step_loss.total
                loss_sums[1] += step_loss.mel
                loss_sums[2] += step_loss.postnet_mel
                loss_sums[3] += step_loss.pitch
                loss_sums[4] += step_loss.energy
                loss_sums[5] += step_loss.duration
                step_count += 1

                if step_callback is not None:
                    step_callback(self.global_step, step_loss)

        if step_count == 0:
            raise ValueError("Training DataLoader yielded 0 batches.")

        avg_losses = [s / step_count for s in loss_sums]
        current_lr = (
            self.optimizer.get_lr()
            if hasattr(self.optimizer, "get_lr")
            else None
        )
        epoch_losses = TrainLosses.from_tuple(avg_losses, lr=current_lr)

        self.history["train_loss"].append(
            {"epoch": self.epoch, "step": self.global_step, **epoch_losses.to_dict()}
        )
        self.epoch += 1
        return epoch_losses

    def validate(
        self,
        dataloader: Iterable[Any],
    ) -> TrainLosses:
        """Run evaluation across a validation DataLoader.

        Computes sample-weighted average loss across the validation set, matching
        original FastSpeech 2 evaluate.py behavior.

        Args:
            dataloader: Validation dataloader.

        Returns:
            Weighted average TrainLosses across the validation set.
        """
        self.model.eval()
        loss_sums = [0.0] * 6
        total_samples = 0

        for item in dataloader:
            batches = item if isinstance(item, list) and len(item) > 0 and isinstance(item[0], (tuple, list)) else [item]

            for batch in batches:
                step_loss, count = self.validate_step(batch)
                loss_sums[0] += step_loss.total * count
                loss_sums[1] += step_loss.mel * count
                loss_sums[2] += step_loss.postnet_mel * count
                loss_sums[3] += step_loss.pitch * count
                loss_sums[4] += step_loss.energy * count
                loss_sums[5] += step_loss.duration * count
                total_samples += count

        if total_samples == 0:
            raise ValueError("Validation DataLoader yielded 0 samples.")

        avg_losses = [s / total_samples for s in loss_sums]
        val_losses = TrainLosses.from_tuple(avg_losses)

        # Update best validation loss
        is_best = val_losses.total < self.best_val_loss
        if is_best:
            self.best_val_loss = val_losses.total

        self.history["val_loss"].append(
            {
                "epoch": self.epoch,
                "step": self.global_step,
                "is_best": is_best,
                **val_losses.to_dict(),
            }
        )
        return val_losses

    def save_checkpoint(
        self,
        is_best: bool = False,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save a training checkpoint via CheckpointManager.

        Args:
            is_best: Whether this checkpoint represents the best validation loss.
            extra_metadata: Optional additional metadata to save.

        Returns:
            Path to the saved checkpoint file.
        """
        if self.checkpoint_manager is None:
            raise RuntimeError(
                "Cannot save checkpoint: neither checkpoint_manager nor checkpoint_dir was provided."
            )

        return self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.epoch,
            global_step=self.global_step,
            best_val_loss=self.best_val_loss,
            history=self.history,
            is_best=is_best,
            extra_metadata=extra_metadata,
        )

    def resume_from_checkpoint(
        self,
        checkpoint_target: str | Path | int | None = None,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """Restore model, optimizer, scheduler, step, epoch, and RNG states.

        Args:
            checkpoint_target: Step number, path, or None to load the latest checkpoint.
            restore_rng: Whether to restore random number generator states.

        Returns:
            The loaded checkpoint dictionary.
        """
        if self.checkpoint_manager is None:
            raise RuntimeError(
                "Cannot resume: neither checkpoint_manager nor checkpoint_dir was provided."
            )

        ckpt = self.checkpoint_manager.restore_into(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_target=checkpoint_target,
            map_location=self.device,
            restore_rng=restore_rng,
        )

        if "epoch" in ckpt:
            self.epoch = int(ckpt["epoch"])
        if "global_step" in ckpt:
            self.global_step = int(ckpt["global_step"])
        if "best_val_loss" in ckpt:
            self.best_val_loss = float(ckpt["best_val_loss"])
        if "history" in ckpt and isinstance(ckpt["history"], dict):
            self.history = ckpt["history"]

        return ckpt


__all__ = ["Trainer", "TrainLosses", "_resolve_device"]

