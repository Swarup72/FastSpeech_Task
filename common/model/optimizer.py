"""FastSpeech 2 Scheduled Optimizer and Learning Rate Scheduler.

Implements the original FastSpeech 2 ScheduledOptim class, wrapping Adam
with the Transformer Noam learning rate schedule and step-based annealing:

    lr = init_lr * min(step ** -0.5, step * (warmup ** -1.5)) * (anneal_rate ** n_anneals)

Features:
    - Serializable scheduler state (current_step, init_lr, anneal settings)
    - Full checkpoint compatibility with original FastSpeech 2 checkpoints
    - Device-agnostic (CPU, CUDA, Apple Silicon MPS)
    - Config-object (TrainConfig, ModelConfig) and dictionary support
    - Pure-Python numerical calculations avoiding NumPy scalar overhead
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn as nn


class ScheduledOptim:
    """Scheduled optimizer for FastSpeech 2 training.

    Wraps a PyTorch Adam optimizer with a Noam learning-rate schedule
    incorporating linear warmup, inverse-square-root decay, and step-based
    annealing.
    """

    def __init__(
        self,
        model_or_params: nn.Module | Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        train_config: Any,
        model_config: Any,
        current_step: int = 0,
    ) -> None:
        """Initialize ScheduledOptim.

        Args:
            model_or_params: A PyTorch nn.Module or an iterable of parameters.
            train_config: TrainConfig dataclass or dictionary with optimizer settings.
            model_config: ModelConfig dataclass or dictionary with transformer hidden dim.
            current_step: Initial step count (e.g. when resuming training).
        """
        if hasattr(model_or_params, "parameters"):
            params = model_or_params.parameters()
        else:
            params = model_or_params

        # Extract optimizer settings from TrainConfig or dict
        opt_cfg = (
            train_config.optimizer
            if hasattr(train_config, "optimizer")
            else train_config.get("optimizer", train_config)
            if isinstance(train_config, Mapping)
            else train_config
        )

        betas = (
            list(opt_cfg.betas)
            if hasattr(opt_cfg, "betas")
            else list(opt_cfg.get("betas", [0.9, 0.98]))
            if isinstance(opt_cfg, Mapping)
            else [0.9, 0.98]
        )
        eps = float(
            opt_cfg.eps
            if hasattr(opt_cfg, "eps")
            else opt_cfg.get("eps", 1e-9)
            if isinstance(opt_cfg, Mapping)
            else 1e-9
        )
        weight_decay = float(
            opt_cfg.weight_decay
            if hasattr(opt_cfg, "weight_decay")
            else opt_cfg.get("weight_decay", 0.0)
            if isinstance(opt_cfg, Mapping)
            else 0.0
        )
        warm_up_step = int(
            opt_cfg.warm_up_step
            if hasattr(opt_cfg, "warm_up_step")
            else opt_cfg.get("warm_up_step", 4000)
            if isinstance(opt_cfg, Mapping)
            else 4000
        )
        anneal_steps = (
            list(opt_cfg.anneal_steps)
            if hasattr(opt_cfg, "anneal_steps")
            else list(opt_cfg.get("anneal_steps", []))
            if isinstance(opt_cfg, Mapping)
            else []
        )
        anneal_rate = float(
            opt_cfg.anneal_rate
            if hasattr(opt_cfg, "anneal_rate")
            else opt_cfg.get("anneal_rate", 0.3)
            if isinstance(opt_cfg, Mapping)
            else 0.3
        )

        # Extract transformer hidden dimension (d_model) from ModelConfig or dict
        if hasattr(model_config, "transformer"):
            hidden = model_config.transformer.encoder_hidden
        elif isinstance(model_config, Mapping) and "transformer" in model_config:
            t_cfg = model_config["transformer"]
            hidden = (
                t_cfg.encoder_hidden
                if hasattr(t_cfg, "encoder_hidden")
                else t_cfg["encoder_hidden"]
            )
        elif isinstance(model_config, Mapping) and "encoder_hidden" in model_config:
            hidden = model_config["encoder_hidden"]
        elif isinstance(model_config, (int, float)):
            hidden = int(model_config)
        else:
            hidden = 256

        self._optimizer = torch.optim.Adam(
            params,
            betas=(float(betas[0]), float(betas[1])),
            eps=eps,
            weight_decay=weight_decay,
        )

        self.n_warmup_steps: int = warm_up_step
        self.anneal_steps: list[int] = [int(s) for s in anneal_steps]
        self.anneal_rate: float = anneal_rate
        self.current_step: int = int(current_step)
        self.init_lr: float = math.pow(float(hidden), -0.5)

        # If starting from a nonzero step, set initial learning rate on param groups
        if self.current_step > 0:
            initial_lr = self.init_lr * self._get_lr_scale()
            for param_group in self._optimizer.param_groups:
                param_group["lr"] = initial_lr

    @property
    def optimizer(self) -> torch.optim.Adam:
        """Access underlying PyTorch Adam optimizer."""
        return self._optimizer

    def get_lr(self) -> float:
        """Return the current learning rate."""
        if self._optimizer.param_groups:
            return float(self._optimizer.param_groups[0]["lr"])
        return self.init_lr * self._get_lr_scale()

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero out gradients of optimized parameters."""
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        """Perform a single optimization step without updating learning rate."""
        self._optimizer.step()

    def step_and_update_lr(self) -> float:
        """Advance the schedule by one step, update learning rate, and step optimizer."""
        lr = self._update_learning_rate()
        self._optimizer.step()
        return lr

    def _get_lr_scale(self, step: int | None = None) -> float:
        """Calculate the Noam learning rate scale factor for a given step.

        Scale = min(step ** -0.5, step * (warmup ** -1.5)) * prod_{s < step}(anneal_rate)
        """
        s = self.current_step if step is None else step
        if s <= 0:
            return 0.0

        scale = min(
            math.pow(s, -0.5),
            math.pow(self.n_warmup_steps, -1.5) * s,
        )
        for threshold in self.anneal_steps:
            if s > threshold:
                scale *= self.anneal_rate
        return scale

    def _update_learning_rate(self) -> float:
        """Advance current_step by 1 and set learning rate on all param groups."""
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def state_dict(self) -> dict[str, Any]:
        """Serialize optimizer state and scheduler parameters."""
        return {
            "optimizer": self._optimizer.state_dict(),
            "current_step": self.current_step,
            "n_warmup_steps": self.n_warmup_steps,
            "anneal_steps": list(self.anneal_steps),
            "anneal_rate": self.anneal_rate,
            "init_lr": self.init_lr,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any] | Any) -> None:
        """Load optimizer and scheduler state.

        Supports:
        - Full ScheduledOptim state dict containing 'optimizer' and 'current_step'.
        - Raw PyTorch optimizer state dict (backward compatibility with original checkpoints).
        """
        if isinstance(state_dict, Mapping) and "optimizer" in state_dict:
            self._optimizer.load_state_dict(state_dict["optimizer"])
            if "current_step" in state_dict:
                self.current_step = int(state_dict["current_step"])
            if "n_warmup_steps" in state_dict:
                self.n_warmup_steps = int(state_dict["n_warmup_steps"])
            if "anneal_steps" in state_dict:
                self.anneal_steps = [int(s) for s in state_dict["anneal_steps"]]
            if "anneal_rate" in state_dict:
                self.anneal_rate = float(state_dict["anneal_rate"])
            if "init_lr" in state_dict:
                self.init_lr = float(state_dict["init_lr"])

            # Sync current learning rate to optimizer param groups
            if self.current_step > 0:
                current_lr = self.init_lr * self._get_lr_scale()
                for param_group in self._optimizer.param_groups:
                    param_group["lr"] = current_lr
        else:
            # Raw optimizer state dict
            self._optimizer.load_state_dict(state_dict)


__all__ = ["ScheduledOptim"]

