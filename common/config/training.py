"""Training configuration structures (`train.yaml`) for FastSpeech 2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.config.base import (
    ConfigNode,
    ConfigValidationError,
    _load_yaml_dict,
    _require_field,
    _require_mapping,
    _validate_dropout,
    _validate_non_empty_str,
    _validate_positive_float,
    _validate_positive_int,
)


@dataclass
class TrainPathConfig(ConfigNode):
    """Checkpoint, TensorBoard log, and synthesis result paths for training."""

    ckpt_path: str
    log_path: str
    result_path: str

    def __post_init__(self) -> None:
        self.ckpt_path = _validate_non_empty_str(
            self.ckpt_path, "path.ckpt_path"
        )
        self.log_path = _validate_non_empty_str(self.log_path, "path.log_path")
        self.result_path = _validate_non_empty_str(
            self.result_path, "path.result_path"
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainPathConfig:
        mapping = _require_mapping(data, "train_config.path")
        return cls(
            ckpt_path=_require_field(mapping, "ckpt_path", "train_config.path"),
            log_path=_require_field(mapping, "log_path", "train_config.path"),
            result_path=_require_field(
                mapping, "result_path", "train_config.path"
            ),
        )


@dataclass
class OptimizerConfig(ConfigNode):
    """Adam optimizer, gradient accumulation, warmup, and learning-rate annealing settings."""

    batch_size: int = 16
    betas: list[float] = field(default_factory=lambda: [0.9, 0.98])
    eps: float = 1e-9
    weight_decay: float = 0.0
    grad_clip_thresh: float = 1.0
    grad_acc_step: int = 1
    warm_up_step: int = 4000
    anneal_steps: list[int] = field(default_factory=list)
    anneal_rate: float = 0.3

    def __post_init__(self) -> None:
        self.batch_size = _validate_positive_int(
            self.batch_size, "optimizer.batch_size"
        )
        if not isinstance(self.betas, (list, tuple)) or len(self.betas) != 2:
            raise ConfigValidationError(
                f"'optimizer.betas' must be a 2-element list/tuple, got {self.betas!r}."
            )
        b0 = _validate_dropout(self.betas[0], "optimizer.betas[0]")
        b1 = _validate_dropout(self.betas[1], "optimizer.betas[1]")
        self.betas = [b0, b1]

        self.eps = _validate_positive_float(self.eps, "optimizer.eps")
        self.weight_decay = _validate_positive_float(
            self.weight_decay, "optimizer.weight_decay", allow_zero=True
        )
        self.grad_clip_thresh = _validate_positive_float(
            self.grad_clip_thresh, "optimizer.grad_clip_thresh"
        )
        self.grad_acc_step = _validate_positive_int(
            self.grad_acc_step, "optimizer.grad_acc_step"
        )
        self.warm_up_step = _validate_positive_int(
            self.warm_up_step, "optimizer.warm_up_step"
        )

        if not isinstance(self.anneal_steps, (list, tuple)):
            raise ConfigValidationError(
                f"'optimizer.anneal_steps' must be a list of positive integers, got {self.anneal_steps!r}."
            )
        cleaned_steps: list[int] = []
        for idx, step_val in enumerate(self.anneal_steps):
            s = _validate_positive_int(
                step_val, f"optimizer.anneal_steps[{idx}]"
            )
            if cleaned_steps and s <= cleaned_steps[-1]:
                raise ConfigValidationError(
                    f"'optimizer.anneal_steps' must be strictly increasing, got {self.anneal_steps!r}."
                )
            cleaned_steps.append(s)
        self.anneal_steps = cleaned_steps

        self.anneal_rate = _validate_positive_float(
            self.anneal_rate, "optimizer.anneal_rate"
        )
        if self.anneal_rate > 1.0:
            raise ConfigValidationError(
                f"'optimizer.anneal_rate' must be in (0.0, 1.0], got {self.anneal_rate}."
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizerConfig:
        mapping = _require_mapping(data, "train_config.optimizer")
        return cls(
            batch_size=mapping.get("batch_size", 16),
            betas=list(mapping.get("betas", [0.9, 0.98])),
            eps=mapping.get("eps", 1e-9),
            weight_decay=mapping.get("weight_decay", 0.0),
            grad_clip_thresh=mapping.get("grad_clip_thresh", 1.0),
            grad_acc_step=mapping.get("grad_acc_step", 1),
            warm_up_step=mapping.get("warm_up_step", 4000),
            anneal_steps=list(mapping.get("anneal_steps", [])),
            anneal_rate=mapping.get("anneal_rate", 0.3),
        )


@dataclass
class StepConfig(ConfigNode):
    """Step intervals for training, logging, validation, synthesis, and checkpointing."""

    total_step: int = 900000
    log_step: int = 100
    synth_step: int = 1000
    val_step: int = 1000
    save_step: int = 100000

    def __post_init__(self) -> None:
        self.total_step = _validate_positive_int(
            self.total_step, "step.total_step"
        )
        self.log_step = _validate_positive_int(self.log_step, "step.log_step")
        self.synth_step = _validate_positive_int(
            self.synth_step, "step.synth_step"
        )
        self.val_step = _validate_positive_int(self.val_step, "step.val_step")
        self.save_step = _validate_positive_int(
            self.save_step, "step.save_step"
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StepConfig:
        mapping = _require_mapping(data, "train_config.step")
        return cls(
            total_step=mapping.get("total_step", 900000),
            log_step=mapping.get("log_step", 100),
            synth_step=mapping.get("synth_step", 1000),
            val_step=mapping.get("val_step", 1000),
            save_step=mapping.get("save_step", 100000),
        )


@dataclass
class TrainConfig(ConfigNode):
    """Top-level training configuration (`train.yaml`)."""

    path: TrainPathConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    step: StepConfig = field(default_factory=StepConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.path, TrainPathConfig):
            self.path = TrainPathConfig.from_dict(self.path)
        if not isinstance(self.optimizer, OptimizerConfig):
            self.optimizer = OptimizerConfig.from_dict(self.optimizer)
        if not isinstance(self.step, StepConfig):
            self.step = StepConfig.from_dict(self.step)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainConfig:
        mapping = _require_mapping(data, "train_config")
        path_raw = _require_field(mapping, "path", "train_config")
        optimizer_raw = _require_field(mapping, "optimizer", "train_config")
        step_raw = _require_field(mapping, "step", "train_config")
        return cls(
            path=TrainPathConfig.from_dict(path_raw),
            optimizer=OptimizerConfig.from_dict(optimizer_raw),
            step=StepConfig.from_dict(step_raw),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        return cls.from_dict(_load_yaml_dict(path))


__all__ = [
    "TrainPathConfig",
    "OptimizerConfig",
    "StepConfig",
    "TrainConfig",
]

