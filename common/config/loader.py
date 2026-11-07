"""Complete pipeline configuration bundle (`FastSpeech2Config`) and YAML loader functions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.config.base import ConfigNode, ConfigValidationError
from common.config.model import ModelConfig
from common.config.preprocess import PreprocessConfig
from common.config.training import TrainConfig


@dataclass
class FastSpeech2Config(ConfigNode):
    """Unified configuration bundle validating cross-file constraints across preprocess, model, and train."""

    preprocess: PreprocessConfig
    model: ModelConfig
    train: TrainConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.preprocess, PreprocessConfig):
            self.preprocess = PreprocessConfig.from_dict(self.preprocess)
        if not isinstance(self.model, ModelConfig):
            self.model = ModelConfig.from_dict(self.model)
        if self.train is not None and not isinstance(self.train, TrainConfig):
            self.train = TrainConfig.from_dict(self.train)

        # Cross-config validation (DEC-04 / B-03):
        # 'log' quantization is invalid when pitch or energy values are Z-score normalized
        if (
            self.preprocess.preprocessing.pitch.normalization
            and self.model.variance_embedding.pitch_quantization == "log"
        ):
            raise ConfigValidationError(
                "Invalid configuration combination: 'variance_embedding.pitch_quantization' cannot be "
                "'log' when 'preprocessing.pitch.normalization' is True (normalized pitch contains "
                "negative values, causing log quantization bins to become NaN; see DEC-04 / B-03)."
            )
        if (
            self.preprocess.preprocessing.energy.normalization
            and self.model.variance_embedding.energy_quantization == "log"
        ):
            raise ConfigValidationError(
                "Invalid configuration combination: 'variance_embedding.energy_quantization' cannot be "
                "'log' when 'preprocessing.energy.normalization' is True (normalized energy contains "
                "negative values, causing log quantization bins to become NaN; see DEC-04 / B-03)."
            )

    @classmethod
    def from_dicts(
        cls,
        preprocess_dict: Mapping[str, Any],
        model_dict: Mapping[str, Any],
        train_dict: Mapping[str, Any] | None = None,
        *,
        n_src_vocab: int | None = None,
    ) -> FastSpeech2Config:
        preprocess_cfg = PreprocessConfig.from_dict(preprocess_dict)
        model_cfg = ModelConfig.from_dict(model_dict, n_src_vocab=n_src_vocab)
        train_cfg = (
            TrainConfig.from_dict(train_dict)
            if train_dict is not None
            else None
        )
        return cls(preprocess=preprocess_cfg, model=model_cfg, train=train_cfg)

    @classmethod
    def from_yaml_paths(
        cls,
        preprocess_yaml: str | Path,
        model_yaml: str | Path,
        train_yaml: str | Path | None = None,
        *,
        n_src_vocab: int | None = None,
    ) -> FastSpeech2Config:
        preprocess_cfg = PreprocessConfig.from_yaml(preprocess_yaml)
        model_cfg = ModelConfig.from_yaml(model_yaml, n_src_vocab=n_src_vocab)
        train_cfg = (
            TrainConfig.from_yaml(train_yaml)
            if train_yaml is not None
            else None
        )
        return cls(preprocess=preprocess_cfg, model=model_cfg, train=train_cfg)


# ---------------------------------------------------------------------------
# Convenience Loader Functions
# ---------------------------------------------------------------------------


def load_preprocess_config(path: str | Path) -> PreprocessConfig:
    """Load and validate a preprocessing YAML file (`preprocess.yaml`)."""
    return PreprocessConfig.from_yaml(path)


def load_model_config(
    path: str | Path, *, n_src_vocab: int | None = None
) -> ModelConfig:
    """Load and validate a model YAML file (`model.yaml`)."""
    return ModelConfig.from_yaml(path, n_src_vocab=n_src_vocab)


def load_train_config(path: str | Path) -> TrainConfig:
    """Load and validate a training YAML file (`train.yaml`)."""
    return TrainConfig.from_yaml(path)


def load_fastspeech2_config(
    preprocess_yaml: str | Path,
    model_yaml: str | Path,
    train_yaml: str | Path | None = None,
    *,
    n_src_vocab: int | None = None,
) -> FastSpeech2Config:
    """Load and cross-validate preprocess, model, and optional train YAML files."""
    return FastSpeech2Config.from_yaml_paths(
        preprocess_yaml=preprocess_yaml,
        model_yaml=model_yaml,
        train_yaml=train_yaml,
        n_src_vocab=n_src_vocab,
    )


__all__ = [
    "FastSpeech2Config",
    "load_preprocess_config",
    "load_model_config",
    "load_train_config",
    "load_fastspeech2_config",
]

