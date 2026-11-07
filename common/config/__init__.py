"""Modular configuration package for FastSpeech 2 (`common.config`).

Supports independent English (Svarah) and Telugu (IndicTTS) pipelines while
enforcing architectural invariants defined in:
- research/COMMON_ARCHITECTURE_SPECIFICATION.md
- research/IMPLEMENTATION_DECISIONS.md

Re-exports all configuration classes, constants, exceptions, and loader functions
for backward-compatible `from common.config import ...` access.
"""

from __future__ import annotations

from common.config.base import (
    VALID_FEATURE_LEVELS,
    VALID_QUANTIZATION_MODES,
    ConfigNode,
    ConfigValidationError,
    _is_bool,
    _load_yaml_dict,
    _require_field,
    _require_mapping,
    _validate_bool,
    _validate_choice,
    _validate_dropout,
    _validate_non_empty_str,
    _validate_non_negative_int,
    _validate_odd_positive_int,
    _validate_positive_float,
    _validate_positive_int,
)
from common.config.loader import (
    FastSpeech2Config,
    load_fastspeech2_config,
    load_model_config,
    load_preprocess_config,
    load_train_config,
)
from common.config.model import (
    ModelConfig,
    PostNetConfig,
    TransformerConfig,
    VarianceEmbeddingConfig,
    VariancePredictorConfig,
    VocoderConfig,
)
from common.config.preprocess import (
    AudioConfig,
    EnergyConfig,
    MelConfig,
    PitchConfig,
    PreprocessConfig,
    PreprocessingSettings,
    PreprocessPathConfig,
    STFTConfig,
    TextConfig,
)
from common.config.training import (
    OptimizerConfig,
    StepConfig,
    TrainConfig,
    TrainPathConfig,
)

__all__ = [
    "AudioConfig",
    "ConfigNode",
    "ConfigValidationError",
    "EnergyConfig",
    "FastSpeech2Config",
    "MelConfig",
    "ModelConfig",
    "OptimizerConfig",
    "PitchConfig",
    "PostNetConfig",
    "PreprocessConfig",
    "PreprocessPathConfig",
    "PreprocessingSettings",
    "STFTConfig",
    "StepConfig",
    "TextConfig",
    "TrainConfig",
    "TrainPathConfig",
    "TransformerConfig",
    "VALID_FEATURE_LEVELS",
    "VALID_QUANTIZATION_MODES",
    "VarianceEmbeddingConfig",
    "VariancePredictorConfig",
    "VocoderConfig",
    "load_fastspeech2_config",
    "load_model_config",
    "load_preprocess_config",
    "load_train_config",
]

