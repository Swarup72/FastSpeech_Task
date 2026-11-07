"""Base configuration node, validation exception, constants, and shared validation helpers.

Enforces architectural invariants defined in:
- research/COMMON_ARCHITECTURE_SPECIFICATION.md
- research/IMPLEMENTATION_DECISIONS.md
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

VALID_FEATURE_LEVELS: tuple[str, str] = ("phoneme_level", "frame_level")
VALID_QUANTIZATION_MODES: tuple[str, str] = ("linear", "log")


class ConfigValidationError(ValueError):
    """Raised when a FastSpeech 2 configuration is missing required fields or violates constraints."""


# ---------------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------------


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _require_mapping(data: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ConfigValidationError(
            f"Expected a mapping/dict for '{context}', got {type(data).__name__}."
        )
    return data


def _require_field(data: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in data or data[key] is None:
        raise ConfigValidationError(
            f"Missing required configuration field: '{context}.{key}'."
        )
    return data[key]


def _validate_non_empty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(
            f"'{name}' must be a non-empty string, got {value!r}."
        )
    return value


def _validate_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigValidationError(
            f"'{name}' must be a boolean (True/False), got {value!r} ({type(value).__name__})."
        )
    return value


def _validate_positive_int(value: Any, name: str, min_value: int = 1) -> int:
    if _is_bool(value) or not isinstance(value, int) or value < min_value:
        raise ConfigValidationError(
            f"'{name}' must be an integer >= {min_value}, got {value!r}."
        )
    return value


def _validate_non_negative_int(value: Any, name: str) -> int:
    return _validate_positive_int(value, name, min_value=0)


def _validate_odd_positive_int(value: Any, name: str) -> int:
    val = _validate_positive_int(value, name, min_value=1)
    if val % 2 != 1:
        raise ConfigValidationError(
            f"'{name}' must be an odd positive integer to preserve sequence length, got {val}."
        )
    return val


def _validate_positive_float(
    value: Any, name: str, *, allow_zero: bool = False
) -> float:
    if _is_bool(value) or not isinstance(value, (int, float)):
        raise ConfigValidationError(
            f"'{name}' must be a number, got {value!r} ({type(value).__name__})."
        )
    val = float(value)
    if allow_zero:
        if val < 0.0:
            raise ConfigValidationError(
                f"'{name}' must be >= 0.0, got {val}."
            )
    else:
        if val <= 0.0:
            raise ConfigValidationError(
                f"'{name}' must be > 0.0, got {val}."
            )
    return val


def _validate_dropout(value: Any, name: str) -> float:
    if _is_bool(value) or not isinstance(value, (int, float)):
        raise ConfigValidationError(
            f"'{name}' must be a float in [0.0, 1.0), got {value!r}."
        )
    val = float(value)
    if not (0.0 <= val < 1.0):
        raise ConfigValidationError(
            f"'{name}' must be in the range [0.0, 1.0), got {val}."
        )
    return val


def _validate_choice(value: Any, name: str, choices: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ConfigValidationError(
            f"'{name}' must be one of {list(choices)}, got {value!r}."
        )
    return value


def _load_yaml_dict(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigValidationError(
            f"Configuration YAML file does not exist: '{file_path}'."
        )
    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigValidationError(
            f"Failed to parse YAML file '{file_path}': {exc}"
        ) from exc
    if not isinstance(content, Mapping):
        raise ConfigValidationError(
            f"Expected top-level YAML mapping in '{file_path}', got {type(content).__name__}."
        )
    return dict(content)


# ---------------------------------------------------------------------------
# Base Mapping-Compatible Config Node
# ---------------------------------------------------------------------------


class ConfigNode(Mapping[str, Any]):
    """Base class allowing both attribute access (`cfg.field`) and dict access (`cfg['field']`)."""

    def __getitem__(self, key: str) -> Any:
        for f in fields(self):  # type: ignore[arg-type]
            if f.name == key:
                return getattr(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        for f in fields(self):  # type: ignore[arg-type]
            yield f.name

    def __len__(self) -> int:
        return len(fields(self))  # type: ignore[arg-type]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return any(f.name == key for f in fields(self))  # type: ignore[arg-type]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for f in fields(self):  # type: ignore[arg-type]
            val = getattr(self, f.name)
            if isinstance(val, ConfigNode):
                result[f.name] = val.to_dict()
            elif isinstance(val, list):
                result[f.name] = list(val)
            elif isinstance(val, dict):
                result[f.name] = dict(val)
            else:
                result[f.name] = val
        return result


__all__ = [
    "VALID_FEATURE_LEVELS",
    "VALID_QUANTIZATION_MODES",
    "ConfigValidationError",
    "ConfigNode",
]

