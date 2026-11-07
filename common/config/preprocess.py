"""Preprocessing configuration structures (`preprocess.yaml`) for FastSpeech 2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.config.base import (
    VALID_FEATURE_LEVELS,
    ConfigNode,
    ConfigValidationError,
    _load_yaml_dict,
    _require_field,
    _require_mapping,
    _validate_bool,
    _validate_choice,
    _validate_non_empty_str,
    _validate_non_negative_int,
    _validate_positive_float,
    _validate_positive_int,
)


@dataclass
class PreprocessPathConfig(ConfigNode):
    """Dataset and output paths for preprocessing."""

    corpus_path: str
    raw_path: str
    preprocessed_path: str
    lexicon_path: str | None = None

    def __post_init__(self) -> None:
        self.corpus_path = _validate_non_empty_str(
            self.corpus_path, "path.corpus_path"
        )
        self.raw_path = _validate_non_empty_str(self.raw_path, "path.raw_path")
        self.preprocessed_path = _validate_non_empty_str(
            self.preprocessed_path, "path.preprocessed_path"
        )
        if self.lexicon_path is not None:
            self.lexicon_path = _validate_non_empty_str(
                self.lexicon_path, "path.lexicon_path"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PreprocessPathConfig:
        mapping = _require_mapping(data, "path")
        return cls(
            corpus_path=_require_field(mapping, "corpus_path", "path"),
            raw_path=_require_field(mapping, "raw_path", "path"),
            preprocessed_path=_require_field(
                mapping, "preprocessed_path", "path"
            ),
            lexicon_path=mapping.get("lexicon_path"),
        )


@dataclass
class TextConfig(ConfigNode):
    """Language and text-cleaner settings for the text frontend."""

    language: str
    text_cleaners: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.language = _validate_non_empty_str(
            self.language, "preprocessing.text.language"
        )
        if not isinstance(self.text_cleaners, (list, tuple)):
            raise ConfigValidationError(
                "'preprocessing.text.text_cleaners' must be a list of strings."
            )
        cleaned: list[str] = []
        for idx, cleaner in enumerate(self.text_cleaners):
            cleaned.append(
                _validate_non_empty_str(
                    cleaner, f"preprocessing.text.text_cleaners[{idx}]"
                )
            )
        self.text_cleaners = cleaned

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TextConfig:
        mapping = _require_mapping(data, "preprocessing.text")
        return cls(
            language=_require_field(mapping, "language", "preprocessing.text"),
            text_cleaners=list(mapping.get("text_cleaners", [])),
        )


@dataclass
class AudioConfig(ConfigNode):
    """Waveform sampling rate and normalization settings."""

    sampling_rate: int = 22050
    max_wav_value: float = 32768.0

    def __post_init__(self) -> None:
        self.sampling_rate = _validate_positive_int(
            self.sampling_rate, "preprocessing.audio.sampling_rate"
        )
        self.max_wav_value = _validate_positive_float(
            self.max_wav_value, "preprocessing.audio.max_wav_value"
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AudioConfig:
        mapping = _require_mapping(data, "preprocessing.audio")
        return cls(
            sampling_rate=mapping.get("sampling_rate", 22050),
            max_wav_value=mapping.get("max_wav_value", 32768.0),
        )


@dataclass
class STFTConfig(ConfigNode):
    """Short-Time Fourier Transform (STFT) window and hop settings."""

    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024

    def __post_init__(self) -> None:
        self.filter_length = _validate_positive_int(
            self.filter_length, "preprocessing.stft.filter_length"
        )
        self.hop_length = _validate_positive_int(
            self.hop_length, "preprocessing.stft.hop_length"
        )
        self.win_length = _validate_positive_int(
            self.win_length, "preprocessing.stft.win_length"
        )
        if self.win_length > self.filter_length:
            raise ConfigValidationError(
                f"'preprocessing.stft.win_length' ({self.win_length}) cannot exceed "
                f"'preprocessing.stft.filter_length' ({self.filter_length})."
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> STFTConfig:
        mapping = _require_mapping(data, "preprocessing.stft")
        return cls(
            filter_length=mapping.get("filter_length", 1024),
            hop_length=mapping.get("hop_length", 256),
            win_length=mapping.get("win_length", 1024),
        )


@dataclass
class MelConfig(ConfigNode):
    """Mel-filterbank settings."""

    n_mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float | None = 8000.0

    def __post_init__(self) -> None:
        self.n_mel_channels = _validate_positive_int(
            self.n_mel_channels, "preprocessing.mel.n_mel_channels"
        )
        self.mel_fmin = _validate_positive_float(
            self.mel_fmin, "preprocessing.mel.mel_fmin", allow_zero=True
        )
        if self.mel_fmax is not None:
            self.mel_fmax = _validate_positive_float(
                self.mel_fmax, "preprocessing.mel.mel_fmax"
            )
            if self.mel_fmax <= self.mel_fmin:
                raise ConfigValidationError(
                    f"'preprocessing.mel.mel_fmax' ({self.mel_fmax}) must be greater than "
                    f"'preprocessing.mel.mel_fmin' ({self.mel_fmin})."
                )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MelConfig:
        mapping = _require_mapping(data, "preprocessing.mel")
        return cls(
            n_mel_channels=mapping.get("n_mel_channels", 80),
            mel_fmin=mapping.get("mel_fmin", 0.0),
            mel_fmax=mapping.get("mel_fmax", 8000.0)
            if "mel_fmax" not in mapping
            else mapping["mel_fmax"],
        )


@dataclass
class PitchConfig(ConfigNode):
    """Pitch (F0) extraction and normalization settings."""

    feature: str = "phoneme_level"
    normalization: bool = True

    def __post_init__(self) -> None:
        self.feature = _validate_choice(
            self.feature, "preprocessing.pitch.feature", VALID_FEATURE_LEVELS
        )
        self.normalization = _validate_bool(
            self.normalization, "preprocessing.pitch.normalization"
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PitchConfig:
        mapping = _require_mapping(data, "preprocessing.pitch")
        return cls(
            feature=mapping.get("feature", "phoneme_level"),
            normalization=mapping.get("normalization", True),
        )


@dataclass
class EnergyConfig(ConfigNode):
    """Energy extraction and normalization settings."""

    feature: str = "phoneme_level"
    normalization: bool = True

    def __post_init__(self) -> None:
        self.feature = _validate_choice(
            self.feature, "preprocessing.energy.feature", VALID_FEATURE_LEVELS
        )
        self.normalization = _validate_bool(
            self.normalization, "preprocessing.energy.normalization"
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EnergyConfig:
        mapping = _require_mapping(data, "preprocessing.energy")
        return cls(
            feature=mapping.get("feature", "phoneme_level"),
            normalization=mapping.get("normalization", True),
        )


@dataclass
class PreprocessingSettings(ConfigNode):
    """Container for all feature-extraction settings under `preprocessing:`."""

    text: TextConfig
    val_size: int = 512
    audio: AudioConfig = field(default_factory=AudioConfig)
    stft: STFTConfig = field(default_factory=STFTConfig)
    mel: MelConfig = field(default_factory=MelConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.text, TextConfig):
            self.text = TextConfig.from_dict(self.text)
        self.val_size = _validate_non_negative_int(
            self.val_size, "preprocessing.val_size"
        )
        if not isinstance(self.audio, AudioConfig):
            self.audio = AudioConfig.from_dict(self.audio)
        if not isinstance(self.stft, STFTConfig):
            self.stft = STFTConfig.from_dict(self.stft)
        if not isinstance(self.mel, MelConfig):
            self.mel = MelConfig.from_dict(self.mel)
        if not isinstance(self.pitch, PitchConfig):
            self.pitch = PitchConfig.from_dict(self.pitch)
        if not isinstance(self.energy, EnergyConfig):
            self.energy = EnergyConfig.from_dict(self.energy)

        nyquist = self.audio.sampling_rate / 2.0
        if self.mel.mel_fmax is not None and self.mel.mel_fmax > nyquist:
            raise ConfigValidationError(
                f"'preprocessing.mel.mel_fmax' ({self.mel.mel_fmax}) cannot exceed "
                f"the Nyquist frequency ({nyquist} Hz for sampling_rate={self.audio.sampling_rate})."
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PreprocessingSettings:
        mapping = _require_mapping(data, "preprocessing")
        text_raw = _require_field(mapping, "text", "preprocessing")
        return cls(
            text=TextConfig.from_dict(text_raw),
            val_size=mapping.get("val_size", 512),
            audio=AudioConfig.from_dict(mapping.get("audio", {})),
            stft=STFTConfig.from_dict(mapping.get("stft", {})),
            mel=MelConfig.from_dict(mapping.get("mel", {})),
            pitch=PitchConfig.from_dict(mapping.get("pitch", {})),
            energy=EnergyConfig.from_dict(mapping.get("energy", {})),
        )


@dataclass
class PreprocessConfig(ConfigNode):
    """Top-level preprocessing configuration (`preprocess.yaml`)."""

    dataset: str
    path: PreprocessPathConfig
    preprocessing: PreprocessingSettings

    def __post_init__(self) -> None:
        self.dataset = _validate_non_empty_str(self.dataset, "dataset")
        if not isinstance(self.path, PreprocessPathConfig):
            self.path = PreprocessPathConfig.from_dict(self.path)
        if not isinstance(self.preprocessing, PreprocessingSettings):
            self.preprocessing = PreprocessingSettings.from_dict(
                self.preprocessing
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PreprocessConfig:
        mapping = _require_mapping(data, "preprocess_config")
        dataset = _require_field(mapping, "dataset", "preprocess_config")
        path_raw = _require_field(mapping, "path", "preprocess_config")
        prep_raw = _require_field(mapping, "preprocessing", "preprocess_config")
        return cls(
            dataset=dataset,
            path=PreprocessPathConfig.from_dict(path_raw),
            preprocessing=PreprocessingSettings.from_dict(prep_raw),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PreprocessConfig:
        return cls.from_dict(_load_yaml_dict(path))


__all__ = [
    "PreprocessPathConfig",
    "TextConfig",
    "AudioConfig",
    "STFTConfig",
    "MelConfig",
    "PitchConfig",
    "EnergyConfig",
    "PreprocessingSettings",
    "PreprocessConfig",
]

