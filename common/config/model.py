"""Model configuration structures (`model.yaml`) for FastSpeech 2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.config.base import (
    VALID_QUANTIZATION_MODES,
    ConfigNode,
    ConfigValidationError,
    _load_yaml_dict,
    _require_field,
    _require_mapping,
    _validate_bool,
    _validate_choice,
    _validate_dropout,
    _validate_non_empty_str,
    _validate_odd_positive_int,
    _validate_positive_int,
)


@dataclass
class TransformerConfig(ConfigNode):
    """Hyperparameters for the FastSpeech 2 Transformer Encoder and Decoder."""

    encoder_layer: int = 4
    encoder_head: int = 2
    encoder_hidden: int = 256
    decoder_layer: int = 6
    decoder_head: int = 2
    decoder_hidden: int = 256
    conv_filter_size: int = 1024
    conv_kernel_size: list[int] = field(default_factory=lambda: [9, 1])
    encoder_dropout: float = 0.2
    decoder_dropout: float = 0.2

    def __post_init__(self) -> None:
        self.encoder_layer = _validate_positive_int(
            self.encoder_layer, "transformer.encoder_layer"
        )
        self.encoder_head = _validate_positive_int(
            self.encoder_head, "transformer.encoder_head"
        )
        self.encoder_hidden = _validate_positive_int(
            self.encoder_hidden, "transformer.encoder_hidden"
        )
        self.decoder_layer = _validate_positive_int(
            self.decoder_layer, "transformer.decoder_layer"
        )
        self.decoder_head = _validate_positive_int(
            self.decoder_head, "transformer.decoder_head"
        )
        self.decoder_hidden = _validate_positive_int(
            self.decoder_hidden, "transformer.decoder_hidden"
        )
        self.conv_filter_size = _validate_positive_int(
            self.conv_filter_size, "transformer.conv_filter_size"
        )

        if self.encoder_hidden % self.encoder_head != 0:
            raise ConfigValidationError(
                f"'transformer.encoder_hidden' ({self.encoder_hidden}) must be divisible by "
                f"'transformer.encoder_head' ({self.encoder_head})."
            )
        if self.decoder_hidden % self.decoder_head != 0:
            raise ConfigValidationError(
                f"'transformer.decoder_hidden' ({self.decoder_hidden}) must be divisible by "
                f"'transformer.decoder_head' ({self.decoder_head})."
            )
        if self.encoder_hidden != self.decoder_hidden:
            raise ConfigValidationError(
                f"'transformer.encoder_hidden' ({self.encoder_hidden}) must equal "
                f"'transformer.decoder_hidden' ({self.decoder_hidden}) because VarianceAdaptor "
                f"and LengthRegulator pass hidden states directly from Encoder to Decoder."
            )

        if (
            not isinstance(self.conv_kernel_size, (list, tuple))
            or len(self.conv_kernel_size) != 2
        ):
            raise ConfigValidationError(
                f"'transformer.conv_kernel_size' must be a 2-element list/tuple of odd integers, "
                f"got {self.conv_kernel_size!r}."
            )
        k0 = _validate_odd_positive_int(
            self.conv_kernel_size[0], "transformer.conv_kernel_size[0]"
        )
        k1 = _validate_odd_positive_int(
            self.conv_kernel_size[1], "transformer.conv_kernel_size[1]"
        )
        self.conv_kernel_size = [k0, k1]

        self.encoder_dropout = _validate_dropout(
            self.encoder_dropout, "transformer.encoder_dropout"
        )
        self.decoder_dropout = _validate_dropout(
            self.decoder_dropout, "transformer.decoder_dropout"
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TransformerConfig:
        mapping = _require_mapping(data, "transformer")
        return cls(
            encoder_layer=mapping.get("encoder_layer", 4),
            encoder_head=mapping.get("encoder_head", 2),
            encoder_hidden=mapping.get("encoder_hidden", 256),
            decoder_layer=mapping.get("decoder_layer", 6),
            decoder_head=mapping.get("decoder_head", 2),
            decoder_hidden=mapping.get("decoder_hidden", 256),
            conv_filter_size=mapping.get("conv_filter_size", 1024),
            conv_kernel_size=list(mapping.get("conv_kernel_size", [9, 1])),
            encoder_dropout=mapping.get("encoder_dropout", 0.2),
            decoder_dropout=mapping.get("decoder_dropout", 0.2),
        )


@dataclass
class VariancePredictorConfig(ConfigNode):
    """Hyperparameters for Duration, Pitch, and Energy `VariancePredictor` modules."""

    filter_size: int = 256
    kernel_size: int = 3
    dropout: float = 0.5

    def __post_init__(self) -> None:
        self.filter_size = _validate_positive_int(
            self.filter_size, "variance_predictor.filter_size"
        )
        self.kernel_size = _validate_odd_positive_int(
            self.kernel_size, "variance_predictor.kernel_size"
        )
        self.dropout = _validate_dropout(
            self.dropout, "variance_predictor.dropout"
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VariancePredictorConfig:
        mapping = _require_mapping(data, "variance_predictor")
        return cls(
            filter_size=mapping.get("filter_size", 256),
            kernel_size=mapping.get("kernel_size", 3),
            dropout=mapping.get("dropout", 0.5),
        )


@dataclass
class VarianceEmbeddingConfig(ConfigNode):
    """Quantization settings for Pitch and Energy embeddings."""

    pitch_quantization: str = "linear"
    energy_quantization: str = "linear"
    n_bins: int = 256

    def __post_init__(self) -> None:
        self.pitch_quantization = _validate_choice(
            self.pitch_quantization,
            "variance_embedding.pitch_quantization",
            VALID_QUANTIZATION_MODES,
        )
        self.energy_quantization = _validate_choice(
            self.energy_quantization,
            "variance_embedding.energy_quantization",
            VALID_QUANTIZATION_MODES,
        )
        self.n_bins = _validate_positive_int(
            self.n_bins, "variance_embedding.n_bins", min_value=2
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VarianceEmbeddingConfig:
        mapping = _require_mapping(data, "variance_embedding")
        return cls(
            pitch_quantization=mapping.get("pitch_quantization", "linear"),
            energy_quantization=mapping.get("energy_quantization", "linear"),
            n_bins=mapping.get("n_bins", 256),
        )


@dataclass
class PostNetConfig(ConfigNode):
    """Hyperparameters for the 5-layer convolutional PostNet."""

    postnet_embedding_dim: int = 512
    postnet_kernel_size: int = 5
    postnet_n_convolutions: int = 5
    postnet_dropout: float = 0.5

    def __post_init__(self) -> None:
        self.postnet_embedding_dim = _validate_positive_int(
            self.postnet_embedding_dim, "postnet.postnet_embedding_dim"
        )
        self.postnet_kernel_size = _validate_odd_positive_int(
            self.postnet_kernel_size, "postnet.postnet_kernel_size"
        )
        self.postnet_n_convolutions = _validate_positive_int(
            self.postnet_n_convolutions,
            "postnet.postnet_n_convolutions",
            min_value=2,
        )
        self.postnet_dropout = _validate_dropout(
            self.postnet_dropout, "postnet.postnet_dropout"
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PostNetConfig:
        mapping = _require_mapping(data, "postnet")
        return cls(
            postnet_embedding_dim=mapping.get("postnet_embedding_dim", 512),
            postnet_kernel_size=mapping.get("postnet_kernel_size", 5),
            postnet_n_convolutions=mapping.get("postnet_n_convolutions", 5),
            postnet_dropout=mapping.get("postnet_dropout", 0.5),
        )


@dataclass
class VocoderConfig(ConfigNode):
    """Vocoder selection metadata (`HiFi-GAN` or `MelGAN`)."""

    model: str = "HiFi-GAN"
    speaker: str = "universal"

    def __post_init__(self) -> None:
        self.model = _validate_non_empty_str(self.model, "vocoder.model")
        self.speaker = _validate_non_empty_str(self.speaker, "vocoder.speaker")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VocoderConfig:
        mapping = _require_mapping(data, "vocoder")
        return cls(
            model=mapping.get("model", "HiFi-GAN"),
            speaker=mapping.get("speaker", "universal"),
        )


@dataclass
class ModelConfig(ConfigNode):
    """Top-level FastSpeech 2 model configuration (`model.yaml`).

    Note: `n_src_vocab` is explicitly required and never assumed to be identical
    between English (Svarah) and Telugu (IndicTTS) (DEC-06).
    """

    n_src_vocab: int
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    variance_predictor: VariancePredictorConfig = field(
        default_factory=VariancePredictorConfig
    )
    variance_embedding: VarianceEmbeddingConfig = field(
        default_factory=VarianceEmbeddingConfig
    )
    postnet: PostNetConfig = field(default_factory=PostNetConfig)
    multi_speaker: bool = False
    n_speaker: int | None = None
    max_seq_len: int = 1000
    vocoder: VocoderConfig = field(default_factory=VocoderConfig)
    stats: dict[str, list[float]] | None = None

    def __post_init__(self) -> None:
        # Index 0 is reserved for PAD, so vocabulary must contain at least 2 symbols (PAD + >=1 token)
        self.n_src_vocab = _validate_positive_int(
            self.n_src_vocab, "n_src_vocab", min_value=2
        )
        if not isinstance(self.transformer, TransformerConfig):
            self.transformer = TransformerConfig.from_dict(self.transformer)
        if not isinstance(self.variance_predictor, VariancePredictorConfig):
            self.variance_predictor = VariancePredictorConfig.from_dict(
                self.variance_predictor
            )
        if not isinstance(self.variance_embedding, VarianceEmbeddingConfig):
            self.variance_embedding = VarianceEmbeddingConfig.from_dict(
                self.variance_embedding
            )
        if not isinstance(self.postnet, PostNetConfig):
            self.postnet = PostNetConfig.from_dict(self.postnet)
        if not isinstance(self.vocoder, VocoderConfig):
            self.vocoder = VocoderConfig.from_dict(self.vocoder)

        self.multi_speaker = _validate_bool(self.multi_speaker, "multi_speaker")
        if self.n_speaker is not None:
            self.n_speaker = _validate_positive_int(
                self.n_speaker, "n_speaker", min_value=1
            )
        self.max_seq_len = _validate_positive_int(
            self.max_seq_len, "max_seq_len", min_value=1
        )

        if self.stats is not None:
            self._validate_stats(self.stats)

    def _validate_stats(self, stats: Mapping[str, Any]) -> None:
        mapping = _require_mapping(stats, "stats")
        for key, quant_mode in (
            ("pitch", self.variance_embedding.pitch_quantization),
            ("energy", self.variance_embedding.energy_quantization),
        ):
            if key not in mapping:
                raise ConfigValidationError(
                    f"Missing required key '{key}' in 'stats' dictionary."
                )
            vals = mapping[key]
            if not isinstance(vals, (list, tuple)) or len(vals) < 2:
                raise ConfigValidationError(
                    f"'stats.{key}' must be a sequence of at least [min, max], got {vals!r}."
                )
            min_val = float(vals[0])
            max_val = float(vals[1])
            if min_val >= max_val:
                raise ConfigValidationError(
                    f"'stats.{key}' min ({min_val}) must be strictly less than max ({max_val})."
                )
            if quant_mode == "log" and min_val <= 0.0:
                raise ConfigValidationError(
                    f"Cannot use 'log' quantization for '{key}' when stats.{key} min ({min_val}) <= 0.0 "
                    f"(DEC-04 / B-03)."
                )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        n_src_vocab: int | None = None,
    ) -> ModelConfig:
        mapping = _require_mapping(data, "model_config")
        transformer_raw = _require_field(mapping, "transformer", "model_config")
        vp_raw = _require_field(mapping, "variance_predictor", "model_config")
        ve_raw = _require_field(mapping, "variance_embedding", "model_config")

        resolved_vocab = (
            n_src_vocab
            if n_src_vocab is not None
            else mapping.get(
                "n_src_vocab",
                transformer_raw.get("n_src_vocab")
                if isinstance(transformer_raw, Mapping)
                else None,
            )
        )
        if resolved_vocab is None:
            raise ConfigValidationError(
                "Missing required configuration field: 'model_config.n_src_vocab' "
                "(must be specified in model.yaml or passed explicitly per language)."
            )

        return cls(
            n_src_vocab=resolved_vocab,
            transformer=TransformerConfig.from_dict(transformer_raw),
            variance_predictor=VariancePredictorConfig.from_dict(vp_raw),
            variance_embedding=VarianceEmbeddingConfig.from_dict(ve_raw),
            postnet=PostNetConfig.from_dict(mapping.get("postnet", {})),
            multi_speaker=mapping.get("multi_speaker", False),
            n_speaker=mapping.get("n_speaker"),
            max_seq_len=mapping.get("max_seq_len", 1000),
            vocoder=VocoderConfig.from_dict(mapping.get("vocoder", {})),
            stats=mapping.get("stats"),
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        n_src_vocab: int | None = None,
    ) -> ModelConfig:
        return cls.from_dict(_load_yaml_dict(path), n_src_vocab=n_src_vocab)


__all__ = [
    "TransformerConfig",
    "VariancePredictorConfig",
    "VarianceEmbeddingConfig",
    "PostNetConfig",
    "VocoderConfig",
    "ModelConfig",
]

