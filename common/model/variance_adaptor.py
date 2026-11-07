"""Variance Adaptor: duration, pitch and energy prediction + length regulation."""

from __future__ import annotations

import json
import math
import os
import warnings
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from common.config import ModelConfig, PreprocessConfig
from common.model.length_regulator import LengthRegulator
from common.model.variance_predictor import VariancePredictor
from common.paths import resolve_project_path
from common.utils.mask import get_mask_from_lengths


class VarianceAdaptor(nn.Module):
    """Variance Adaptor: duration, pitch and energy prediction + length regulation.

    Data flow (``H`` = encoder hidden size, ``L_src`` tokens, ``L_mel`` frames)::

        x [B, L_src, H]
          -> duration_predictor -> log_duration_prediction [B, L_src]   (log(d + 1))
          -> (phoneme_level) pitch / energy predictor, bucketize, embed, x += emb
          -> durations (ground truth, or converted from log-duration) [B, L_src] int64
          -> LengthRegulator -> x [B, L_mel, H], mel_len [B]
          -> (frame_level) pitch / energy predictor, bucketize, embed, x += emb

    Quantization follows the original repository: ``n_bins - 1`` boundaries
    (linear, or exp of linspace in log-space) between the ``stats`` min and max,
    ``torch.bucketize`` -> ``nn.Embedding(n_bins, H)``.

    Approved corrections: DEC-01 (no global device), DEC-02 (``e_control`` for energy),
    DEC-03 (``d_control`` applied before rounding, int64 durations), DEC-04 (log-bin
    validation), DEC-12 (stats from argument / ``model_config.stats`` / ``stats.json``).
    """

    def __init__(
        self,
        preprocess_config: PreprocessConfig | Mapping[str, Any],
        model_config: ModelConfig | Mapping[str, Any],
        stats: Mapping[str, Any] | None = None,
        n_src_vocab: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(preprocess_config, Mapping):
            preprocess_config = PreprocessConfig.from_dict(preprocess_config)
        if isinstance(model_config, Mapping):
            model_config = ModelConfig.from_dict(
                model_config, n_src_vocab=n_src_vocab
            )
        if not isinstance(preprocess_config, PreprocessConfig) or not isinstance(
            model_config, ModelConfig
        ):
            raise TypeError(
                "VarianceAdaptor requires PreprocessConfig/ModelConfig (or mappings)."
            )

        self.pitch_feature_level = preprocess_config.preprocessing.pitch.feature
        self.energy_feature_level = preprocess_config.preprocessing.energy.feature
        for name, level in (
            ("pitch", self.pitch_feature_level),
            ("energy", self.energy_feature_level),
        ):
            if level not in ("phoneme_level", "frame_level"):
                raise ValueError(f"Invalid {name} feature level: {level!r}.")

        hidden = model_config.transformer.encoder_hidden
        self.hidden_size = hidden
        ve = model_config.variance_embedding
        self.n_bins = ve.n_bins
        self.pitch_quantization = ve.pitch_quantization
        self.energy_quantization = ve.energy_quantization

        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        # Resolve stats: explicit argument > model_config.stats > stats.json on disk.
        if stats is None:
            stats = model_config.stats
        if stats is None:
            raw_prep_path = preprocess_config.path.preprocessed_path
            resolved_prep = resolve_project_path(raw_prep_path)
            resolved_stats = resolved_prep / "stats.json"
            if resolved_stats.is_file():
                stats_path = str(resolved_stats)
            else:
                stats_path = os.path.join(raw_prep_path, "stats.json")
            if not os.path.isfile(stats_path):
                raise FileNotFoundError(
                    f"No pitch/energy stats provided and '{stats_path}' does not exist."
                )
            with open(stats_path, encoding="utf-8") as f:
                stats = json.load(f)
        pitch_min, pitch_max = self._read_min_max(stats, "pitch")
        energy_min, energy_max = self._read_min_max(stats, "energy")

        # Non-trainable `nn.Parameter`s, exactly as in the original implementation: they
        # appear in `parameters()` and `state_dict()` (same names/shapes/dtypes) but
        # `requires_grad=False` guarantees they never receive gradients.
        self.pitch_bins = nn.Parameter(
            self._make_bins(pitch_min, pitch_max, self.pitch_quantization, "pitch"),
            requires_grad=False,
        )
        self.energy_bins = nn.Parameter(
            self._make_bins(
                energy_min, energy_max, self.energy_quantization, "energy"
            ),
            requires_grad=False,
        )
        self.pitch_embedding = nn.Embedding(self.n_bins, hidden)
        self.energy_embedding = nn.Embedding(self.n_bins, hidden)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _read_min_max(stats: Mapping[str, Any], key: str) -> tuple[float, float]:
        if not isinstance(stats, Mapping) or key not in stats:
            raise ValueError(f"stats must contain a '{key}' entry [min, max, ...].")
        values = stats[key]
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            raise ValueError(f"stats['{key}'] must have at least [min, max].")
        lo, hi = float(values[0]), float(values[1])
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
            raise ValueError(f"Invalid stats['{key}'] range: min={lo}, max={hi}.")
        return lo, hi

    def _make_bins(
        self, lo: float, hi: float, mode: str, name: str
    ) -> torch.Tensor:
        if mode == "log":
            if lo <= 0.0:
                raise ValueError(
                    f"'log' quantization requires {name}_min > 0, got {lo}."
                )
            return torch.exp(
                torch.linspace(math.log(lo), math.log(hi), self.n_bins - 1)
            )
        return torch.linspace(lo, hi, self.n_bins - 1)

    @staticmethod
    def log_duration_to_frames(
        log_duration: torch.Tensor, d_control: float = 1.0
    ) -> torch.Tensor:
        """Convert predicted ``log(d + 1)`` to non-negative int64 frame counts.

        Order of operations (DEC-03)::

            1. d      = exp(log_duration) - 1      # undo the log(d + 1) transform
            2. d      = d * d_control              # speaking-rate control
            3. frames = round(d)                   # round-half-to-even (torch.round)
            4. frames = clamp(frames, min=0)       # no negative durations
            5. frames = frames.long()              # int64 frame counts

        Differences from the original FastSpeech 2 (``round(exp(x) - 1) * d_control``,
        clamp, float result, later truncated by ``int()`` inside ``LengthRegulator``):

        * The original rounds *before* multiplying by ``d_control`` and leaves fractional
          durations that the regulator floors; here ``d_control`` is applied *before* a
          single rounding step and the result is integral.
        * Results are identical when ``d_control == 1.0``. For ``d_control != 1.0`` they
          can differ (e.g. ``d=1, d_control=0.8``: original -> 0 frames, here -> 1 frame;
          ``d=1.5, d_control=1.5``: original -> 3 frames then 3, here -> 2).

        Args:
            log_duration: predicted ``log(d + 1)`` values (any shape).
            d_control: non-negative duration scale (``> 1`` slows speech, ``< 1`` speeds it).

        Returns:
            int64 tensor with the same shape as ``log_duration``.
        """
        frames = torch.clamp(
            torch.round((torch.exp(log_duration) - 1.0) * d_control), min=0
        )
        return frames.long()

    def _embed(
        self,
        predictor: VariancePredictor,
        embedding: nn.Embedding,
        bins: torch.Tensor,
        x: torch.Tensor,
        target: torch.Tensor | None,
        mask: torch.Tensor | None,
        control: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = predictor(x, mask)
        if target is not None:
            if target.shape != prediction.shape:
                raise ValueError(
                    f"Target shape {tuple(target.shape)} does not match "
                    f"prediction shape {tuple(prediction.shape)}."
                )
            values = target.to(device=x.device, dtype=bins.dtype)
        else:
            prediction = prediction * control
            values = prediction.detach().to(dtype=bins.dtype)
        emb = embedding(torch.bucketize(values, bins))
        if mask is not None:
            emb = emb.masked_fill(mask.unsqueeze(-1), 0.0)
        return prediction, emb

    def get_pitch_embedding(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None,
        mask: torch.Tensor | None,
        control: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._embed(
            self.pitch_predictor,
            self.pitch_embedding,
            self.pitch_bins,
            x,
            target,
            mask,
            control,
        )

    def get_energy_embedding(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None,
        mask: torch.Tensor | None,
        control: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._embed(
            self.energy_predictor,
            self.energy_embedding,
            self.energy_bins,
            x,
            target,
            mask,
            control,
        )

    # ----------------------------------------------------------- forward stages
    def _validate_inputs(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor,
        p_control: float,
        e_control: float,
        d_control: float,
    ) -> torch.Tensor:
        """Validate input tensor shapes, types, device, and control values."""
        if not isinstance(x, torch.Tensor) or x.dim() != 3:
            raise ValueError("x must be a 3D tensor [B, L_src, H].")
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"x hidden size {x.shape[-1]} != encoder_hidden {self.hidden_size}."
            )
        if src_mask is None:
            raise ValueError("src_mask is required.")
        if (
            not isinstance(src_mask, torch.Tensor)
            or src_mask.shape != x.shape[:2]
        ):
            raise ValueError(
                f"src_mask shape {getattr(src_mask, 'shape', None)} must be "
                f"{tuple(x.shape[:2])}."
            )
        src_mask = src_mask.to(device=x.device, dtype=torch.bool)
        fully_padded = src_mask.all(dim=1)
        if fully_padded.any().item():
            idx = torch.nonzero(fully_padded).flatten().tolist()
            raise ValueError(
                f"src_mask rows {idx} are entirely padded (no valid tokens); every "
                "utterance must contain at least one non-padded source token."
            )
        for name, val in (
            ("p_control", p_control),
            ("e_control", e_control),
            ("d_control", d_control),
        ):
            if not math.isfinite(float(val)) or float(val) < 0:
                raise ValueError(f"{name} must be a finite non-negative number.")
        return src_mask

    def _process_pitch(
        self,
        x: torch.Tensor,
        pitch_target: torch.Tensor | None,
        mask: torch.Tensor,
        p_control: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict or accept pitch targets, quantize, embed, and condition representations."""
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            x, pitch_target, mask, p_control
        )
        x = x + pitch_embedding
        return x, pitch_prediction

    def _process_energy(
        self,
        x: torch.Tensor,
        energy_target: torch.Tensor | None,
        mask: torch.Tensor,
        e_control: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict or accept energy targets, quantize, embed, and condition representations."""
        energy_prediction, energy_embedding = self.get_energy_embedding(
            x, energy_target, mask, e_control
        )
        x = x + energy_embedding
        return x, energy_prediction

    def _process_durations(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor,
        log_duration_prediction: torch.Tensor,
        duration_target: torch.Tensor | None,
        d_control: float,
        max_len: int | None,
        mel_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Regulate length using either ground-truth or predicted durations.

        Args:
            x: FloatTensor `[B, L_src, H]`.
            src_mask: BoolTensor `[B, L_src]`.
            log_duration_prediction: FloatTensor `[B, L_src]`.
            duration_target: Optional LongTensor `[B, L_src]`.
            d_control: Speaking-rate scale factor.
            max_len: Optional target maximum frame length.
            mel_mask: Optional target mel mask for training.

        Returns:
            Tuple of `(x_expanded, duration_rounded, mel_len, mel_mask)`.
        """
        if duration_target is not None:
            if (
                not isinstance(duration_target, torch.Tensor)
                or duration_target.shape != src_mask.shape
            ):
                raise ValueError(
                    "duration_target must be a tensor with shape [B, L_src]."
                )
            duration_rounded = duration_target.to(device=x.device)
            if duration_rounded.is_floating_point():
                raise ValueError(
                    "duration_target must be an integer tensor, got "
                    f"{duration_rounded.dtype}."
                )
            duration_rounded = duration_rounded.long()
            if (duration_rounded.masked_fill(src_mask, 0) < 0).any().item():
                raise ValueError("duration_target contains negative values.")
            # Padded tokens never produce frames.
            duration_rounded = duration_rounded.masked_fill(src_mask, 0)
            x_expanded, mel_len = self.length_regulator(x, duration_rounded, max_len)
            if (mel_len == 0).any().item():
                idx = torch.nonzero(mel_len == 0).flatten().tolist()
                raise ValueError(
                    f"duration_target sums to 0 frames for batch indices {idx}; "
                    "ground-truth utterances must have at least one Mel frame. "
                    "(Ground-truth durations are never modified.)"
                )
            if mel_mask is None:
                mel_mask = get_mask_from_lengths(mel_len, max_len=x_expanded.shape[1])
            else:
                mel_mask = mel_mask.to(device=x.device, dtype=torch.bool)
                if mel_mask.shape != x_expanded.shape[:2]:
                    raise ValueError(
                        f"mel_mask shape {tuple(mel_mask.shape)} does not match "
                        f"expanded shape {tuple(x_expanded.shape[:2])}."
                    )
        else:
            duration_rounded = self.log_duration_to_frames(
                log_duration_prediction.detach(), d_control
            ).masked_fill(src_mask, 0)
            empty = duration_rounded.sum(dim=1) == 0
            if empty.any().item():
                empty_idx = torch.nonzero(empty).flatten().tolist()
                shown = empty_idx[:10]
                more = (
                    f" (+{len(empty_idx) - 10} more)" if len(empty_idx) > 10 else ""
                )
                # One warning per forward call (not per utterance), indices capped at 10.
                warnings.warn(
                    "VarianceAdaptor inference fallback: predicted durations were all 0 "
                    f"for batch indices {shown}{more}; assigned 1 frame to the first "
                    "valid token of each.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                first_valid = (~src_mask).long().argmax(dim=1)
                fix = torch.zeros_like(duration_rounded)
                fix[torch.arange(x.shape[0], device=x.device), first_valid] = 1
                duration_rounded = torch.where(
                    empty.unsqueeze(1), fix, duration_rounded
                )
            x_expanded, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len, max_len=x_expanded.shape[1])

        return x_expanded, duration_rounded, mel_len, mel_mask

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor,
        mel_mask: torch.Tensor | None = None,
        max_len: int | None = None,
        pitch_target: torch.Tensor | None = None,
        energy_target: torch.Tensor | None = None,
        duration_target: torch.Tensor | None = None,
        p_control: float = 1.0,
        e_control: float = 1.0,
        d_control: float = 1.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run the Variance Adaptor.

        Args:
            x: ``[B, L_src, H]`` encoder output.
            src_mask: ``[B, L_src]`` bool, ``True`` = PAD (required).
            mel_mask: ``[B, L_mel]`` bool, training only (derived from durations if None).
            max_len: optional padded frame length (training: max Mel length in batch).
            pitch_target / energy_target: ``[B, L_src]`` (phoneme_level) or
                ``[B, L_mel]`` (frame_level), or None to use predictions.
            duration_target: ``[B, L_src]`` integer durations, or None to predict.
            p_control / e_control / d_control: inference-only scale factors. Pitch/energy
                controls multiply the prediction before bucketizing (ignored when the
                corresponding target is given); ``d_control`` scales durations before
                rounding (ignored when ``duration_target`` is given).

        Returns:
            ``(x [B, L_mel, H], pitch_prediction, energy_prediction,
            log_duration_prediction [B, L_src], duration_rounded [B, L_src] int64,
            mel_len [B] int64, mel_mask [B, L_mel] bool)``.

        In inference, an utterance whose predicted durations all round to 0 gets its first
        valid token set to 1 frame (DEF-02 fallback) so downstream convolutions never see
        a zero-length sequence; the change is visible in ``duration_rounded`` and announced by a single
        ``RuntimeWarning`` per call listing the affected batch indices. Ground-truth
        durations are never modified; ground-truth utterances with 0 frames raise
        ``ValueError``. Source-mask rows that are entirely padded raise ``ValueError``.
        """
        src_mask = self._validate_inputs(
            x, src_mask, p_control, e_control, d_control
        )

        log_duration_prediction = self.duration_predictor(x, src_mask)

        # Phoneme-level conditioning
        if self.pitch_feature_level == "phoneme_level":
            x, pitch_prediction = self._process_pitch(
                x, pitch_target, src_mask, p_control
            )
        if self.energy_feature_level == "phoneme_level":
            x, energy_prediction = self._process_energy(
                x, energy_target, src_mask, e_control
            )

        # Length regulation and duration processing
        x, duration_rounded, mel_len, mel_mask = self._process_durations(
            x=x,
            src_mask=src_mask,
            log_duration_prediction=log_duration_prediction,
            duration_target=duration_target,
            d_control=d_control,
            max_len=max_len,
            mel_mask=mel_mask,
        )

        # Frame-level conditioning
        if self.pitch_feature_level == "frame_level":
            x, pitch_prediction = self._process_pitch(
                x, pitch_target, mel_mask, p_control
            )
        if self.energy_feature_level == "frame_level":
            x, energy_prediction = self._process_energy(
                x, energy_target, mel_mask, e_control
            )

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


__all__ = ["VarianceAdaptor"]

