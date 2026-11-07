"""FastSpeech 2 multi-task loss module.

Computes the 5-term unweighted multi-task loss for FastSpeech 2:
    1. Initial Mel Reconstruction Loss: MAE (L1) over valid frames.
    2. PostNet Mel Reconstruction Loss: MAE (L1) over valid frames.
    3. Pitch Prediction Loss: MSE over valid tokens or frames.
    4. Energy Prediction Loss: MSE over valid tokens or frames.
    5. Duration Prediction Loss: MSE over valid source tokens in log domain:
       log_d_target = log(d_target + 1.0).

Total loss is the unweighted sum (DEC-14):
    total_loss = mel_loss + postnet_mel_loss + pitch_loss + energy_loss + duration_loss

Includes DEC-10 bug-fix:
    When pitch or energy are configured at 'frame_level', targets and predictions
    are sliced along the temporal dimension to match the Decoder-truncated
    mel mask length (:mel_masks.shape[1]), preventing dimension mismatch crashes.
    Empty masked selections are detected and rejected with informative ValueErrors
    to prevent 0/0 NaN reductions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn


def _extract_feature_level(cfg: Any, feature_name: str) -> str:
    """Extract 'phoneme_level' or 'frame_level' from preprocess configuration.

    Supports PreprocessConfig objects, plain dictionaries, and nested mappings.
    """
    if hasattr(cfg, "preprocessing"):
        prep = cfg.preprocessing
    elif isinstance(cfg, Mapping) and "preprocessing" in cfg:
        prep = cfg["preprocessing"]
    else:
        prep = cfg

    if hasattr(prep, feature_name):
        feature_cfg = getattr(prep, feature_name)
    elif isinstance(prep, Mapping) and feature_name in prep:
        feature_cfg = prep[feature_name]
    else:
        raise ValueError(
            f"Could not find '{feature_name}' configuration in preprocess_config."
        )

    if hasattr(feature_cfg, "feature"):
        level = feature_cfg.feature
    elif isinstance(feature_cfg, Mapping) and "feature" in feature_cfg:
        level = feature_cfg["feature"]
    else:
        level = "phoneme_level"

    if level not in ("phoneme_level", "frame_level"):
        raise ValueError(
            f"Invalid feature level for '{feature_name}': {level!r}. "
            "Expected 'phoneme_level' or 'frame_level'."
        )
    return str(level)


class FastSpeech2Loss(nn.Module):
    """FastSpeech 2 multi-task loss function.

    Computes reconstruction and predictor losses for FastSpeech 2 training
    and validation. Preserves exact parameter naming, loss types, reduction
    methods, and return signature of the original implementation while fixing
    DEC-10 frame-level temporal truncation and empty-mask NaN vulnerabilities.
    """

    def __init__(
        self,
        preprocess_config: Any,
        model_config: Any | None = None,
    ) -> None:
        """Initialize loss function with feature-level settings.

        Args:
            preprocess_config: Configuration containing pitch and energy feature
                levels ('phoneme_level' or 'frame_level').
            model_config: Optional model configuration (retained for signature
                compatibility with the original codebase).
        """
        super().__init__()
        self.preprocess_config = preprocess_config
        self.model_config = model_config

        self.pitch_feature_level = _extract_feature_level(
            preprocess_config, "pitch"
        )
        self.energy_feature_level = _extract_feature_level(
            preprocess_config, "energy"
        )

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def _unpack_inputs(
        self, inputs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack target tensors from batch sequence, mapping, or keyword inputs.

        Returns:
            Tuple of (mel_targets, pitch_targets, energy_targets, duration_targets).
        """
        if isinstance(inputs, Mapping):
            mel_targets = inputs.get("mel_targets", inputs.get("mels"))
            pitch_targets = inputs.get("pitch_targets", inputs.get("pitches"))
            energy_targets = inputs.get("energy_targets", inputs.get("energies"))
            duration_targets = inputs.get(
                "duration_targets", inputs.get("durations")
            )
            if any(
                t is None
                for t in (
                    mel_targets,
                    pitch_targets,
                    energy_targets,
                    duration_targets,
                )
            ):
                raise ValueError(
                    "Dictionary inputs must provide mel_targets (or mels), "
                    "pitch_targets (or pitches), energy_targets (or energies), "
                    "and duration_targets (or durations)."
                )
            return mel_targets, pitch_targets, energy_targets, duration_targets

        if isinstance(inputs, (list, tuple)):
            # Standard 12-element training batch:
            # (ids, raw_texts, speakers, texts, src_lens, max_src_len,
            #  mel_targets, mel_lens, max_mel_len, pitch_targets, energy_targets, duration_targets)
            if len(inputs) >= 12:
                mel_targets = inputs[6]
                pitch_targets = inputs[9]
                energy_targets = inputs[10]
                duration_targets = inputs[11]
                return mel_targets, pitch_targets, energy_targets, duration_targets

            # 6-element targets tuple:
            # (mel_targets, _, _, pitch_targets, energy_targets, duration_targets)
            # matching inputs[6:] from original loss.py
            if len(inputs) == 6:
                mel_targets = inputs[0]
                pitch_targets = inputs[3]
                energy_targets = inputs[4]
                duration_targets = inputs[5]
                return mel_targets, pitch_targets, energy_targets, duration_targets

            # 4-element target tuple: (mels, pitches, energies, durations)
            if len(inputs) == 4:
                return inputs[0], inputs[1], inputs[2], inputs[3]

        raise ValueError(
            "inputs must be a 12-element FastSpeech 2 batch tuple, a 6-element "
            "targets tuple inputs[6:], a 4-element target tuple, or a mapping "
            "containing mel, pitch, energy, and duration targets."
        )

    def _unpack_predictions(
        self, predictions: Any
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Unpack prediction tensors from model output sequence or mapping.

        Returns:
            Tuple of:
            (mel_predictions, postnet_mel_predictions, pitch_predictions,
             energy_predictions, log_duration_predictions, src_masks, mel_masks).
        """
        if isinstance(predictions, Mapping):
            mel_predictions = predictions.get(
                "mel_predictions", predictions.get("output")
            )
            postnet_mel_predictions = predictions.get(
                "postnet_mel_predictions", predictions.get("postnet_output")
            )
            pitch_predictions = predictions.get(
                "pitch_predictions", predictions.get("p_predictions")
            )
            energy_predictions = predictions.get(
                "energy_predictions", predictions.get("e_predictions")
            )
            log_duration_predictions = predictions.get(
                "log_duration_predictions", predictions.get("log_d_predictions")
            )
            src_masks = predictions.get("src_masks")
            mel_masks = predictions.get("mel_masks")

            if any(
                p is None
                for p in (
                    mel_predictions,
                    postnet_mel_predictions,
                    pitch_predictions,
                    energy_predictions,
                    log_duration_predictions,
                    src_masks,
                    mel_masks,
                )
            ):
                raise ValueError(
                    "Dictionary predictions must contain mel_predictions (or output), "
                    "postnet_mel_predictions (or postnet_output), pitch_predictions "
                    "(or p_predictions), energy_predictions (or e_predictions), "
                    "log_duration_predictions (or log_d_predictions), src_masks, and mel_masks."
                )
            return (
                mel_predictions,
                postnet_mel_predictions,
                pitch_predictions,
                energy_predictions,
                log_duration_predictions,
                src_masks,
                mel_masks,
            )

        if isinstance(predictions, (list, tuple)) and len(predictions) >= 8:
            # 10-tuple from FastSpeech2.forward:
            # (output, postnet_output, p_predictions, e_predictions,
            #  log_d_predictions, d_rounded, src_masks, mel_masks, src_lens, mel_lens)
            mel_predictions = predictions[0]
            postnet_mel_predictions = predictions[1]
            pitch_predictions = predictions[2]
            energy_predictions = predictions[3]
            log_duration_predictions = predictions[4]
            src_masks = predictions[6]
            mel_masks = predictions[7]
            return (
                mel_predictions,
                postnet_mel_predictions,
                pitch_predictions,
                energy_predictions,
                log_duration_predictions,
                src_masks,
                mel_masks,
            )

        raise ValueError(
            "predictions must be a 10-tuple from FastSpeech2.forward (or sequence of length >= 8) "
            "or a mapping containing all prediction and mask tensors."
        )

    def forward(
        self,
        inputs: Any,
        predictions: Any,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compute the FastSpeech 2 loss components and total loss.

        Args:
            inputs: 12-element training batch tuple, 6-element target tuple,
                or mapping with target tensors.
            predictions: 10-element model output tuple or mapping with predictions.

        Returns:
            6-tuple of scalar loss tensors:
            (total_loss, mel_loss, postnet_mel_loss, pitch_loss, energy_loss, duration_loss).
        """
        mel_targets, pitch_targets, energy_targets, duration_targets = (
            self._unpack_inputs(inputs)
        )
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            src_masks,
            mel_masks,
        ) = self._unpack_predictions(predictions)

        # Validate types
        for name, t in (
            ("mel_targets", mel_targets),
            ("pitch_targets", pitch_targets),
            ("energy_targets", energy_targets),
            ("duration_targets", duration_targets),
            ("mel_predictions", mel_predictions),
            ("postnet_mel_predictions", postnet_mel_predictions),
            ("pitch_predictions", pitch_predictions),
            ("energy_predictions", energy_predictions),
            ("log_duration_predictions", log_duration_predictions),
            ("src_masks", src_masks),
            ("mel_masks", mel_masks),
        ):
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}.")

        batch_size = mel_predictions.shape[0]
        if mel_predictions.dim() != 3 or postnet_mel_predictions.dim() != 3:
            raise ValueError(
                "mel_predictions and postnet_mel_predictions must be 3D [B, L_mel, C]."
            )
        if mel_targets.dim() != 3:
            raise ValueError("mel_targets must be 3D [B, L_mel, C].")
        if mel_predictions.shape[2] != mel_targets.shape[2]:
            raise ValueError(
                f"mel_predictions channels ({mel_predictions.shape[2]}) must match "
                f"mel_targets channels ({mel_targets.shape[2]})."
            )

        if (
            pitch_predictions.dim() != 2
            or pitch_targets.dim() != 2
            or energy_predictions.dim() != 2
            or energy_targets.dim() != 2
            or log_duration_predictions.dim() != 2
            or duration_targets.dim() != 2
            or src_masks.dim() != 2
            or mel_masks.dim() != 2
        ):
            raise ValueError(
                "pitch, energy, duration, and mask tensors must all be 2D [B, L]."
            )

        for name, t in (
            ("mel_targets", mel_targets),
            ("postnet_mel_predictions", postnet_mel_predictions),
            ("pitch_targets", pitch_targets),
            ("pitch_predictions", pitch_predictions),
            ("energy_targets", energy_targets),
            ("energy_predictions", energy_predictions),
            ("duration_targets", duration_targets),
            ("log_duration_predictions", log_duration_predictions),
            ("src_masks", src_masks),
            ("mel_masks", mel_masks),
        ):
            if t.shape[0] != batch_size:
                raise ValueError(
                    f"{name} batch size ({t.shape[0]}) does not match "
                    f"mel_predictions batch size ({batch_size})."
                )

        if (duration_targets < 0).any():
            raise ValueError("duration_targets must contain non-negative integer frame counts.")

        # Invert masks: in FastSpeech 2, True indicates PAD in model outputs;
        # after inversion, True indicates VALID tokens/frames.
        valid_src_masks = ~src_masks.bool()
        valid_mel_masks = ~mel_masks.bool()

        src_len_pred = valid_src_masks.shape[1]
        mel_len_pred = valid_mel_masks.shape[1]

        # DEC-10: Slice targets and predictions to match decoder-truncated lengths.
        # This prevents RuntimeError when ground-truth length exceeds Decoder max_seq_len.
        if mel_targets.shape[1] < mel_len_pred:
            raise ValueError(
                f"mel_targets temporal length ({mel_targets.shape[1]}) is shorter than "
                f"mel_masks temporal length ({mel_len_pred})."
            )
        mel_targets = mel_targets[:, :mel_len_pred, :].detach()
        mel_predictions = mel_predictions[:, :mel_len_pred, :]
        postnet_mel_predictions = postnet_mel_predictions[:, :mel_len_pred, :]

        if duration_targets.shape[1] < src_len_pred:
            raise ValueError(
                f"duration_targets length ({duration_targets.shape[1]}) is shorter than "
                f"src_masks length ({src_len_pred})."
            )
        log_duration_targets = torch.log(
            duration_targets[:, :src_len_pred].float() + 1.0
        ).detach()
        log_duration_predictions = log_duration_predictions[:, :src_len_pred]

        if self.pitch_feature_level == "phoneme_level":
            if pitch_targets.shape[1] < src_len_pred:
                raise ValueError(
                    f"pitch_targets length ({pitch_targets.shape[1]}) is shorter than "
                    f"src_masks length ({src_len_pred}) for phoneme-level pitch."
                )
            pitch_predictions = pitch_predictions[:, :src_len_pred]
            pitch_targets = pitch_targets[:, :src_len_pred].detach()
            pitch_mask = valid_src_masks
        else:  # frame_level
            if pitch_targets.shape[1] < mel_len_pred:
                raise ValueError(
                    f"pitch_targets length ({pitch_targets.shape[1]}) is shorter than "
                    f"mel_masks length ({mel_len_pred}) for frame-level pitch."
                )
            pitch_predictions = pitch_predictions[:, :mel_len_pred]
            pitch_targets = pitch_targets[:, :mel_len_pred].detach()
            pitch_mask = valid_mel_masks

        if self.energy_feature_level == "phoneme_level":
            if energy_targets.shape[1] < src_len_pred:
                raise ValueError(
                    f"energy_targets length ({energy_targets.shape[1]}) is shorter than "
                    f"src_masks length ({src_len_pred}) for phoneme-level energy."
                )
            energy_predictions = energy_predictions[:, :src_len_pred]
            energy_targets = energy_targets[:, :src_len_pred].detach()
            energy_mask = valid_src_masks
        else:  # frame_level
            if energy_targets.shape[1] < mel_len_pred:
                raise ValueError(
                    f"energy_targets length ({energy_targets.shape[1]}) is shorter than "
                    f"mel_masks length ({mel_len_pred}) for frame-level energy."
                )
            energy_predictions = energy_predictions[:, :mel_len_pred]
            energy_targets = energy_targets[:, :mel_len_pred].detach()
            energy_mask = valid_mel_masks

        # Masked selections
        mel_mask_3d = valid_mel_masks.unsqueeze(-1)
        mel_predictions_sel = mel_predictions.masked_select(mel_mask_3d)
        postnet_mel_predictions_sel = postnet_mel_predictions.masked_select(
            mel_mask_3d
        )
        mel_targets_sel = mel_targets.masked_select(mel_mask_3d)

        pitch_predictions_sel = pitch_predictions.masked_select(pitch_mask)
        pitch_targets_sel = pitch_targets.masked_select(pitch_mask)

        energy_predictions_sel = energy_predictions.masked_select(energy_mask)
        energy_targets_sel = energy_targets.masked_select(energy_mask)

        log_duration_predictions_sel = log_duration_predictions.masked_select(
            valid_src_masks
        )
        log_duration_targets_sel = log_duration_targets.masked_select(
            valid_src_masks
        )

        # Empty selection validation: prevents 0/0 NaN reductions
        if mel_targets_sel.numel() == 0 or mel_predictions_sel.numel() == 0:
            raise ValueError(
                "No valid mel frames selected by mel_masks; cannot compute mel reconstruction loss on empty selection."
            )
        if postnet_mel_predictions_sel.numel() == 0:
            raise ValueError(
                "No valid mel frames selected by mel_masks; cannot compute PostNet reconstruction loss on empty selection."
            )
        if pitch_predictions_sel.numel() == 0 or pitch_targets_sel.numel() == 0:
            raise ValueError(
                "No valid elements selected by mask for pitch loss computation."
            )
        if energy_predictions_sel.numel() == 0 or energy_targets_sel.numel() == 0:
            raise ValueError(
                "No valid elements selected by mask for energy loss computation."
            )
        if (
            log_duration_predictions_sel.numel() == 0
            or log_duration_targets_sel.numel() == 0
        ):
            raise ValueError(
                "No valid source tokens selected by src_masks for duration loss computation."
            )

        # Loss calculations
        mel_loss = self.mae_loss(mel_predictions_sel, mel_targets_sel)
        postnet_mel_loss = self.mae_loss(
            postnet_mel_predictions_sel, mel_targets_sel
        )

        pitch_loss = self.mse_loss(pitch_predictions_sel, pitch_targets_sel)
        energy_loss = self.mse_loss(energy_predictions_sel, energy_targets_sel)
        duration_loss = self.mse_loss(
            log_duration_predictions_sel, log_duration_targets_sel
        )

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )


__all__ = ["FastSpeech2Loss"]

