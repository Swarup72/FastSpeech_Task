"""Complete FastSpeech 2 acoustic model (`common.model.fastspeech2.FastSpeech2`).

Mirrors ``original_fastspeech2/model/fastspeech2.py`` (module hierarchy, ``forward``
arguments and the 10-tuple output contract) with the approved corrections:

* DEC-01: no module-global ``device``; everything inherits devices from its inputs.
* DEC-05: ``n_mel_channels`` and the PostNet hyper-parameters come from configuration.
* DEC-06: the source vocabulary size is configured per language (``n_src_vocab``).
* DEC-12: ``n_speaker`` / ``stats`` may be given in memory (otherwise ``speakers.json`` /
  ``stats.json`` are read from ``preprocessed_path``).
* DEF-01 (approved mask-aware PostNet): the Mel mask is forwarded to the PostNet so
  padded frames never influence valid frames through BatchNorm statistics.

Mask convention everywhere: ``True`` = padded position, ``False`` = valid.

Output contract (identical order to the original)::

    index  name                  shape                       dtype
    0      mel_output            [B, L_dec, n_mel_channels]  float32  (before PostNet)
    1      postnet_output        [B, L_dec, n_mel_channels]  float32  (PostNet(x) + x)
    2      p_predictions         [B, L_src] | [B, L_mel]     float32  (pitch_feature_level)
    3      e_predictions         [B, L_src] | [B, L_mel]     float32  (energy_feature_level)
    4      log_d_predictions     [B, L_src]                  float32  (log(d + 1))
    5      d_rounded             [B, L_src]                  int64    (frames per token)
    6      src_masks             [B, L_src]                  bool
    7      mel_masks             [B, L_dec]                  bool     (after Decoder)
    8      src_lens              [B]                         int64
    9      mel_lens              [B]                         int64    (from LengthRegulator)

``L_dec = min(L_mel, max_seq_len)`` in training, and ``L_mel`` in inference (the Decoder
expands its positional table dynamically). Ground-truth pitch/energy targets passed to
the model are *not* truncated; the loss is responsible for slicing (DEC-10).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from common.config import ModelConfig, PreprocessConfig
from common.model.postnet import PostNet
from common.model.variance_adaptor import VarianceAdaptor
from common.paths import resolve_project_path
from common.transformer.models import Decoder, Encoder
from common.utils.tools import get_mask_from_lengths

FASTSPEECH2_OUTPUT_NAMES: tuple[str, ...] = (
    "mel_output",
    "postnet_output",
    "p_predictions",
    "e_predictions",
    "log_d_predictions",
    "d_rounded",
    "src_masks",
    "mel_masks",
    "src_lens",
    "mel_lens",
)


class FastSpeech2(nn.Module):
    """FastSpeech 2: Encoder -> (+speaker) -> VarianceAdaptor -> Decoder -> mel_linear -> PostNet.

    Args:
        preprocess_config: ``PreprocessConfig`` (or mapping). Supplies ``n_mel_channels``,
            pitch/energy feature levels and ``preprocessed_path``.
        model_config: ``ModelConfig`` (or mapping).
        n_speaker: number of speakers (multi-speaker only). Defaults to
            ``model_config.n_speaker``, then ``len(speakers.json)``.
        stats: ``{"pitch": [min, max, ...], "energy": [min, max, ...]}``. Defaults to
            ``model_config.stats``, then ``stats.json``.
        n_src_vocab: only needed when ``model_config`` is a mapping without it.
    """

    def __init__(
        self,
        preprocess_config: PreprocessConfig | Mapping[str, Any],
        model_config: ModelConfig | Mapping[str, Any],
        n_speaker: int | None = None,
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
                "FastSpeech2 requires PreprocessConfig/ModelConfig (or mappings)."
            )
        self.model_config = model_config
        self.preprocess_config = preprocess_config

        self.n_mel_channels = preprocess_config.preprocessing.mel.n_mel_channels
        hidden = model_config.transformer.encoder_hidden
        if hidden != model_config.transformer.decoder_hidden:
            raise ValueError(
                "encoder_hidden must equal decoder_hidden for FastSpeech2 "
                f"({hidden} != {model_config.transformer.decoder_hidden})."
            )
        self.max_seq_len = model_config.max_seq_len

        # Same attribute names as the original => compatible state_dict prefixes.
        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(
            preprocess_config, model_config, stats=stats
        )
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config.transformer.decoder_hidden, self.n_mel_channels
        )
        self.postnet = PostNet.from_config(
            model_config, n_mel_channels=self.n_mel_channels
        )

        self.speaker_emb: nn.Embedding | None = None
        self.n_speaker: int | None = None
        if model_config.multi_speaker:
            n_spk = n_speaker if n_speaker is not None else model_config.n_speaker
            if n_spk is None:
                raw_prep_path = preprocess_config.path.preprocessed_path
                resolved_prep = resolve_project_path(raw_prep_path)
                resolved_candidate = resolved_prep / "speakers.json"
                if resolved_candidate.is_file():
                    path = str(resolved_candidate)
                else:
                    path = os.path.join(raw_prep_path, "speakers.json")
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        "multi_speaker=True but n_speaker was not provided and "
                        f"'{path}' does not exist."
                    )
                with open(path, encoding="utf-8") as f:
                    n_spk = len(json.load(f))
            if not isinstance(n_spk, int) or isinstance(n_spk, bool) or n_spk < 1:
                raise ValueError(f"n_speaker must be a positive int, got {n_spk!r}.")
            self.n_speaker = n_spk
            self.speaker_emb = nn.Embedding(n_spk, hidden)

    # ------------------------------------------------------------------ validation
    @staticmethod
    def _check_lengths(
        name: str, lens: Any, batch: int, device: torch.device
    ) -> torch.Tensor:
        if not isinstance(lens, torch.Tensor) or lens.dim() != 1:
            raise ValueError(f"{name} must be a 1D integer tensor [B].")
        if lens.dtype.is_floating_point or lens.dtype == torch.bool:
            raise ValueError(f"{name} must have an integer dtype, got {lens.dtype}.")
        if lens.shape[0] != batch:
            raise ValueError(f"{name} has length {lens.shape[0]}, expected {batch}.")
        return lens.to(device=device, dtype=torch.long)

    def forward(
        self,
        speakers: torch.Tensor | None,
        texts: torch.Tensor,
        src_lens: torch.Tensor,
        max_src_len: int | None = None,
        mels: torch.Tensor | None = None,
        mel_lens: torch.Tensor | None = None,
        max_mel_len: int | None = None,
        p_targets: torch.Tensor | None = None,
        e_targets: torch.Tensor | None = None,
        d_targets: torch.Tensor | None = None,
        p_control: float = 1.0,
        e_control: float = 1.0,
        d_control: float = 1.0,
    ) -> tuple[torch.Tensor, ...]:
        """Run the model (training with targets, or inference without them).

        Args:
            speakers: ``[B]`` int speaker ids (required when ``multi_speaker``; ignored
                otherwise, ``None`` allowed).
            texts: ``[B, L_src]`` int64 token ids in ``[0, n_src_vocab - 1]`` (0 = PAD).
            src_lens: ``[B]`` valid token counts (each in ``[1, L_src]``).
            max_src_len: optional; must equal ``texts.shape[1]`` if given.
            mels: optional ``[B, L_mel, n_mel_channels]`` ground truth (shape-checked only;
                the loss consumes it, the model does not).
            mel_lens: optional ``[B]`` ground-truth frame counts (training). Requires
                ``d_targets`` and must equal the per-utterance sum of ``d_targets``.
            max_mel_len: optional padded frame length (training).
            p_targets, e_targets: ground-truth pitch/energy (``[B, L_src]`` phoneme-level or
                ``[B, L_mel]`` frame-level); ``None`` uses predictions.
            d_targets: ``[B, L_src]`` int ground-truth durations; ``None`` predicts them.
            p_control, e_control, d_control: inference-time scale factors (see
                ``VarianceAdaptor``).

        Returns:
            The 10-tuple documented in the module docstring
            (see ``FASTSPEECH2_OUTPUT_NAMES``).
        """
        if not isinstance(texts, torch.Tensor) or texts.dim() != 2:
            raise ValueError("texts must be a 2D integer tensor [B, L_src].")
        batch, src_len_max = texts.shape
        device = texts.device
        if max_src_len is not None and max_src_len != src_len_max:
            raise ValueError(
                f"max_src_len ({max_src_len}) must equal texts.shape[1] ({src_len_max})."
            )
        src_lens_t = self._check_lengths("src_lens", src_lens, batch, device)
        if (src_lens_t < 1).any().item() or (src_lens_t > src_len_max).any().item():
            raise ValueError(
                f"src_lens must be in [1, {src_len_max}], got {src_lens_t.tolist()}."
            )

        if self.speaker_emb is not None:
            if speakers is None:
                raise ValueError("speakers is required when multi_speaker=True.")
            if (
                not isinstance(speakers, torch.Tensor)
                or speakers.dim() != 1
                or speakers.shape[0] != batch
                or speakers.dtype.is_floating_point
                or speakers.dtype == torch.bool
            ):
                raise ValueError("speakers must be an integer tensor of shape [B].")
            speakers = speakers.to(device=device, dtype=torch.long)
            if (speakers < 0).any().item() or (speakers >= self.n_speaker).any().item():
                raise ValueError(
                    f"speaker ids must be in [0, {self.n_speaker - 1}], "
                    f"got {speakers.tolist()}."
                )

        if mels is not None and (
            not isinstance(mels, torch.Tensor)
            or mels.dim() != 3
            or mels.shape[0] != batch
            or mels.shape[2] != self.n_mel_channels
        ):
            raise ValueError(
                f"mels must have shape [B, L_mel, {self.n_mel_channels}], got "
                f"{getattr(mels, 'shape', None)}."
            )

        mel_lens_in: torch.Tensor | None = None
        if mel_lens is not None:
            if d_targets is None:
                raise ValueError(
                    "mel_lens was given without d_targets; ground-truth mel lengths are "
                    "only meaningful together with ground-truth durations."
                )
            mel_lens_in = self._check_lengths("mel_lens", mel_lens, batch, device)
            if (mel_lens_in < 1).any().item():
                raise ValueError(
                    f"mel_lens must be >= 1, got {mel_lens_in.tolist()}."
                )

        src_masks = get_mask_from_lengths(src_lens_t, src_len_max)
        mel_masks = (
            get_mask_from_lengths(mel_lens_in, max_mel_len)
            if mel_lens_in is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, src_len_max, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens_out,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )
        if mel_lens_in is not None and not torch.equal(mel_lens_in, mel_lens_out):
            raise ValueError(
                f"mel_lens {mel_lens_in.tolist()} do not match the per-utterance sum of "
                f"d_targets {mel_lens_out.tolist()}."
            )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        # Mask-aware PostNet (approved): padded frames cannot affect valid frames.
        postnet_output = self.postnet(output, mel_masks) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens_t,
            mel_lens_out,
        )


__all__ = ["FastSpeech2", "FASTSPEECH2_OUTPUT_NAMES"]

