"""English FastSpeech 2 Inference Engine for Svarah Dataset & Assignment Evaluation.
Provides reproducible, high-fidelity speech synthesis for trained English FastSpeech 2 models:
- Compatible with original_fastspeech2/synthesize.py architecture and CLI arguments.
- Supports single-sentence synthesis, text manifest batch synthesis, and canonical assignment samples.
- Integrates text normalization (English.text.normalizer.normalize_text), CMU/LibriSpeech lexicon
  lookup, and automatic G2P fallback (g2p_en) for out-of-vocabulary words.
- Integrates the pretrained HiFi-GAN universal vocoder (22,050 Hz, 80 mel channels, hop=256).
- Supports speaker profile embedding selection (88 trained profiles in profile_stratified_v1)
  and centroid fallback for unseen speaker IDs.
- Provides fine-grained pitch, energy, and duration prosody controls.
- Emits verified 16-bit PCM WAV audio files with integrity validation (finite samples, exact sr).
"""
from __future__ import annotations
import argparse
import io
import json
import os
import re
import string
import sys
import time
import zipfile
from pathlib import Path
from typing import Any
import numpy as np
import scipy.io.wavfile as wavfile
import torch
import yaml
# Ensure project root and original_fastspeech2 are importable
_CURRENT_DIR = Path(__file__).resolve().parent
_BOOTSTRAP_ROOT = _CURRENT_DIR.parent
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from common.config import FastSpeech2Config
from common.model.fastspeech2 import FastSpeech2
from common.paths import (
    ENGLISH_RESULTS_DIR,
    HIFIGAN_CONFIG_PATH,
    HIFIGAN_WEIGHTS_PATH,
    HIFIGAN_WEIGHTS_ZIP_PATH,
    LEXICON_PATH,
    ORIGINAL_FS2_DIR,
    PROJECT_ROOT,
    ensure_output_dir,
    get_checkpoint_candidate_dirs,
    get_config_path,
    resolve_project_path,
    setup_project_paths,
)

setup_project_paths()
_PROJECT_ROOT = PROJECT_ROOT
_ORIGINAL_FS2_DIR = ORIGINAL_FS2_DIR

from English.text.normalizer import normalize_text
from original_fastspeech2 import hifigan
from original_fastspeech2.text import text_to_sequence
# Canonical Assignment Test Sentences (Preserved exactly as specified in the assignment)
ASSIGNMENT_SAMPLES: dict[str, str] = {
    "E1": (
        "The antidisestablishmentarianism movement gained considerable"
        " momentum during the parliamentary proceedings."
    ),
    "E2_a": (
        "The orchestra's conduct during the ceremony was praised by the audience."
    ),
    "E2_b": (
        "The researchers will conduct an experiment to evaluate the proposed method."
    ),
    "E3": "The rhyth m of the orchestra changed unexpectedly.",
    "E4": (
        "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."
    ),
}
DEFAULT_LEXICON_PATH = LEXICON_PATH
DEFAULT_HIFIGAN_CONFIG = HIFIGAN_CONFIG_PATH
DEFAULT_HIFIGAN_WEIGHTS = HIFIGAN_WEIGHTS_PATH
DEFAULT_HIFIGAN_ZIP = HIFIGAN_WEIGHTS_ZIP_PATH
DEFAULT_OUTPUT_DIR = ENGLISH_RESULTS_DIR / "audio"


def _resolve_file_path(path_like: str | Path) -> Path:
    """Resolve an input file path, checking direct path first then project-root-relative."""
    p = Path(path_like).expanduser()
    if p.is_file() or p.is_absolute():
        return p
    proj_p = resolve_project_path(p)
    return proj_p if proj_p.is_file() else p


def read_lexicon(lex_path: str | Path) -> dict[str, list[str]]:
    """Load a pronunciation dictionary into a lowercase-keyed lookup map."""
    lex_path = _resolve_file_path(lex_path)
    if not lex_path.is_file():
        raise FileNotFoundError(f"Pronunciation lexicon not found at: {lex_path}")
    lexicon: dict[str, list[str]] = {}
    with open(lex_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = re.split(r"\s+", line.strip())
            if not parts or not parts[0]:
                continue
            word = parts[0].lower()
            if word not in lexicon:
                lexicon[word] = parts[1:]
    return lexicon
def preprocess_english_text(
    raw_text: str,
    lexicon: dict[str, list[str]],
    text_cleaners: list[str] | tuple[str, ...],
    g2p_instance: Any = None,
) -> tuple[np.ndarray, str, str]:
    """Normalize raw text, phonemize into ARPAbet tokens, and convert to integer IDs.
    Args:
        raw_text: Raw English string.
        lexicon: Preloaded pronunciation lexicon dictionary.
        text_cleaners: Text cleaners list from preprocess.yaml (e.g. ['english_cleaners']).
        g2p_instance: Optional cached g2p_en.G2p instance for OOV fallback.
    Returns:
        (sequence_np, phonemes_curly_string, normalized_text_string)
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Cannot preprocess empty or whitespace-only text.")
    # 1. Auditable text normalization (numbers, dates, times, initialisms)
    normalized_text = normalize_text(raw_text)
    # 2. Tokenize into words and punctuation
    cleaned_for_lexicon = normalized_text.rstrip(string.punctuation)
    tokens = re.split(r"([,;.-\?\!\s+])", cleaned_for_lexicon)
    phones: list[str] = []
    g2p = g2p_instance
    for token in tokens:
        t = token.strip()
        if not t:
            continue
        if t in string.punctuation or t in (",", ".", ";", "-", "?", "!"):
            phones.append("sp")
            continue
        w_lower = t.lower()
        if w_lower in lexicon:
            phones.extend(lexicon[w_lower])
        else:
            # Fall back to G2P for out-of-vocabulary words
            if g2p is None:
                try:
                    from g2p_en import G2p
                    g2p = G2p()
                except ImportError as exc:
                    raise ImportError(
                        f"Out-of-vocabulary word '{t}' encountered and g2p_en is not installed. "
                        "Install it via 'pip install g2p_en'."
                    ) from exc
            g2p_phones = [p for p in g2p(t) if p.strip()]
            for p in g2p_phones:
                if p in string.punctuation:
                    phones.append("sp")
                else:
                    phones.append(p)
    if not phones:
        raise ValueError(f"No phonemes generated for text: {raw_text!r}")
    phones_str = "{" + "}{".join(phones) + "}"
    phones_str = re.sub(r"\{[^\w\s]?\}", "{sp}", phones_str)
    phones_str = phones_str.replace("}{", " ")
    sequence = text_to_sequence(phones_str, list(text_cleaners))
    if not sequence or len(sequence) == 0:
        raise ValueError(f"text_to_sequence produced an empty sequence for phonemes: {phones_str!r}")
    return np.array(sequence, dtype=np.int64), phones_str, normalized_text
def load_english_fastspeech2(
    checkpoint_path: str | Path,
    configs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    device: torch.device | str = "cpu",
) -> tuple[FastSpeech2, dict[str, Any]]:
    """Instantiate and load weights for FastSpeech 2 model."""
    ckpt_path = _resolve_file_path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found at: {ckpt_path}\n"
            "Ensure the checkpoint file exists or that the Google Drive checkpoint "
            "has been placed in the checkpoint directory."
        )
    preprocess_dict, model_dict, train_dict = configs
    raw_prep_path = Path(preprocess_dict["path"]["preprocessed_path"])
    stats_path = raw_prep_path / "stats.json"
    if not stats_path.is_file() and not raw_prep_path.is_absolute():
        stats_path = resolve_project_path(raw_prep_path) / "stats.json"
    if stats_path.is_file() and "stats" not in model_dict:
        with open(stats_path, encoding="utf-8") as f:
            model_dict["stats"] = json.load(f)
    fs2_cfg = FastSpeech2Config.from_dicts(preprocess_dict, model_dict, train_dict)
    model = FastSpeech2(fs2_cfg.preprocess, fs2_cfg.model)
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt_data["model"] if isinstance(ckpt_data, dict) and "model" in ckpt_data else ckpt_data
    # Strip any DataParallel 'module.' prefix
    cleaned_state_dict: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[key] = v
    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    metadata = {
        "global_step": ckpt_data.get("global_step") if isinstance(ckpt_data, dict) else None,
        "epoch": ckpt_data.get("epoch") if isinstance(ckpt_data, dict) else None,
        "best_val_loss": ckpt_data.get("best_val_loss") if isinstance(ckpt_data, dict) else None,
        "checkpoint_path": str(ckpt_path),
    }
    return model, metadata
def load_hifigan_vocoder(
    config_path: str | Path = DEFAULT_HIFIGAN_CONFIG,
    weights_path: str | Path = DEFAULT_HIFIGAN_WEIGHTS,
    zip_path: str | Path = DEFAULT_HIFIGAN_ZIP,
    device: torch.device | str = "cpu",
) -> torch.nn.Module:
    """Load pretrained HiFi-GAN vocoder generator."""
    cfg_p = _resolve_file_path(config_path)
    wt_p = _resolve_file_path(weights_path)
    zp_p = _resolve_file_path(zip_path)
    if not cfg_p.is_file():
        raise FileNotFoundError(f"HiFi-GAN config not found at: {cfg_p}")
    if not wt_p.is_file():
        if zp_p.is_file():
            print(f"[VOC] Unzipping HiFi-GAN weights from {zp_p.name}...")
            with zipfile.ZipFile(zp_p, "r") as zf:
                zf.extractall(wt_p.parent)
        else:
            raise FileNotFoundError(
                f"HiFi-GAN weights not found at {wt_p} and zip archive {zp_p} is missing."
            )
    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    h = hifigan.AttrDict(cfg)
    vocoder = hifigan.Generator(h)
    ckpt = torch.load(wt_p, map_location=device, weights_only=False)
    state = ckpt["generator"] if isinstance(ckpt, dict) and "generator" in ckpt else ckpt
    vocoder.load_state_dict(state)
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    for p in vocoder.parameters():
        p.requires_grad = False
    return vocoder
def synthesize_sentence(
    model: FastSpeech2,
    vocoder: torch.nn.Module,
    text: str,
    speaker_id: int,
    configs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    lexicon: dict[str, list[str]],
    control_values: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: torch.device | str = "cpu",
    g2p_instance: Any = None,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Synthesize speech waveform for a single raw sentence.
    Args:
        model: Loaded FastSpeech2 model in eval mode.
        vocoder: Loaded HiFi-GAN Generator vocoder in eval mode.
        text: Raw English sentence string.
        speaker_id: Multi-speaker profile integer ID.
        configs: (preprocess_config, model_config, train_config) dictionaries.
        lexicon: Preloaded pronunciation lexicon.
        control_values: (pitch_control, energy_control, duration_control).
        device: PyTorch device ('cpu', 'mps', or 'cuda').
        g2p_instance: Optional pre-instantiated G2p instance.
    Returns:
        (wav_int16_array, sampling_rate, synthesis_metadata)
    """
    preprocess_dict, model_dict, _ = configs
    cleaners = preprocess_dict["preprocessing"]["text"]["text_cleaners"]
    sr = int(preprocess_dict["preprocessing"]["audio"]["sampling_rate"])
    p_ctrl, e_ctrl, d_ctrl = control_values
    # 1. Speaker validation and fallback
    if model.speaker_emb is not None:
        if not isinstance(speaker_id, int) or speaker_id < 0:
            raise ValueError(f"Invalid speaker_id: {speaker_id!r}. Must be a non-negative integer.")
        if model.n_speaker is not None and speaker_id >= model.n_speaker:
            raise ValueError(
                f"speaker_id {speaker_id} exceeds model.n_speaker ({model.n_speaker}). "
                f"Valid range is [0, {model.n_speaker - 1}]."
            )
        # Note: 88 profiles were trained; if speaker_id >= 88, warn about untrained random vector
        if speaker_id >= 88:
            print(
                f"[WARN] speaker_id={speaker_id} is an untrained profile in profile_stratified_v1 "
                "(only profiles 0..87 were present in training). Falling back or proceeding as requested."
            )
    # 2. Text preprocessing
    seq_np, phones_str, norm_text = preprocess_english_text(
        text, lexicon, cleaners, g2p_instance=g2p_instance
    )
    src_len = len(seq_np)
    texts_t = torch.as_tensor(seq_np, dtype=torch.long, device=device).unsqueeze(0)  # [1, L_src]
    src_lens_t = torch.as_tensor([src_len], dtype=torch.long, device=device)  # [1]
    spk_t = (
        torch.as_tensor([speaker_id], dtype=torch.long, device=device)
        if model.speaker_emb is not None
        else None
    )
    # 3. Model forward pass (strictly inference: targets=None)
    with torch.no_grad():
        output = model(
            speakers=spk_t,
            texts=texts_t,
            src_lens=src_lens_t,
            mels=None,
            mel_lens=None,
            p_targets=None,
            e_targets=None,
            d_targets=None,
            p_control=p_ctrl,
            e_control=e_ctrl,
            d_control=d_ctrl,
        )
        # FastSpeech 2 returns 10-tuple: (output, postnet_output, p_pred, e_pred, log_d_pred, d_rounded, ...)
        postnet_mel = output[1]  # [1, T_mel, 80]
        mel_len = int(output[9][0].item())
        pred_dur = output[5][0, :src_len].cpu().numpy()
        pred_pitch = output[2][0].cpu().numpy()
        pred_energy = output[3][0].cpu().numpy()
        # 4. HiFi-GAN Vocoder inference: expects [B, 80, T_mel]
        mel_for_vocoder = postnet_mel[:, :mel_len, :].transpose(1, 2)
        wav_t = vocoder(mel_for_vocoder).squeeze(1)[0]  # [T_wav]
        wav_raw = wav_t.cpu().numpy()
    # 5. Format audio as 16-bit PCM
    if not np.isfinite(wav_raw).all():
        raise ValueError("Vocoder produced non-finite (NaN or Inf) audio samples.")
    # Peak normalize if amplitude exceeds [-1, 1]
    max_amp = float(np.max(np.abs(wav_raw)))
    if max_amp > 1.0:
        wav_raw = wav_raw / max_amp
    wav_int16 = (wav_raw * 32767.0).astype(np.int16)
    metadata: dict[str, Any] = {
        "raw_text": text,
        "normalized_text": norm_text,
        "phonemes": phones_str,
        "phoneme_tokens": src_len,
        "mel_frames": mel_len,
        "audio_samples": len(wav_int16),
        "duration_seconds": round(float(len(wav_int16)) / float(sr), 4),
        "speaker_id": speaker_id,
        "controls": {"pitch": p_ctrl, "energy": e_ctrl, "duration": d_ctrl},
        "variance_predictions": {
            "duration": pred_dur.tolist(),
            "pitch": pred_pitch.tolist(),
            "energy": pred_energy.tolist(),
            "pitch_mean": float(np.mean(pred_pitch)),
            "energy_mean": float(np.mean(pred_energy)),
        },
    }
    return wav_int16, sr, metadata
def verify_wav_file(wav_path: str | Path, expected_sr: int = 22050) -> dict[str, Any]:
    """Verify that a saved WAV file exists, is non-empty, and satisfies all acoustic invariants."""
    p = Path(wav_path)
    if not p.is_file():
        raise FileNotFoundError(f"WAV file does not exist: {p}")
    size_bytes = p.stat().st_size
    if size_bytes <= 44:  # 44 bytes is the standard empty RIFF WAV header size
        raise ValueError(f"WAV file {p} is empty or contains only header ({size_bytes} bytes).")
    sr, data = wavfile.read(p)
    if sr != expected_sr:
        raise ValueError(f"Sampling rate mismatch in {p}: expected {expected_sr}, got {sr}")
    if len(data) == 0:
        raise ValueError(f"WAV file {p} contains zero audio samples.")
    if not np.isfinite(data).all():
        raise ValueError(f"WAV file {p} contains NaN or Inf audio samples.")
    duration_sec = float(len(data)) / float(sr)
    return {
        "path": str(p),
        "file_size_bytes": size_bytes,
        "sampling_rate": sr,
        "num_samples": len(data),
        "duration_seconds": round(duration_sec, 4),
        "dtype": str(data.dtype),
        "is_finite": True,
    }
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference for English FastSpeech 2 (Svarah) model."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "batch", "assignment"],
        default="single",
        help="Inference mode: 'single' (one text), 'batch' (from source file), or 'assignment' (E1-E4).",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Raw sentence string to synthesize (required for mode='single').",
    )
    parser.add_argument(
        "--sample",
        type=str,
        choices=["E1", "E2_a", "E2_b", "E3", "E4", "all"],
        default=None,
        help="Synthesize specific assignment sample (E1, E2_a, E2_b, E3, E4, or 'all').",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to manifest file for batch mode (lines: id|speaker|text|raw_text or id|text).",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="Speaker profile index for multi-speaker synthesis (default: 0).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit path to FastSpeech 2 checkpoint (.pth.tar).",
    )
    parser.add_argument(
        "--restore_step",
        type=int,
        default=None,
        help="Checkpoint step number to restore (looks in train_config['path']['ckpt_path']).",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        default=str(_PROJECT_ROOT / "config" / "Svarah" / "preprocess.yaml"),
        help="Path to preprocess.yaml",
    )
    parser.add_argument(
        "-m",
        "--model_config",
        type=str,
        default=str(_PROJECT_ROOT / "config" / "Svarah" / "model.yaml"),
        help="Path to model.yaml",
    )
    parser.add_argument(
        "-t",
        "--train_config",
        type=str,
        default=str(_PROJECT_ROOT / "config" / "Svarah" / "train_profile_stratified_v1.yaml"),
        help="Path to train.yaml or train_profile_stratified_v1.yaml",
    )
    parser.add_argument(
        "--lexicon_path",
        type=str,
        default=str(DEFAULT_LEXICON_PATH),
        help="Path to LibriSpeech pronunciation dictionary.",
    )
    parser.add_argument(
        "--vocoder_weights",
        type=str,
        default=str(DEFAULT_HIFIGAN_WEIGHTS),
        help="Path to generator_universal.pth.tar vocoder weights.",
    )
    parser.add_argument(
        "--vocoder_config",
        type=str,
        default=str(DEFAULT_HIFIGAN_CONFIG),
        help="Path to HiFi-GAN config.json.",
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="Pitch multiplier (> 1.0 higher pitch, < 1.0 lower pitch).",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="Energy multiplier (> 1.0 louder, < 1.0 softer).",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="Speaking duration multiplier (> 1.0 slower rate, < 1.0 faster rate).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Destination directory for output WAV files.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Filename for single synthesis output (default: based on sample or timestamp).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cuda', 'mps', or 'cpu'). Default: auto-detect.",
    )
    return parser.parse_args()
def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
def resolve_checkpoint_path(args: argparse.Namespace, train_config: dict[str, Any]) -> Path:
    """Resolve checkpoint path from --checkpoint or --restore_step or default locations."""
    if args.checkpoint:
        p = _resolve_file_path(args.checkpoint)
        if not p.is_file():
            raise FileNotFoundError(f"Specified checkpoint not found: {p}")
        return p
    raw_ckpt_base = Path(train_config.get("path", {}).get("ckpt_path", "./output/ckpt/Svarah_profile_stratified_v1"))
    ckpt_base = raw_ckpt_base if raw_ckpt_base.is_dir() else resolve_project_path(raw_ckpt_base)
    # Also check Google Drive colab layout and local checkpoints/ folder
    candidate_dirs = [
        raw_ckpt_base,
        ckpt_base,
        *get_checkpoint_candidate_dirs("english"),
    ]
    if args.restore_step is not None:
        for cdir in candidate_dirs:
            p1 = cdir / f"checkpoint_{args.restore_step}.pth.tar"
            p2 = cdir / f"{args.restore_step}.pth.tar"
            if p1.is_file():
                return p1
            if p2.is_file():
                return p2
        raise FileNotFoundError(
            f"Checkpoint for step {args.restore_step} not found in candidate directories: {candidate_dirs}"
        )
    # If no step or path given, search for checkpoint_best.pth.tar or latest
    for cdir in candidate_dirs:
        best_p = cdir / "checkpoint_best.pth.tar"
        if best_p.is_file():
            return best_p
        latest_p = cdir / "checkpoint_latest.pth.tar"
        if latest_p.is_file():
            return latest_p
    raise FileNotFoundError(
        "No checkpoint specified and no default checkpoint_best.pth.tar found.\n"
        "Please provide --checkpoint <path_to_checkpoint.pth.tar> or ensure your checkpoint "
        "is placed in 'checkpoints/Svarah_profile_stratified_v1/'."
    )
def main() -> None:
    args = parse_args()
    print("=" * 72)
    print("FASTSPEECH 2 ENGLISH (SVARAH) INFERENCE PIPELINE")
    print("=" * 72)
    device = resolve_device(args.device)
    print(f"Compute Device       : {device}")
    # Load Configs
    prep_cfg_path = _resolve_file_path(args.preprocess_config)
    model_cfg_path = _resolve_file_path(args.model_config)
    train_cfg_path = _resolve_file_path(args.train_config)
    with open(prep_cfg_path, "r", encoding="utf-8") as f:
        prep_cfg = yaml.safe_load(f)
    with open(model_cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    with open(train_cfg_path, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    configs = (prep_cfg, model_cfg, train_cfg)
    # Resolve and Load FastSpeech 2 Checkpoint
    ckpt_path = resolve_checkpoint_path(args, train_cfg)
    print(f"Loading FastSpeech 2 : {ckpt_path}")
    model, ckpt_meta = load_english_fastspeech2(ckpt_path, configs, device=device)
    if ckpt_meta["best_val_loss"] is not None:
        print(f"Checkpoint Best Loss : {ckpt_meta['best_val_loss']:.4f} (Step {ckpt_meta['global_step']})")
    # Load HiFi-GAN Vocoder
    print(f"Loading HiFi-GAN Voc : {args.vocoder_weights}")
    vocoder = load_hifigan_vocoder(
        config_path=args.vocoder_config,
        weights_path=args.vocoder_weights,
        device=device,
    )
    # Load Lexicon
    print(f"Loading Lexicon      : {args.lexicon_path}")
    lexicon = read_lexicon(args.lexicon_path)
    print(f"Lexicon Words        : {len(lexicon):,}")
    out_dir = ensure_output_dir(resolve_project_path(args.output_dir))
    control_vals = (args.pitch_control, args.energy_control, args.duration_control)
    # Prepare sentences to synthesize
    tasks: list[tuple[str, str]] = []  # (output_name, raw_text)
    if args.sample:
        if args.sample == "all":
            tasks = list(ASSIGNMENT_SAMPLES.items())
        else:
            tasks = [(args.sample, ASSIGNMENT_SAMPLES[args.sample])]
    elif args.mode == "assignment":
        tasks = list(ASSIGNMENT_SAMPLES.items())
    elif args.mode == "single":
        if not args.text:
            raise ValueError("--text is required when --mode=single and --sample is not specified.")
        fname = args.output_filename or f"synth_{int(time.time())}.wav"
        if not fname.endswith(".wav"):
            fname += ".wav"
        tasks = [(Path(fname).stem, args.text)]
    elif args.mode == "batch":
        source_path = _resolve_file_path(args.source) if args.source else None
        if not source_path or not source_path.is_file():
            raise FileNotFoundError(f"Batch source file not found: {args.source}")
        with open(source_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 4:
                    uid, raw = parts[0], parts[3]
                elif len(parts) == 2:
                    uid, raw = parts[0], parts[1]
                else:
                    uid, raw = f"utt_{idx:04d}", parts[0]
                tasks.append((uid, raw))
    print(f"\nSynthesizing {len(tasks)} utterance(s) into: {out_dir}")
    print("-" * 72)
    results: list[dict[str, Any]] = []
    t_start_all = time.perf_counter()
    for item_idx, (out_name, raw_text) in enumerate(tasks, 1):
        t0 = time.perf_counter()
        print(f"\n[{item_idx}/{len(tasks)}] Processing: {out_name}")
        print(f"  Text: {raw_text}")
        wav_arr, sr, meta = synthesize_sentence(
            model=model,
            vocoder=vocoder,
            text=raw_text,
            speaker_id=args.speaker_id,
            configs=configs,
            lexicon=lexicon,
            control_values=control_vals,
            device=device,
        )
        out_path = out_dir / f"{out_name}.wav"
        wavfile.write(out_path, sr, wav_arr)
        audit = verify_wav_file(out_path, expected_sr=sr)
        elapsed = time.perf_counter() - t0
        audit["elapsed_sec"] = round(elapsed, 3)
        audit["phonemes"] = meta["phonemes"]
        results.append(audit)
        print(f"  Saved -> {out_path} ({audit['duration_seconds']}s, {audit['file_size_bytes']:,} bytes, {elapsed:.2f}s)")
    total_sec = time.perf_counter() - t_start_all
    print("\n" + "=" * 72)
    print(f"Inference Completed! Total time: {total_sec:.2f}s ({len(results)} files generated)")
    print("=" * 72)
if __name__ == "__main__":
    main()
