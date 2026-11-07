"""Telugu FastSpeech 2 Inference Engine for IndicTTS Dataset & Evaluation.

Provides reproducible, high-fidelity speech synthesis and prosody visualization
for trained Telugu FastSpeech 2 models:
- Mirrors English/inference.py architecture and CLI conventions while adapting all
  text, speaker, prosody normalization, and plotting components for Telugu.
- Integrates Telugu Unicode NFC normalization (Telugu.text.normalizer.normalize_text),
  grapheme tokenization (Telugu.text.normalizer.text_to_sequence / sequence_to_text),
  and IndicMFA word-boundary pause mapping (@sp, token ID 4).
- Avoids feeding untrained ordinary-space and punctuation token IDs (7..18) to the encoder.
- Safely handles digits and reports any out-of-vocabulary (<unk>, token ID 1) characters
  prior to synthesis without silently dropping meaningful Telugu characters.
- Supports multi-speaker synthesis across both trained IndicTTS Telugu speakers:
  spk_000 (female, ID 0) and spk_001 (male, ID 1).
- Reuses the pretrained HiFi-GAN universal vocoder (22,050 Hz, 80 mel channels, hop=256).
- Performs exact inverse normalization of predicted frame-level pitch (Hz) and energy
  (STFT L2-norm) using preprocessed_data/IndicTTS/stats.json.
- Generates verified 16-bit PCM WAV audio, duration/pitch/energy plots, and JSON metadata
  under Results/Telugu/{audio,plots,metadata}/.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
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
    HIFIGAN_CONFIG_PATH,
    HIFIGAN_WEIGHTS_PATH,
    HIFIGAN_WEIGHTS_ZIP_PATH,
    ORIGINAL_FS2_DIR,
    PROJECT_ROOT,
    TELUGU_RESULTS_DIR,
    ensure_output_dir,
    get_checkpoint_candidate_dirs,
    get_config_path,
    get_preprocessed_dir,
    resolve_project_path,
    setup_project_paths,
)

setup_project_paths()
_PROJECT_ROOT = PROJECT_ROOT
_ORIGINAL_FS2_DIR = ORIGINAL_FS2_DIR

try:
    from common import vocoder as hifigan
except ImportError:
    from original_fastspeech2 import hifigan
from Telugu.text.normalizer import normalize_text, sequence_to_text, text_to_sequence
from Telugu.text.symbols import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    VOCAB_SIZE,
    id_to_symbol,
    symbol_to_id,
    symbols,
)

# ==============================================================================
# 1. CONSTANTS & CANONICAL CONFIGURATION DEFAULTS
# ==============================================================================

DEFAULT_PREPROCESS_CONFIG = get_config_path("telugu", "preprocess")
DEFAULT_MODEL_CONFIG = get_config_path("telugu", "model")
DEFAULT_TRAIN_CONFIG = get_config_path("telugu", "train")
DEFAULT_STATS_PATH = get_preprocessed_dir("telugu") / "stats.json"
DEFAULT_SPEAKERS_PATH = get_preprocessed_dir("telugu") / "speakers.json"

DEFAULT_HIFIGAN_CONFIG = HIFIGAN_CONFIG_PATH
DEFAULT_HIFIGAN_WEIGHTS = HIFIGAN_WEIGHTS_PATH
DEFAULT_HIFIGAN_ZIP = HIFIGAN_WEIGHTS_ZIP_PATH

DEFAULT_RESULTS_ROOT = TELUGU_RESULTS_DIR
DEFAULT_AUDIO_DIR = DEFAULT_RESULTS_ROOT / "audio"
DEFAULT_PLOTS_DIR = DEFAULT_RESULTS_ROOT / "plots"
DEFAULT_METADATA_DIR = DEFAULT_RESULTS_ROOT / "metadata"


def _resolve_file_path(path_like: str | Path) -> Path:
    """Resolve an input file path, checking direct path first then project-root-relative."""
    p = Path(path_like).expanduser()
    if p.is_file() or p.is_absolute():
        return p
    proj_p = resolve_project_path(p)
    return proj_p if proj_p.is_file() else p

EXPECTED_BEST_STEP = 13000
EXPECTED_VOCAB_SIZE = 101
EXPECTED_NUM_SPEAKERS = 2
EXPECTED_SAMPLING_RATE = 22050
EXPECTED_N_MEL_CHANNELS = 80
EXPECTED_HOP_LENGTH = 256
EXPECTED_FFT_SIZE = 1024

# Verified Telugu training normalization statistics from preprocessed_data/IndicTTS/stats.json
TELUGU_PITCH_MEAN_HZ = 195.619853412152
TELUGU_PITCH_STD_HZ = 46.419600809453875
TELUGU_ENERGY_MEAN = 28.10710166547053
TELUGU_ENERGY_STD = 26.661124763887056

# Exact Telugu speaker mapping from preprocessed_data/IndicTTS/speakers.json
TELUGU_SPEAKER_MAP: dict[str, int] = {
    "spk_000": 0,
    "spk_001": 1,
}
TELUGU_SPEAKER_INFO: dict[int, dict[str, Any]] = {
    0: {"speaker_id": 0, "speaker_code": "spk_000", "gender": "female"},
    1: {"speaker_id": 1, "speaker_code": "spk_001", "gender": "male"},
}

# Canonical Telugu Assignment Test Sentences (Preserved exactly as specified)
ASSIGNMENT_SAMPLES: dict[str, str] = {
    "T1": "న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
    "T2_a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
    "T2_b": "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
    "T3": "పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",
    "T4": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
}

SAMPLE_ID_ALIASES: dict[str, str] = {
    "T1": "T1",
    "T2_a": "T2_a",
    "T2(a)": "T2_a",
    "T2a": "T2_a",
    "T2_b": "T2_b",
    "T2(b)": "T2_b",
    "T2b": "T2_b",
    "T3": "T3",
    "T4": "T4",
}

# Untrained ordinary-space and punctuation token IDs (indices 7..18 in Telugu/text/symbols.py)
UNTRAINED_PUNCT_AND_SPACE_IDS: frozenset[int] = frozenset(range(7, 19))

# Clause/phrase boundary punctuation mapped to @sp (token ID 4)
_CLAUSE_PAUSE_CHARS: frozenset[str] = frozenset({" ", ",", ";", ":", "-", ".", "!", "?"})
# Non-phonemic quote/bracket characters stripped without emitting untrained IDs 7..18
_IGNORED_QUOTE_BRACKET_CHARS: frozenset[str] = frozenset({"'", '"', "(", ")"})

# Spoken Telugu words for digits 0..9 (composed exclusively of trained Telugu graphemes)
_DIGIT_TO_TELUGU_WORD: dict[str, str] = {
    "0": "సున్నా",
    "1": "ఒకటి",
    "2": "రెండు",
    "3": "మూడు",
    "4": "నాలుగు",
    "5": "ఐదు",
    "6": "ఆరు",
    "7": "ఏడు",
    "8": "ఎనిమిది",
    "9": "తొమ్మిది",
    "౦": "సున్నా",
    "౧": "ఒకటి",
    "౨": "రెండు",
    "౩": "మూడు",
    "౪": "నాలుగు",
    "౫": "ఐదు",
    "౬": "ఆరు",
    "౭": "ఏడు",
    "౮": "ఎనిమిది",
    "౯": "తొమ్మిది",
}
_ASCII_TO_TELUGU_DIGIT: dict[str, str] = {
    "0": "౦",
    "1": "౧",
    "2": "౨",
    "3": "౩",
    "4": "౪",
    "5": "౫",
    "6": "౬",
    "7": "౭",
    "8": "౮",
    "9": "౯",
}

_CURLY_SEQ_RE = re.compile(r"^\{(.+)\}$")

# Preferred Telugu-capable fonts for Matplotlib plots with safe fallbacks
TELUGU_FONT_CANDIDATES: tuple[str, ...] = (
    "Kohinoor Telugu",
    "Noto Sans Telugu",
    "Telugu Sangam MN",
    "Telugu MN",
    "Lava Telugu",
    "Tiro Telugu",
    "Lohit Telugu",
    "Arial Unicode MS",
    "DejaVu Sans",
)


# ==============================================================================
# 2. SPEAKER & TOKENIZER UTILITIES
# ==============================================================================


def resolve_telugu_speaker(
    speaker: int | str = 0,
    speakers_json_path: str | Path = DEFAULT_SPEAKERS_PATH,
) -> dict[str, Any]:
    """Validate and resolve a Telugu speaker identifier (0/1, spk_000/spk_001, female/male).

    Args:
        speaker: Speaker integer ID (0 or 1), code ('spk_000', 'spk_001'), or label ('female', 'male').
        speakers_json_path: Path to speakers.json for verification.

    Returns:
        Dictionary with keys: 'speaker_id' (int), 'speaker_code' (str), 'gender' (str).
    """
    spk_path = _resolve_file_path(speakers_json_path)
    if spk_path.is_file():
        with open(spk_path, "r", encoding="utf-8") as f:
            disk_map = json.load(f)
        if disk_map != TELUGU_SPEAKER_MAP:
            raise ValueError(
                f"speakers.json mapping mismatch at {spk_path}: expected {TELUGU_SPEAKER_MAP}, got {disk_map}"
            )

    if isinstance(speaker, bool):
        raise ValueError(f"Invalid speaker identifier: {speaker!r}. Must be 0, 1, 'spk_000', or 'spk_001'.")

    if isinstance(speaker, int):
        if speaker not in TELUGU_SPEAKER_INFO:
            raise ValueError(
                f"Invalid Telugu speaker_id={speaker}. Valid speaker IDs are 0 ('spk_000', female) "
                "and 1 ('spk_001', male)."
            )
        return dict(TELUGU_SPEAKER_INFO[speaker])

    if isinstance(speaker, str):
        s_clean = speaker.strip().lower()
        if s_clean in ("0", "spk_000", "female", "telugu_female"):
            return dict(TELUGU_SPEAKER_INFO[0])
        if s_clean in ("1", "spk_001", "male", "telugu_male"):
            return dict(TELUGU_SPEAKER_INFO[1])

    raise ValueError(
        f"Invalid Telugu speaker={speaker!r}. Expected 0 ('spk_000', female) or 1 ('spk_001', male)."
    )


def preprocess_telugu_text(
    raw_text: str,
    digit_mode: str = "telugu_digits",
    strict_no_unk: bool = False,
) -> tuple[np.ndarray, list[str], str, str, list[dict[str, Any]]]:
    """Normalize raw Telugu text and tokenize into IndicMFA grapheme + @sp symbol IDs.

    Guarantees:
    1. Applies canonical Unicode NFC normalization via `Telugu.text.normalizer.normalize_text`.
    2. Maps word boundaries (spaces) and internal clause punctuation (`,`, `;`, `:`, `-`, `.`, `!`, `?`)
       to `@sp` (token ID 4), collapsing consecutive pauses and trimming leading/trailing `@sp`.
    3. Never emits untrained ordinary-space or punctuation token IDs (indices 7..18).
    4. Preserves all meaningful Telugu characters (vowels, consonants, matras, virama, modifiers, digits).
    5. Safely converts ASCII digits ('0'..'9') to Telugu digits ('౦'..'౯', IDs 91..100) when
       `digit_mode='telugu_digits'` or spoken Telugu words when `digit_mode='words'`.
    6. Reports any out-of-vocabulary (`<unk>`, token ID 1) tokens before inference.

    Args:
        raw_text: Raw Telugu string or pre-tokenized `{tok1 tok2 ...}` string.
        digit_mode: 'telugu_digits' (maps ASCII 0-9 to Telugu ౦-౯) or 'words' (expands digits to Telugu words).
        strict_no_unk: If True, raises ValueError when any UNK token is encountered.

    Returns:
        Tuple of:
            - sequence_np: 1D np.ndarray (int64) of symbol IDs.
            - tokens: List of token strings (e.g. ['వ', 'ై', ..., '@sp', ...]).
            - curly_tokens_str: Space-separated token string enclosed in braces `"{...}"`.
            - normalized_text: NFC-normalized text string.
            - unk_report: List of dicts describing any UNK characters/tokens encountered.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Cannot preprocess empty or whitespace-only Telugu text.")
    if digit_mode not in ("telugu_digits", "words"):
        raise ValueError(f"Invalid digit_mode={digit_mode!r}. Expected 'telugu_digits' or 'words'.")

    stripped_input = raw_text.strip()
    curly_match = _CURLY_SEQ_RE.match(stripped_input)
    unk_report: list[dict[str, Any]] = []
    tokens: list[str] = []

    if curly_match:
        # Input is already a curly-braced token sequence "{tok1 tok2 ...}"
        raw_tokens = curly_match.group(1).split()
        for idx, tok in enumerate(raw_tokens):
            if tok in ("@sp", "sp", "@sil", "sil"):
                if tokens and tokens[-1] != "@sp":
                    tokens.append("@sp")
            elif tok in _CLAUSE_PAUSE_CHARS:
                if tokens and tokens[-1] != "@sp":
                    tokens.append("@sp")
            elif tok in _IGNORED_QUOTE_BRACKET_CHARS:
                continue
            elif tok in _ASCII_TO_TELUGU_DIGIT:
                if digit_mode == "words":
                    word = _DIGIT_TO_TELUGU_WORD[tok]
                    for ch in normalize_text(word):
                        tokens.append(ch)
                else:
                    tokens.append(_ASCII_TO_TELUGU_DIGIT[tok])
            elif tok in symbol_to_id and symbol_to_id[tok] not in UNTRAINED_PUNCT_AND_SPACE_IDS and tok != "_":
                if tok == "<unk>":
                    unk_report.append({"position": idx, "char": tok, "reason": "explicit_unk_token"})
                tokens.append(tok)
            else:
                unk_report.append({"position": idx, "char": tok, "reason": "unmapped_token_in_curly_sequence"})
                tokens.append("<unk>")
        while tokens and tokens[-1] == "@sp":
            tokens.pop()
        normalized_text = sequence_to_text([symbol_to_id.get(t, UNK_ID) for t in tokens])
    else:
        # 1. Canonical Unicode NFC + Telugu text normalization
        normalized_text = normalize_text(raw_text)
        if not normalized_text:
            raise ValueError(f"Text became empty after normalization: {raw_text!r}")

        # 2. Optional spoken-word expansion for digits when digit_mode == 'words'
        if digit_mode == "words":
            expanded_chars: list[str] = []
            for ch in normalized_text:
                if ch in _DIGIT_TO_TELUGU_WORD:
                    expanded_chars.append(f" {_DIGIT_TO_TELUGU_WORD[ch]} ")
                else:
                    expanded_chars.append(ch)
            iter_text = normalize_text("".join(expanded_chars))
        else:
            iter_text = "".join(_ASCII_TO_TELUGU_DIGIT.get(ch, ch) for ch in normalized_text)

        # 3. Character-by-character grapheme & pause tokenization
        for pos, ch in enumerate(iter_text):
            if ch in _CLAUSE_PAUSE_CHARS:
                if tokens and tokens[-1] != "@sp":
                    tokens.append("@sp")
            elif ch in _IGNORED_QUOTE_BRACKET_CHARS:
                continue
            elif ch in symbol_to_id:
                tok_id = symbol_to_id[ch]
                if tok_id in UNTRAINED_PUNCT_AND_SPACE_IDS or tok_id == PAD_ID:
                    if tokens and tokens[-1] != "@sp":
                        tokens.append("@sp")
                else:
                    tokens.append(ch)
            else:
                unk_report.append(
                    {
                        "position": pos,
                        "char": ch,
                        "codepoint": f"U+{ord(ch):04X}",
                        "reason": "unsupported_character_not_in_telugu_vocab",
                    }
                )
                tokens.append("<unk>")

        # Trim trailing @sp so utterance ends on a spoken grapheme (matching train.txt/val.txt)
        while tokens and tokens[-1] == "@sp":
            tokens.pop()
        while tokens and tokens[0] == "@sp":
            tokens.pop(0)

    if not tokens:
        raise ValueError(f"No valid Telugu tokens produced for text: {raw_text!r}")

    curly_tokens_str = "{" + " ".join(tokens) + "}"
    seq_list = text_to_sequence(curly_tokens_str)
    if not seq_list or len(seq_list) != len(tokens):
        raise ValueError(
            f"Token length mismatch in text_to_sequence: {len(seq_list)} IDs vs {len(tokens)} tokens."
        )

    seq_np = np.asarray(seq_list, dtype=np.int64)

    # Validate vocabulary ID bounds and guarantee zero untrained space/punctuation IDs (7..18)
    if (seq_np <= PAD_ID).any() or (seq_np >= VOCAB_SIZE).any():
        invalid_ids = seq_np[(seq_np <= PAD_ID) | (seq_np >= VOCAB_SIZE)].tolist()
        raise ValueError(f"Invalid vocabulary token IDs {invalid_ids} outside [1, {VOCAB_SIZE - 1}].")

    untrained_found = [int(i) for i in seq_np if int(i) in UNTRAINED_PUNCT_AND_SPACE_IDS]
    if untrained_found:
        raise RuntimeError(
            f"Tokenizer emitted untrained space/punctuation token IDs {untrained_found}; "
            "word boundaries and pauses must map to @sp (ID 4)."
        )

    # Report UNK tokens before inference
    unk_positions = [idx for idx, tid in enumerate(seq_list) if tid == UNK_ID]
    if unk_positions:
        unk_msg = (
            f"[WARN] {len(unk_positions)} UNK token(s) (ID {UNK_ID}) detected at token indices "
            f"{unk_positions}: {unk_report}"
        )
        print(unk_msg)
        if strict_no_unk:
            raise ValueError(unk_msg)

    return seq_np, tokens, curly_tokens_str, normalized_text, unk_report


# ==============================================================================
# 3. MODEL LOADING, CHECKPOINT VALIDATION & PROSODY INVERSE NORMALIZATION
# ==============================================================================


def load_telugu_stats(stats_path: str | Path = DEFAULT_STATS_PATH) -> dict[str, Any]:
    """Load and validate Telugu pitch and energy normalization statistics from stats.json."""
    p = _resolve_file_path(stats_path)
    if not p.is_file():
        raise FileNotFoundError(f"Telugu normalization stats.json not found at: {p}")
    with open(p, "r", encoding="utf-8") as f:
        stats = json.load(f)
    for key in ("pitch", "energy"):
        if key not in stats or len(stats[key]) != 4:
            raise ValueError(f"Malformed '{key}' entry in {p}: expected [min, max, mean, std].")
    return stats


def inverse_normalize_prosody(
    pitch_norm: np.ndarray | Sequence[float],
    energy_norm: np.ndarray | Sequence[float],
    stats: dict[str, Any] | None = None,
    hop_length: int = EXPECTED_HOP_LENGTH,
    sampling_rate: int = EXPECTED_SAMPLING_RATE,
) -> dict[str, np.ndarray | float]:
    """Convert normalized model pitch/energy predictions to physical units (Hz and L2-norm).

    Formulas (from preprocessed_data/IndicTTS/stats.json):
        pitch_hz     = pitch_prediction * 46.419600809453875 + 195.619853412152
        energy       = energy_prediction * 26.661124763887056 + 28.10710166547053
        time_seconds = frame_index * 256 / 22050
    """
    if stats is not None:
        pitch_mean = float(stats["pitch"][2])
        pitch_std = float(stats["pitch"][3])
        energy_mean = float(stats["energy"][2])
        energy_std = float(stats["energy"][3])
    else:
        pitch_mean = TELUGU_PITCH_MEAN_HZ
        pitch_std = TELUGU_PITCH_STD_HZ
        energy_mean = TELUGU_ENERGY_MEAN
        energy_std = TELUGU_ENERGY_STD

    p_arr = np.asarray(pitch_norm, dtype=np.float64)
    e_arr = np.asarray(energy_norm, dtype=np.float64)
    if len(p_arr) != len(e_arr):
        raise ValueError(
            f"Pitch and energy frame counts must match: got {len(p_arr)} vs {len(e_arr)}."
        )

    pitch_hz = p_arr * pitch_std + pitch_mean
    energy_raw = e_arr * energy_std + energy_mean
    frame_indices = np.arange(len(p_arr), dtype=np.float64)
    time_seconds = frame_indices * float(hop_length) / float(sampling_rate)

    return {
        "pitch_hz": pitch_hz,
        "energy_raw": energy_raw,
        "time_seconds": time_seconds,
        "pitch_mean_hz": pitch_mean,
        "pitch_std_hz": pitch_std,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
    }


def load_telugu_fastspeech2(
    checkpoint_path: str | Path,
    configs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    device: torch.device | str = "cpu",
    expected_step: int | None = EXPECTED_BEST_STEP,
    strict_step: bool = False,
) -> tuple[FastSpeech2, dict[str, Any]]:
    """Instantiate, validate, and load checkpoint weights for the Telugu FastSpeech 2 model."""
    ckpt_path = _resolve_file_path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Telugu FastSpeech 2 checkpoint not found at: {ckpt_path}\n"
            "Ensure checkpoint_best.pth.tar (step 13000) is placed in 'output/ckpt/IndicTTS/' "
            "or pass --checkpoint <path>."
        )
    if ckpt_path.stat().st_size <= 0:
        raise ValueError(f"Checkpoint file is empty (0 bytes): {ckpt_path}")

    preprocess_dict, model_dict, train_dict = configs
    model_dict = dict(model_dict)
    raw_prep = Path(preprocess_dict["path"]["preprocessed_path"])
    stats_path = raw_prep / "stats.json"
    if not stats_path.is_file() and not raw_prep.is_absolute():
        stats_path = resolve_project_path(raw_prep) / "stats.json"
    if stats_path.is_file() and "stats" not in model_dict:
        model_dict["stats"] = load_telugu_stats(stats_path)

    fs2_cfg = FastSpeech2Config.from_dicts(preprocess_dict, model_dict, train_dict)

    # Validate configuration invariants against Telugu IndicTTS specifications
    if fs2_cfg.model.n_src_vocab != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            f"Telugu vocab size mismatch: expected {EXPECTED_VOCAB_SIZE}, got {fs2_cfg.model.n_src_vocab}"
        )
    if not fs2_cfg.model.multi_speaker or fs2_cfg.model.n_speaker != EXPECTED_NUM_SPEAKERS:
        raise ValueError(
            f"Telugu speaker config mismatch: expected multi_speaker=True, n_speaker={EXPECTED_NUM_SPEAKERS}, "
            f"got multi_speaker={fs2_cfg.model.multi_speaker}, n_speaker={fs2_cfg.model.n_speaker}"
        )
    if fs2_cfg.preprocess.preprocessing.audio.sampling_rate != EXPECTED_SAMPLING_RATE:
        raise ValueError(
            f"Sampling rate mismatch: expected {EXPECTED_SAMPLING_RATE}, "
            f"got {fs2_cfg.preprocess.preprocessing.audio.sampling_rate}"
        )
    if fs2_cfg.preprocess.preprocessing.mel.n_mel_channels != EXPECTED_N_MEL_CHANNELS:
        raise ValueError(
            f"Mel channels mismatch: expected {EXPECTED_N_MEL_CHANNELS}, "
            f"got {fs2_cfg.preprocess.preprocessing.mel.n_mel_channels}"
        )
    if fs2_cfg.preprocess.preprocessing.stft.hop_length != EXPECTED_HOP_LENGTH:
        raise ValueError(
            f"Hop length mismatch: expected {EXPECTED_HOP_LENGTH}, "
            f"got {fs2_cfg.preprocess.preprocessing.stft.hop_length}"
        )
    if fs2_cfg.preprocess.preprocessing.stft.filter_length != EXPECTED_FFT_SIZE:
        raise ValueError(
            f"FFT filter_length mismatch: expected {EXPECTED_FFT_SIZE}, "
            f"got {fs2_cfg.preprocess.preprocessing.stft.filter_length}"
        )

    model = FastSpeech2(fs2_cfg.preprocess, fs2_cfg.model)

    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = (
        ckpt_data["model"] if isinstance(ckpt_data, dict) and "model" in ckpt_data else ckpt_data
    )
    if not isinstance(state_dict, dict):
        raise ValueError(f"Invalid checkpoint state_dict format in {ckpt_path}")

    # Strip any DataParallel 'module.' prefix
    cleaned_state_dict: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[key] = v

    # Validate key tensor shapes before loading
    emb_key = "encoder.src_word_emb.weight"
    spk_key = "speaker_emb.weight"
    mel_key = "mel_linear.weight"
    if emb_key in cleaned_state_dict:
        vocab_rows = int(cleaned_state_dict[emb_key].shape[0])
        if vocab_rows != EXPECTED_VOCAB_SIZE:
            raise ValueError(
                f"Checkpoint {emb_key} has {vocab_rows} rows, expected {EXPECTED_VOCAB_SIZE} for Telugu."
            )
    if spk_key in cleaned_state_dict:
        spk_rows = int(cleaned_state_dict[spk_key].shape[0])
        if spk_rows != EXPECTED_NUM_SPEAKERS:
            raise ValueError(
                f"Checkpoint {spk_key} has {spk_rows} rows, expected {EXPECTED_NUM_SPEAKERS} for Telugu."
            )
    if mel_key in cleaned_state_dict:
        mel_rows = int(cleaned_state_dict[mel_key].shape[0])
        if mel_rows != EXPECTED_N_MEL_CHANNELS:
            raise ValueError(
                f"Checkpoint {mel_key} has {mel_rows} rows, expected {EXPECTED_N_MEL_CHANNELS}."
            )

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    global_step = ckpt_data.get("global_step") if isinstance(ckpt_data, dict) else None
    epoch = ckpt_data.get("epoch") if isinstance(ckpt_data, dict) else None
    best_val_loss = ckpt_data.get("best_val_loss") if isinstance(ckpt_data, dict) else None

    if expected_step is not None and global_step is not None and int(global_step) != int(expected_step):
        msg = (
            f"[WARN] Loaded checkpoint step ({global_step}) differs from expected best step ({expected_step})."
        )
        print(msg)
        if strict_step:
            raise ValueError(msg)

    metadata = {
        "global_step": int(global_step) if global_step is not None else None,
        "epoch": int(epoch) if epoch is not None else None,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "checkpoint_path": str(ckpt_path),
    }
    return model, metadata


def load_hifigan_vocoder(
    config_path: str | Path = DEFAULT_HIFIGAN_CONFIG,
    weights_path: str | Path = DEFAULT_HIFIGAN_WEIGHTS,
    zip_path: str | Path = DEFAULT_HIFIGAN_ZIP,
    device: torch.device | str = "cpu",
) -> torch.nn.Module:
    """Load pretrained universal HiFi-GAN vocoder generator."""
    cfg_p = _resolve_file_path(config_path)
    wt_p = _resolve_file_path(weights_path)
    zp_p = _resolve_file_path(zip_path)
    if not cfg_p.is_file():
        raise FileNotFoundError(f"HiFi-GAN config not found at: {cfg_p}")
    if not wt_p.is_file():
        if str(weights_path) == str(DEFAULT_HIFIGAN_WEIGHTS) or weights_path is None:
            candidate_weights = [
                PROJECT_ROOT / "checkpoints" / "hifigan" / "generator_universal.pth.tar",
                PROJECT_ROOT / "original_fastspeech2" / "hifigan" / "generator_universal.pth.tar",
                Path("/content/drive/MyDrive/FastSpeech2_Telugu/generator_universal.pth.tar"),
                Path("/content/drive/MyDrive/FastSpeech2_Telugu/original_fastspeech2/hifigan/generator_universal.pth.tar"),
            ]
            found_wt = next((p for p in candidate_weights if p.is_file()), None)
            if found_wt is not None:
                wt_p = found_wt
        if not wt_p.is_file():
            candidate_zips = [
                zp_p,
                Path("/content/drive/MyDrive/FastSpeech2_Telugu/generator_universal.pth.tar.zip"),
                Path("/content/drive/MyDrive/FastSpeech2_Telugu/original_fastspeech2/hifigan/generator_universal.pth.tar.zip"),
            ]
            found_zp = next((p for p in candidate_zips if p.is_file()), None)
            if found_zp is not None:
                wt_p.parent.mkdir(parents=True, exist_ok=True)
                print(f"[VOC] Unzipping HiFi-GAN weights from {found_zp}...")
                with zipfile.ZipFile(found_zp, "r") as zf:
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


def verify_wav_file(wav_path: str | Path, expected_sr: int = EXPECTED_SAMPLING_RATE) -> dict[str, Any]:
    """Verify that a saved WAV file exists, is non-empty, and satisfies all acoustic invariants."""
    p = Path(wav_path)
    if not p.is_file():
        raise FileNotFoundError(f"WAV file does not exist: {p}")
    size_bytes = p.stat().st_size
    if size_bytes <= 44:
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


# ==============================================================================
# 4. DURATION, PITCH & ENERGY PLOTTING UTILITIES
# ==============================================================================


def _select_telugu_font() -> str:
    """Return the highest-priority installed Telugu-capable font name (with safe fallback)."""
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in TELUGU_FONT_CANDIDATES:
        if candidate in available:
            return candidate
    return "DejaVu Sans"


def generate_prosody_plots(
    sample_id: str,
    tokens: Sequence[str],
    durations: np.ndarray | Sequence[int],
    pitch_hz: np.ndarray | Sequence[float],
    energy_raw: np.ndarray | Sequence[float],
    time_seconds: np.ndarray | Sequence[float],
    plots_dir: str | Path = DEFAULT_PLOTS_DIR,
    pitch_mean_hz: float = TELUGU_PITCH_MEAN_HZ,
    energy_mean: float = TELUGU_ENERGY_MEAN,
) -> dict[str, str]:
    """Generate and save duration, pitch (F0), and energy plots for a synthesized Telugu utterance.

    Saves:
        - <plots_dir>/<sample_id>_duration.png
        - <plots_dir>/<sample_id>_pitch.png
        - <plots_dir>/<sample_id>_energy.png

    Returns:
        Dictionary mapping 'duration_plot', 'pitch_plot', 'energy_plot' to their file paths.
    """
    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dur_arr = np.asarray(durations, dtype=np.int64)
    p_hz = np.asarray(pitch_hz, dtype=np.float64)
    e_raw = np.asarray(energy_raw, dtype=np.float64)
    t_sec = np.asarray(time_seconds, dtype=np.float64)

    if len(tokens) != len(dur_arr):
        raise ValueError(f"Token count ({len(tokens)}) != duration count ({len(dur_arr)}).")
    if len(p_hz) != len(e_raw) or len(p_hz) != len(t_sec):
        raise ValueError(
            f"Frame contour length mismatch: pitch={len(p_hz)}, energy={len(e_raw)}, time={len(t_sec)}."
        )

    chosen_font = _select_telugu_font()
    dur_path = out_dir / f"{sample_id}_duration.png"
    pitch_path = out_dir / f"{sample_id}_pitch.png"
    energy_path = out_dir / f"{sample_id}_energy.png"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        # 1. Duration Plot (<sample_id>_duration.png)
        fig_w = max(12.0, min(24.0, 0.25 * len(tokens) + 4.0))
        fig, ax = plt.subplots(figsize=(fig_w, 5.0), dpi=150)
        x_idx = np.arange(len(tokens))
        ax.bar(x_idx, dur_arr, color="#1f77b4", width=0.72)
        ax.set_title(f"{sample_id} - Grapheme Duration Prediction", fontsize=13)
        ax.set_xlabel("Grapheme / Pause Token", fontsize=11)
        ax.set_ylabel("Predicted duration (mel frames)", fontsize=11)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(
            list(tokens),
            fontfamily=chosen_font,
            fontsize=9,
            rotation=90 if len(tokens) > 25 else 0,
        )
        ax.grid(axis="y", linestyle="-", alpha=0.25)
        ax.set_xlim(-0.8, len(tokens) - 0.2)
        fig.tight_layout()
        fig.savefig(dur_path)
        plt.close(fig)

        # 2. Pitch / F0 Contour Plot (<sample_id>_pitch.png)
        fig, ax = plt.subplots(figsize=(11.0, 4.0), dpi=150)
        ax.plot(t_sec, p_hz, color="#d9534f", linewidth=2.0, label="Predicted Pitch (Hz)")
        ax.axhline(
            pitch_mean_hz,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label=f"Corpus Mean ({pitch_mean_hz:.1f} Hz)",
        )
        ax.set_title(f"Predicted Pitch Contour (F0) — Sample {sample_id}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Time (seconds)", fontsize=11)
        ax.set_ylabel("Pitch (Hz)", fontsize=11)
        if len(t_sec) > 0:
            ax.set_xlim(0.0, max(float(t_sec[-1]), 1e-3))
        y_min = max(0.0, min(float(np.min(p_hz)), pitch_mean_hz) - 20.0) if len(p_hz) > 0 else 0.0
        y_max = (max(float(np.max(p_hz)), pitch_mean_hz) + 25.0) if len(p_hz) > 0 else 350.0
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=10)
        fig.tight_layout()
        fig.savefig(pitch_path)
        plt.close(fig)

        # 3. Energy Contour Plot (<sample_id>_energy.png)
        fig, ax = plt.subplots(figsize=(11.0, 4.0), dpi=150)
        ax.plot(t_sec, e_raw, color="#6f42c1", linewidth=2.0, label="Predicted Energy")
        ax.axhline(
            energy_mean,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label=f"Corpus Mean ({energy_mean:.1f})",
        )
        ax.set_title(f"Predicted Energy Contour — Sample {sample_id}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Time (seconds)", fontsize=11)
        ax.set_ylabel("Linear STFT Magnitude L2-Norm", fontsize=11)
        if len(t_sec) > 0:
            ax.set_xlim(0.0, max(float(t_sec[-1]), 1e-3))
        e_min = max(0.0, min(float(np.min(e_raw)), 0.0)) if len(e_raw) > 0 else 0.0
        e_max = (max(float(np.max(e_raw)), energy_mean) + 10.0) if len(e_raw) > 0 else 80.0
        ax.set_ylim(e_min, e_max)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=10)
        fig.tight_layout()
        fig.savefig(energy_path)
        plt.close(fig)

    for p in (dur_path, pitch_path, energy_path):
        if not p.is_file() or p.stat().st_size <= 0:
            raise RuntimeError(f"Failed to generate non-empty plot file: {p}")

    return {
        "duration_plot": str(dur_path),
        "pitch_plot": str(pitch_path),
        "energy_plot": str(energy_path),
        "font_used": chosen_font,
    }


# ==============================================================================
# 5. SINGLE-SENTENCE SYNTHESIS & METADATA GENERATION
# ==============================================================================


def synthesize_sentence(
    model: FastSpeech2,
    vocoder: torch.nn.Module,
    text: str,
    speaker: int | str = 1,
    configs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None,
    control_values: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: torch.device | str = "cpu",
    checkpoint_metadata: dict[str, Any] | None = None,
    digit_mode: str = "telugu_digits",
    strict_no_unk: bool = False,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Synthesize speech waveform and prosody predictions for a single Telugu utterance.

    Args:
        model: Loaded Telugu FastSpeech2 model in eval mode.
        vocoder: Loaded HiFi-GAN universal Generator vocoder in eval mode.
        text: Raw Telugu sentence or curly-braced token sequence.
        speaker: Speaker ID (0 or 1), code ('spk_000' or 'spk_001'), or gender ('female' or 'male').
        configs: Optional (preprocess_config, model_config, train_config) dictionaries.
        control_values: (pitch_control, energy_control, duration_control).
        device: PyTorch execution device ('cpu', 'mps', or 'cuda').
        checkpoint_metadata: Optional metadata dict from load_telugu_fastspeech2.
        digit_mode: 'telugu_digits' or 'words'.
        strict_no_unk: If True, raises ValueError if any UNK token is produced.

    Returns:
        (wav_int16_array, sampling_rate, synthesis_metadata)
    """
    sr = EXPECTED_SAMPLING_RATE
    hop_length = EXPECTED_HOP_LENGTH
    stats_dict: dict[str, Any] | None = None
    if configs is not None:
        preprocess_dict, model_dict, _ = configs
        sr = int(
            preprocess_dict.get("preprocessing", {})
            .get("audio", {})
            .get("sampling_rate", EXPECTED_SAMPLING_RATE)
        )
        hop_length = int(
            preprocess_dict.get("preprocessing", {})
            .get("stft", {})
            .get("hop_length", EXPECTED_HOP_LENGTH)
        )
        stats_dict = model_dict.get("stats")

    if stats_dict is None and DEFAULT_STATS_PATH.is_file():
        stats_dict = load_telugu_stats(DEFAULT_STATS_PATH)

    p_ctrl, e_ctrl, d_ctrl = (
        float(control_values[0]),
        float(control_values[1]),
        float(control_values[2]),
    )

    # 1. Resolve and validate speaker (spk_000=0 female, spk_001=1 male)
    spk_info = resolve_telugu_speaker(speaker)
    speaker_id = spk_info["speaker_id"]
    if model.speaker_emb is not None and model.n_speaker is not None:
        if speaker_id < 0 or speaker_id >= model.n_speaker:
            raise ValueError(
                f"speaker_id {speaker_id} out of range for model.n_speaker={model.n_speaker}"
            )

    # 2. Telugu text normalization and word-boundary @sp tokenization
    seq_np, tokens, curly_tokens_str, norm_text, unk_report = preprocess_telugu_text(
        text,
        digit_mode=digit_mode,
        strict_no_unk=strict_no_unk,
    )
    src_len = int(len(seq_np))

    texts_t = torch.as_tensor(seq_np, dtype=torch.long, device=device).unsqueeze(0)  # [1, L_src]
    src_lens_t = torch.as_tensor([src_len], dtype=torch.long, device=device)  # [1]
    spk_t = (
        torch.as_tensor([speaker_id], dtype=torch.long, device=device)
        if model.speaker_emb is not None
        else None
    )

    # 3. Model forward pass (strictly inference: all ground-truth targets=None)
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
        # FastSpeech 2 10-tuple contract:
        # 0: mel_output [1, T_mel, 80]
        # 1: postnet_output [1, T_mel, 80]
        # 2: p_predictions [1, T_mel]
        # 3: e_predictions [1, T_mel]
        # 4: log_d_predictions [1, L_src]
        # 5: d_rounded [1, L_src]
        # 8: src_lens [1]
        # 9: mel_lens [1]
        postnet_mel = output[1]
        mel_len = int(output[9][0].item())
        pred_log_dur = output[4][0, :src_len].detach().cpu().numpy()
        pred_dur = output[5][0, :src_len].detach().cpu().numpy()
        pred_pitch_norm = output[2][0, :mel_len].detach().cpu().numpy()
        pred_energy_norm = output[3][0, :mel_len].detach().cpu().numpy()

        # 4. HiFi-GAN Vocoder inference: expects [1, 80, T_mel]
        mel_for_vocoder = postnet_mel[:, :mel_len, :].transpose(1, 2)
        if mel_for_vocoder.shape != (1, EXPECTED_N_MEL_CHANNELS, mel_len):
            raise RuntimeError(
                f"Unexpected mel_for_vocoder shape: {tuple(mel_for_vocoder.shape)}, "
                f"expected (1, {EXPECTED_N_MEL_CHANNELS}, {mel_len})."
            )
        wav_out = vocoder(mel_for_vocoder)
        wav_t = wav_out.squeeze(1)[0] if wav_out.dim() == 3 else wav_out.squeeze()
        wav_raw = wav_t.detach().cpu().numpy()

    if wav_raw.ndim != 1 or len(wav_raw) == 0:
        raise RuntimeError(f"Invalid vocoder waveform output shape: {wav_raw.shape}")
    if not np.isfinite(wav_raw).all():
        raise ValueError("Vocoder produced non-finite (NaN or Inf) audio samples.")

    # 5. Peak normalize if amplitude exceeds [-1, 1] and convert to 16-bit PCM
    max_amp = float(np.max(np.abs(wav_raw)))
    if max_amp > 1.0:
        wav_raw = wav_raw / max_amp
    wav_int16 = (wav_raw * 32767.0).astype(np.int16)

    # 6. Inverse-normalize pitch (Hz) and energy (STFT L2-norm)
    inv = inverse_normalize_prosody(
        pitch_norm=pred_pitch_norm,
        energy_norm=pred_energy_norm,
        stats=stats_dict,
        hop_length=hop_length,
        sampling_rate=sr,
    )
    pitch_hz = np.asarray(inv["pitch_hz"], dtype=np.float64)
    energy_raw = np.asarray(inv["energy_raw"], dtype=np.float64)
    time_seconds = np.asarray(inv["time_seconds"], dtype=np.float64)

    duration_sec = round(float(len(wav_int16)) / float(sr), 4)
    metadata: dict[str, Any] = {
        "original_text": text,
        "raw_text": text,
        "normalized_text": norm_text,
        "tokens": tokens,
        "phonemes": curly_tokens_str,
        "phoneme_tokens": src_len,
        "unk_count": len(unk_report),
        "unk_tokens": unk_report,
        "mel_frames": mel_len,
        "audio_samples": int(len(wav_int16)),
        "sampling_rate": sr,
        "duration_seconds": duration_sec,
        "speaker_id": speaker_id,
        "speaker_code": spk_info["speaker_code"],
        "speaker_gender": spk_info["gender"],
        "checkpoint": checkpoint_metadata
        or {
            "checkpoint_path": None,
            "global_step": None,
            "epoch": None,
            "best_val_loss": None,
        },
        "controls": {"pitch": p_ctrl, "energy": e_ctrl, "duration": d_ctrl},
        "inference_settings": {
            "sampling_rate": sr,
            "hop_length": hop_length,
            "filter_length": EXPECTED_FFT_SIZE,
            "win_length": EXPECTED_FFT_SIZE,
            "n_mel_channels": EXPECTED_N_MEL_CHANNELS,
            "vocoder": "HiFi-GAN universal",
            "device": str(device),
            "digit_mode": digit_mode,
        },
        "normalization_stats": {
            "pitch_mean_hz": float(inv["pitch_mean_hz"]),
            "pitch_std_hz": float(inv["pitch_std_hz"]),
            "energy_mean": float(inv["energy_mean"]),
            "energy_std": float(inv["energy_std"]),
        },
        "variance_predictions": {
            "log_duration": [float(x) for x in pred_log_dur.tolist()],
            "duration": [int(x) for x in pred_dur.tolist()],
            "pitch": [float(x) for x in pred_pitch_norm.tolist()],
            "pitch_hz": [float(x) for x in pitch_hz.tolist()],
            "energy": [float(x) for x in pred_energy_norm.tolist()],
            "energy_raw": [float(x) for x in energy_raw.tolist()],
            "time_seconds": [ round(float(x), 6) for x in time_seconds.tolist() ],
            "pitch_mean": float(np.mean(pred_pitch_norm)) if len(pred_pitch_norm) > 0 else 0.0,
            "pitch_mean_hz": float(np.mean(pitch_hz)) if len(pitch_hz) > 0 else 0.0,
            "energy_mean": float(np.mean(pred_energy_norm)) if len(pred_energy_norm) > 0 else 0.0,
            "energy_mean_raw": float(np.mean(energy_raw)) if len(energy_raw) > 0 else 0.0,
        },
    }
    return wav_int16, sr, metadata


# ==============================================================================
# 6. CLI & CHECKPOINT PATH RESOLUTION
# ==============================================================================


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_checkpoint_path(args: argparse.Namespace, train_config: dict[str, Any]) -> Path:
    """Resolve Telugu FastSpeech 2 checkpoint path from CLI or default candidate directories."""
    if getattr(args, "checkpoint", None):
        p = _resolve_file_path(args.checkpoint)
        if not p.is_file():
            raise FileNotFoundError(f"Specified checkpoint not found: {p}")
        return p

    raw_ckpt_base = Path(train_config.get("path", {}).get("ckpt_path", "./output/ckpt/IndicTTS"))
    ckpt_base = raw_ckpt_base if raw_ckpt_base.is_dir() else resolve_project_path(raw_ckpt_base)

    candidate_dirs = [
        raw_ckpt_base,
        ckpt_base,
        *get_checkpoint_candidate_dirs("telugu"),
    ]

    restore_step = getattr(args, "restore_step", None)
    if restore_step is not None:
        for cdir in candidate_dirs:
            p1 = cdir / f"checkpoint_{restore_step}.pth.tar"
            p2 = cdir / f"{restore_step}.pth.tar"
            if p1.is_file():
                return p1
            if p2.is_file():
                return p2
        raise FileNotFoundError(
            f"Checkpoint for step {restore_step} not found in candidate directories: {candidate_dirs}"
        )

    # Require checkpoint_best.pth.tar strictly (do not substitute checkpoint_latest.pth.tar)
    for cdir in candidate_dirs:
        best_p = cdir / "checkpoint_best.pth.tar"
        if best_p.is_file():
            return best_p

    expected_local_path = _PROJECT_ROOT / "output" / "ckpt" / "IndicTTS" / "checkpoint_best.pth.tar"
    raise FileNotFoundError(
        f"Required Telugu best checkpoint not found at: {expected_local_path}\n"
        f"(Searched candidate directories: {[str(d) for d in candidate_dirs]})\n"
        "Do not substitute another checkpoint. Place 'checkpoint_best.pth.tar' (step 13000) "
        "in 'output/ckpt/IndicTTS/checkpoint_best.pth.tar' or pass --checkpoint <path>."
    )


def audit_assignment_samples(
    samples: dict[str, str] | None = None,
    digit_mode: str = "telugu_digits",
) -> dict[str, dict[str, Any]]:
    """Run pre-synthesis tokenizer audit on all canonical Telugu samples.

    Verifies:
        1. Zero UNK tokens (ID 1).
        2. Zero untrained ordinary-space or punctuation token IDs (7..18).
        3. All word boundaries and clause pauses map to @sp (ID 4).
        4. No meaningful Telugu characters (vowels, consonants, matras, virama, modifiers, digits) are lost.
    """
    target_samples = samples if samples is not None else ASSIGNMENT_SAMPLES
    audit_results: dict[str, dict[str, Any]] = {}
    for sample_id, raw_text in target_samples.items():
        seq_np, tokens, curly_str, norm_text, unk_report = preprocess_telugu_text(
            raw_text, digit_mode=digit_mode, strict_no_unk=True
        )
        ids = [int(x) for x in seq_np.tolist()]
        untrained_7_18 = [tid for tid in ids if tid in UNTRAINED_PUNCT_AND_SPACE_IDS]
        if untrained_7_18:
            raise ValueError(
                f"Sample {sample_id} emitted untrained punctuation/space token IDs: {untrained_7_18}"
            )
        reconstructed = sequence_to_text(ids)
        # Compare against expected Telugu characters (with ASCII digits converted to Telugu numerals)
        expected_chars = "".join(
            _ASCII_TO_TELUGU_DIGIT.get(ch, ch)
            for ch in norm_text
            if ch not in _CLAUSE_PAUSE_CHARS and ch not in _IGNORED_QUOTE_BRACKET_CHARS
        )
        if reconstructed != expected_chars:
            raise ValueError(
                f"Meaningful Telugu character loss detected in {sample_id}:\n"
                f"  Expected     : {expected_chars}\n"
                f"  Reconstructed: {reconstructed}"
            )
        audit_results[sample_id] = {
            "sample_id": sample_id,
            "original_text": raw_text,
            "normalized_text": norm_text,
            "num_tokens": len(tokens),
            "sp_count": tokens.count("@sp"),
            "curly_tokens": curly_str,
            "token_ids": ids,
            "reconstructed_graphemes": reconstructed,
            "unk_count": len(unk_report),
            "unk_tokens": unk_report,
            "untrained_ids_7_to_18": untrained_7_18,
            "character_preservation_verified": True,
        }
    return audit_results


def load_assignment_samples(samples_file: str | Path | None = None) -> dict[str, str]:
    """Load canonical Telugu assignment test sentences from ASSIGNMENT_SAMPLES or a JSON/TSV file."""
    if samples_file is not None:
        sf_path = _resolve_file_path(samples_file)
        if not sf_path.is_file():
            raise FileNotFoundError(f"Samples file not found: {sf_path}")
        if sf_path.suffix.lower() == ".json":
            with open(sf_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
            if isinstance(data, list):
                out: dict[str, str] = {}
                for idx, item in enumerate(data, 1):
                    if isinstance(item, dict) and "sample_id" in item and "text" in item:
                        out[str(item["sample_id"])] = str(item["text"])
                    elif isinstance(item, str):
                        out[f"T{idx}"] = item
                return out
        else:
            out = {}
            with open(sf_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f, 1):
                    line_s = line.strip()
                    if not line_s:
                        continue
                    parts = line_s.split("|")
                    if len(parts) >= 2:
                        out[parts[0].strip()] = parts[-1].strip()
                    else:
                        out[f"T{idx}"] = line_s
            return out

    if ASSIGNMENT_SAMPLES:
        return dict(ASSIGNMENT_SAMPLES)

    raise ValueError(
        "Canonical Telugu assignment test sentences have not been supplied yet.\n"
        "Pass --samples_file <path_to_samples.json_or_txt>, --source <manifest>, or --mode single --text '<telugu_text>'."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference and prosody visualization for Telugu FastSpeech 2 (IndicTTS)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "batch", "assignment"],
        default="single",
        help="Inference mode: 'single' (one text), 'batch' (from source file), or 'assignment'.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Raw Telugu sentence string to synthesize (required for mode='single').",
    )
    parser.add_argument(
        "--sample",
        type=str,
        choices=["T1", "T2_a", "T2(a)", "T2a", "T2_b", "T2(b)", "T2b", "T3", "T4", "all"],
        default=None,
        help="Synthesize specific canonical sample ID (T1, T2_a, T2_b, T3, T4, or 'all').",
    )
    parser.add_argument(
        "--samples_file",
        type=str,
        default=None,
        help="Optional JSON or pipe-delimited file defining canonical Telugu assignment samples.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to manifest file for batch mode (lines: id|speaker|tokens|raw_text or id|text).",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=1,
        choices=[0, 1],
        help="Telugu speaker ID: 0 (spk_000, female) or 1 (spk_001, male). Default: 1 (spk_001, male).",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        help="Optional speaker code/label override ('spk_000', 'spk_001', 'female', 'male').",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit path to Telugu FastSpeech 2 checkpoint (.pth.tar).",
    )
    parser.add_argument(
        "--restore_step",
        type=int,
        default=None,
        help="Checkpoint step number to restore (looks in output/ckpt/IndicTTS).",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        default=str(DEFAULT_PREPROCESS_CONFIG),
        help="Path to config/IndicTTS/preprocess.yaml",
    )
    parser.add_argument(
        "-m",
        "--model_config",
        type=str,
        default=str(DEFAULT_MODEL_CONFIG),
        help="Path to config/IndicTTS/model.yaml",
    )
    parser.add_argument(
        "-t",
        "--train_config",
        type=str,
        default=str(DEFAULT_TRAIN_CONFIG),
        help="Path to config/IndicTTS/train.yaml",
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
        "--digit_mode",
        type=str,
        choices=["telugu_digits", "words"],
        default="telugu_digits",
        help="How to handle digits in raw text ('telugu_digits' or 'words').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_AUDIO_DIR),
        help="Destination directory for output WAV files (default: Results/Telugu/audio).",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=str(DEFAULT_PLOTS_DIR),
        help="Destination directory for duration/pitch/energy plots (default: Results/Telugu/plots).",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default=str(DEFAULT_METADATA_DIR),
        help="Destination directory for JSON metadata (default: Results/Telugu/metadata).",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Filename or sample ID for single synthesis output.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cuda', 'mps', or 'cpu'). Default: auto-detect.",
    )
    return parser.parse_args(argv)


def run_inference(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Execute the complete Telugu FastSpeech 2 synthesis, plotting, and metadata export pipeline."""
    print("=" * 72)
    print("FASTSPEECH 2 TELUGU (INDICTTS) INFERENCE PIPELINE")
    print("=" * 72)
    device = resolve_device(args.device)
    print(f"Compute Device       : {device}")

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

    # Pre-synthesis tokenizer audit for canonical assignment samples
    if args.sample or args.mode == "assignment":
        samples_map = load_assignment_samples(args.samples_file)
        audit_res = audit_assignment_samples(samples_map, digit_mode=args.digit_mode)
        print(
            f"Tokenizer Audit      : PASSED ({len(audit_res)} canonical samples, "
            "0 UNK tokens, 0 untrained IDs 7..18)"
        )

    ckpt_path = resolve_checkpoint_path(args, train_cfg)
    print(f"Loading FastSpeech 2 : {ckpt_path}")
    model, ckpt_meta = load_telugu_fastspeech2(ckpt_path, configs, device=device)
    if ckpt_meta["best_val_loss"] is not None:
        print(
            f"Checkpoint Metadata  : Step {ckpt_meta['global_step']} | "
            f"Epoch {ckpt_meta['epoch']} | Best Val Loss {ckpt_meta['best_val_loss']:.4f}"
        )

    print(f"Loading HiFi-GAN Voc : {args.vocoder_weights}")
    vocoder = load_hifigan_vocoder(
        config_path=args.vocoder_config,
        weights_path=args.vocoder_weights,
        device=device,
    )

    speaker_target: int | str = args.speaker if args.speaker is not None else args.speaker_id
    default_spk = resolve_telugu_speaker(speaker_target)
    print(
        f"Active Speaker       : {default_spk['speaker_code']} "
        f"(ID={default_spk['speaker_id']}, {default_spk['gender']})"
    )

    audio_dir = ensure_output_dir(resolve_project_path(args.output_dir))
    plots_dir = ensure_output_dir(resolve_project_path(args.plots_dir))
    metadata_dir = ensure_output_dir(resolve_project_path(args.metadata_dir))

    control_vals = (args.pitch_control, args.energy_control, args.duration_control)

    # Prepare tasks: list of (sample_id, raw_text, item_speaker)
    tasks: list[tuple[str, str, int | str]] = []
    if args.sample:
        samples_map = load_assignment_samples(args.samples_file)
        if args.sample == "all":
            tasks = [(sid, txt, speaker_target) for sid, txt in samples_map.items()]
        else:
            canonical_sid = SAMPLE_ID_ALIASES.get(args.sample, args.sample)
            if canonical_sid not in samples_map:
                raise KeyError(
                    f"Sample {args.sample!r} not found in assignment samples: {list(samples_map.keys())}"
                )
            tasks = [(canonical_sid, samples_map[canonical_sid], speaker_target)]
    elif args.mode == "assignment":
        samples_map = load_assignment_samples(args.samples_file)
        tasks = [(sid, txt, speaker_target) for sid, txt in samples_map.items()]
    elif args.mode == "single":
        if not args.text:
            raise ValueError("--text is required when --mode=single and --sample is not specified.")
        fname = args.output_filename or f"telugu_synth_{int(time.time())}.wav"
        sample_id = Path(fname).stem
        tasks = [(sample_id, args.text, speaker_target)]
    elif args.mode == "batch":
        source_path = _resolve_file_path(args.source) if args.source else None
        if not source_path or not source_path.is_file():
            raise FileNotFoundError(f"Batch source file not found: {args.source}")
        with open(source_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line_s = line.strip()
                if not line_s:
                    continue
                parts = line_s.split("|")
                if len(parts) >= 4:
                    uid, spk_col, _, raw = parts[0], parts[1], parts[2], parts[3]
                    item_spk: int | str = spk_col if args.speaker is None else speaker_target
                elif len(parts) == 3:
                    uid, spk_col, raw = parts[0], parts[1], parts[2]
                    item_spk = spk_col if args.speaker is None else speaker_target
                elif len(parts) == 2:
                    uid, raw = parts[0], parts[1]
                    item_spk = speaker_target
                else:
                    uid, raw = f"T_{idx + 1:02d}", parts[0]
                    item_spk = speaker_target
                tasks.append((uid, raw, item_spk))

    print(f"\nSynthesizing {len(tasks)} Telugu utterance(s):")
    print(f"  Audio Dir    : {audio_dir}")
    print(f"  Plots Dir    : {plots_dir}")
    print(f"  Metadata Dir : {metadata_dir}")
    print("-" * 72)

    all_prosody_metadata: list[dict[str, Any]] = []
    all_samples_summary: list[dict[str, Any]] = []
    t_start_all = time.perf_counter()

    for item_idx, (sample_id, raw_text, item_spk) in enumerate(tasks, 1):
        t0 = time.perf_counter()
        print(f"\n[{item_idx}/{len(tasks)}] Processing: {sample_id}")
        print(f"  Text: {raw_text}")

        wav_arr, sr, meta = synthesize_sentence(
            model=model,
            vocoder=vocoder,
            text=raw_text,
            speaker=item_spk,
            configs=configs,
            control_values=control_vals,
            device=device,
            checkpoint_metadata=ckpt_meta,
            digit_mode=args.digit_mode,
        )

        # 1. Save & verify WAV audio
        wav_path = audio_dir / f"{sample_id}.wav"
        wavfile.write(wav_path, sr, wav_arr)
        wav_audit = verify_wav_file(wav_path, expected_sr=sr)

        # 2. Generate duration, pitch (Hz), and energy plots
        plot_paths = generate_prosody_plots(
            sample_id=sample_id,
            tokens=meta["tokens"],
            durations=meta["variance_predictions"]["duration"],
            pitch_hz=meta["variance_predictions"]["pitch_hz"],
            energy_raw=meta["variance_predictions"]["energy_raw"],
            time_seconds=meta["variance_predictions"]["time_seconds"],
            plots_dir=plots_dir,
            pitch_mean_hz=meta["normalization_stats"]["pitch_mean_hz"],
            energy_mean=meta["normalization_stats"]["energy_mean"],
        )

        # 3. Save per-utterance metadata JSON
        utt_meta_path = metadata_dir / f"{sample_id}_metadata.json"
        elapsed = round(time.perf_counter() - t0, 3)
        meta["sample_id"] = sample_id
        meta["elapsed_sec"] = elapsed
        meta["output_files"] = {
            "audio_wav": str(wav_path),
            "duration_plot": plot_paths["duration_plot"],
            "pitch_plot": plot_paths["pitch_plot"],
            "energy_plot": plot_paths["energy_plot"],
            "metadata_json": str(utt_meta_path),
        }
        with open(utt_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        all_prosody_metadata.append(meta)
        all_samples_summary.append(
            {
                "sample_id": sample_id,
                "original_text": raw_text,
                "normalized_text": meta["normalized_text"],
                "speaker_id": meta["speaker_id"],
                "speaker_code": meta["speaker_code"],
                "speaker_gender": meta["speaker_gender"],
                "checkpoint_step": ckpt_meta["global_step"],
                "audio_file": wav_path.name,
                "sample_rate": sr,
                "duration_seconds": wav_audit["duration_seconds"],
                "num_tokens": meta["phoneme_tokens"],
                "mel_frames": meta["mel_frames"],
                "unk_count": meta["unk_count"],
                "plots": {
                    "duration": Path(plot_paths["duration_plot"]).name,
                    "pitch": Path(plot_paths["pitch_plot"]).name,
                    "energy": Path(plot_paths["energy_plot"]).name,
                },
            }
        )
        print(
            f"  Saved Audio  -> {wav_path} ({wav_audit['duration_seconds']}s, "
            f"{wav_audit['file_size_bytes']:,} bytes, {elapsed:.2f}s)"
        )
        print(
            f"  Saved Plots  -> {Path(plot_paths['duration_plot']).name}, "
            f"{Path(plot_paths['pitch_plot']).name}, {Path(plot_paths['energy_plot']).name}"
        )

    # Save aggregate metadata files in Results/Telugu/metadata/
    prosody_json_path = metadata_dir / "telugu_prosody_metadata.json"
    samples_json_path = metadata_dir / "telugu_samples_metadata.json"
    with open(prosody_json_path, "w", encoding="utf-8") as f:
        json.dump(all_prosody_metadata, f, ensure_ascii=False, indent=2)
    with open(samples_json_path, "w", encoding="utf-8") as f:
        json.dump(all_samples_summary, f, ensure_ascii=False, indent=2)

    total_sec = time.perf_counter() - t_start_all
    print("\n" + "=" * 72)
    print(f"Telugu Inference Completed! Total time: {total_sec:.2f}s ({len(all_prosody_metadata)} utterance(s))")
    print("=" * 72)
    return all_prosody_metadata


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()

