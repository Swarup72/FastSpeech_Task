"""Reproducible, resumable acoustic feature extraction for Telugu Option 1B FastSpeech 2.

Extracts 80-bin log-mel spectrograms, frame-level and phoneme-level fundamental
frequency (F0 via PyWorld DIO + Stonemask with linear interpolation across unvoiced
frames), frame-level and phoneme-level STFT L2-norm energy, and telescoping-reconciled
phoneme durations from verified IndicMFA TextGrid alignments.

Strict guarantees:
1. Processes ONLY Option 1B `train.txt` (2,302 entries -> 2,300 aligned) and
   `val.txt` (422 entries -> 421 aligned).
2. Strictly excludes the 3 failed upstream-truncated recordings:
   - `train_telugufemale_00930`
   - `train_telugufemale_00928`
   - `train_telugumale_02390`
3. NEVER reads, accesses, or processes `test.txt` or its audio.
4. Fits pitch and energy z-score normalization statistics (`stats.json`) strictly
   on training data (`2,300` utterances) after IQR outlier filtering.
5. Resumable execution: validates existing `.npy` feature files on disk and skips
   valid existing outputs without unnecessary overwriting.
6. Validates every generated array for finite values, positive durations,
   `sum(duration) == mel_frames == len(pitch) == len(energy)`, and
   `len(duration) == len(phoneme_sequence)`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import librosa
import numpy as np
import pyarrow.parquet as pq
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import soundfile as sf
import tgt
import torch

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from common.paths import (
    DATA_DIR,
    PREPROCESSED_DIR,
    PROJECT_ROOT,
    TELUGU_DIR,
    setup_project_paths,
)

setup_project_paths()

from English.alignment.duration import (
    PhoneInterval,
    compute_frame_durations_telescoping,
    fill_unannotated_gaps,
    reconcile_durations_with_mel,
    validate_intervals,
)
from Telugu.alignment.resample import resample_audio
from Telugu.text.normalizer import normalize_text, text_to_sequence
from Telugu.text.symbols import UNK_ID, VOCAB_SIZE, symbol_to_id

OPTION_1B_SPLIT_DIR = TELUGU_DIR / "metadata" / "splits" / "option_1b"
TEXTGRID_ROOT_DIR = DATA_DIR / "telugu_mfa_option_1b" / "TextGrid"
DEFAULT_OUT_DIR = PREPROCESSED_DIR / "Telugu_Option_1B"
PARQUET_GLOB = str(
    DATA_DIR / "telugu_parquet" / "data" / "train-*-of-00004.parquet"
)
TEXT_MANIFEST_PATH = TELUGU_DIR / "metadata" / "text_manifest.json"

# FastSpeech 2 & HiFi-GAN standard acoustic parameters
SAMPLING_RATE = 22050
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 256
N_MEL = 80
MEL_FMIN = 0.0
MEL_FMAX = 8000.0

# Silence labels produced by MFA / Kaldi TextGrids
TELUGU_SILENCE_LABELS: frozenset[str] = frozenset(
    {"", "sil", "sp", "spn", "<eps>", "@sp", "@sil", "@spn"}
)

# Strictly excluded upstream-truncated failed alignment recordings
EXCLUDED_FAILED_IDS: frozenset[str] = frozenset(
    {
        "train_telugufemale_00930",
        "train_telugufemale_00928",
        "train_telugumale_02390",
    }
)

SPEAKERS_MAP: dict[str, int] = {
    "spk_000": 0,  # Telugu Female
    "spk_001": 1,  # Telugu Male
}


@dataclass
class TeluguDurationResult:
    """Extracted Telugu grapheme/pause tokens and mel-frame durations from a TextGrid."""

    tokens: list[str]
    durations: list[int]
    start_time: float
    end_time: float
    total_frames: int
    mel_length: int | None
    reconciliation_applied: bool
    discrepancy: int
    fastspeech2_text: str


def remove_outliers(values: np.ndarray) -> np.ndarray:
    """Filter out extreme values using the IQR rule on positive values before fitting scalers."""
    v = np.asarray(values, dtype=np.float64)
    v_nonzero = v[v > 0]
    if len(v_nonzero) < 4:
        return v
    p25 = np.percentile(v_nonzero, 25)
    p75 = np.percentile(v_nonzero, 75)
    iqr = p75 - p25
    lower = p25 - 1.5 * iqr
    upper = p75 + 1.5 * iqr
    return v[(v >= lower) & (v <= upper)]


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via a temporary file rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    """Write a NumPy `.npy` file atomically via a temporary file rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        np.save(f, arr)
    os.replace(tmp_path, path)


def extract_telugu_durations_from_textgrid(
    textgrid_path: str | Path,
    target_mel_length: int | None = None,
    sampling_rate: int = SAMPLING_RATE,
    hop_length: int = HOP_LENGTH,
    min_phone_duration: int = 1,
) -> TeluguDurationResult:
    """Extract Telugu grapheme/pause tokens and frame durations from an IndicMFA TextGrid.

    1. Reads the `phones` tier from the TextGrid.
    2. Validates interval monotonicity and non-negativity.
    3. Fills any unannotated inter-word gaps (> 1e-4 s) with `@sp`.
    4. Trims leading and trailing silences, normalizing internal pauses to `@sp`.
    5. Validates that every token is present in the 101-symbol Telugu vocabulary
       and that no `<unk>` or `spn` token is present.
    6. Computes frame durations via telescoping boundary discretization (`min_phone_duration=1`).
    7. Reconciles durations against `target_mel_length` if provided.
    """
    tg = tgt.io.read_textgrid(str(textgrid_path), include_empty_intervals=True)
    if "phones" not in tg.get_tier_names():
        raise ValueError(
            f"TextGrid {textgrid_path} is missing 'phones' tier. Found: {tg.get_tier_names()}"
        )
    tier = tg.get_tier_by_name("phones")

    raw_intervals: list[PhoneInterval] = []
    for inv in tier.intervals:
        lbl = inv.text.strip()
        if lbl in ("spn", "<unk>", "@spn"):
            raise ValueError(
                f"TextGrid {textgrid_path} contains unknown spoken noise token: {lbl!r}"
            )
        raw_intervals.append(
            PhoneInterval(
                text=lbl,
                start_time=float(inv.start_time),
                end_time=float(inv.end_time),
            )
        )

    validate_intervals(raw_intervals)
    filled_intervals = fill_unannotated_gaps(raw_intervals, gap_label="@sp")

    first_non_sil = -1
    last_non_sil = -1
    for idx, inv in enumerate(filled_intervals):
        if inv.text not in TELUGU_SILENCE_LABELS and inv.duration_seconds > 0:
            if first_non_sil == -1:
                first_non_sil = idx
            last_non_sil = idx

    if first_non_sil == -1:
        raise ValueError(f"TextGrid {textgrid_path} contains no non-silence intervals.")

    start_time = filled_intervals[first_non_sil].start_time
    end_time = filled_intervals[last_non_sil].end_time

    trimmed_intervals: list[PhoneInterval] = []
    tokens: list[str] = []
    for inv in filled_intervals[first_non_sil : last_non_sil + 1]:
        if inv.duration_seconds <= 0:
            continue
        is_sil = inv.text in TELUGU_SILENCE_LABELS
        tok = "@sp" if is_sil else inv.text
        if tok not in symbol_to_id or symbol_to_id[tok] == UNK_ID:
            raise ValueError(
                f"Out-of-vocabulary token {tok!r} in TextGrid {textgrid_path}"
            )
        # Use 'sp' in PhoneInterval.text so PhoneInterval.is_silence is True for pause priority
        trimmed_intervals.append(
            PhoneInterval(
                text="sp" if is_sil else tok,
                start_time=inv.start_time,
                end_time=inv.end_time,
            )
        )
        tokens.append(tok)

    # Discretize boundaries via telescoping rounding; enforce >= 1 frame for all kept intervals
    durations: list[int] = []
    for inv in trimmed_intervals:
        b_start = int(
            np.round((inv.start_time - start_time) * sampling_rate / float(hop_length))
        )
        b_end = int(
            np.round((inv.end_time - start_time) * sampling_rate / float(hop_length))
        )
        d = max(min_phone_duration, b_end - b_start)
        durations.append(int(d))

    reconciliation_applied = False
    discrepancy = 0
    if target_mel_length is not None:
        durations, discrepancy = reconcile_durations_with_mel(
            durations=durations,
            target_length=target_mel_length,
            intervals=trimmed_intervals,
            start_offset=start_time,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
        )
        reconciliation_applied = discrepancy != 0

    fastspeech2_text = "{" + " ".join(tokens) + "}"
    return TeluguDurationResult(
        tokens=tokens,
        durations=durations,
        start_time=start_time,
        end_time=end_time,
        total_frames=int(sum(durations)),
        mel_length=target_mel_length,
        reconciliation_applied=reconciliation_applied,
        discrepancy=int(discrepancy),
        fastspeech2_text=fastspeech2_text,
    )


class TeluguFeatureExtractor:
    """Extracts 80-bin mel, F0 pitch, energy, and reconciled durations for Telugu utterances."""

    def __init__(
        self,
        sample_rate: int = SAMPLING_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        win_length: int = WIN_LENGTH,
        n_mels: int = N_MEL,
        fmin: float = MEL_FMIN,
        fmax: float = MEL_FMAX,
    ) -> None:
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop = hop_length
        self.win = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        self.mel_basis = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        self.window = torch.hann_window(self.win)

    def extract_from_waveform_and_textgrid(
        self,
        audio_22k: np.ndarray,
        tg_path: str | Path,
    ) -> dict[str, Any]:
        """Extract acoustic features from a 22,050 Hz mono waveform and its IndicMFA TextGrid."""
        if not os.path.exists(tg_path):
            raise FileNotFoundError(f"TextGrid not found: {tg_path}")

        if audio_22k.ndim > 1:
            audio_22k = audio_22k.mean(axis=1)
        audio_22k = np.asarray(audio_22k, dtype=np.float32)

        # 1. Initial TextGrid pass to obtain silence-trimmed start and end timestamps
        res_init = extract_telugu_durations_from_textgrid(
            tg_path,
            target_mel_length=None,
            sampling_rate=self.sr,
            hop_length=self.hop,
        )
        if not res_init.tokens or res_init.start_time >= res_init.end_time:
            raise ValueError(
                f"Invalid trimmed span [{res_init.start_time}, {res_init.end_time}] in {tg_path}"
            )

        # 2. Slice waveform to trimmed boundaries
        s_sample = max(0, int(self.sr * res_init.start_time))
        e_sample = min(len(audio_22k), int(self.sr * res_init.end_time))
        audio_trimmed = audio_22k[s_sample:e_sample]

        if len(audio_trimmed) < self.n_fft:
            raise ValueError(
                f"Trimmed audio length ({len(audio_trimmed)} samples) < n_fft ({self.n_fft}) in {tg_path}"
            )

        # 3. Compute STFT, 80-bin Log-Mel Spectrogram, and Frame-Level Energy
        wav_tensor = torch.from_numpy(audio_trimmed).float()
        wav_padded = torch.nn.functional.pad(
            wav_tensor.unsqueeze(0),
            (self.n_fft // 2, self.n_fft // 2),
            mode="reflect",
        ).squeeze(0)

        stft = torch.stft(
            wav_padded,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=self.window,
            center=False,
            return_complex=True,
        )
        mag = torch.abs(stft).numpy()  # [n_fft // 2 + 1, T_mel]
        energy = np.linalg.norm(mag, axis=0)  # [T_mel]
        mel = np.log(np.clip(self.mel_basis @ mag, a_min=1e-5, a_max=None))  # [80, T_mel]
        t_mel = int(mel.shape[1])

        # 4. Reconcile durations with exact mel frame length t_mel
        res = extract_telugu_durations_from_textgrid(
            tg_path,
            target_mel_length=t_mel,
            sampling_rate=self.sr,
            hop_length=self.hop,
        )
        if res.total_frames != t_mel:
            raise ValueError(
                f"Duration sum ({res.total_frames}) != mel frames ({t_mel}) after reconciliation for {tg_path}"
            )

        # 5. Extract Fundamental Frequency (F0) via PyWorld DIO + Stonemask
        _f0, t_f0 = pw.dio(
            audio_trimmed.astype(np.float64),
            self.sr,
            frame_period=self.hop / self.sr * 1000.0,
        )
        f0 = pw.stonemask(audio_trimmed.astype(np.float64), _f0, t_f0, self.sr)

        if len(f0) < t_mel:
            f0 = np.pad(f0, (0, t_mel - len(f0)), mode="edge")
        else:
            f0 = f0[:t_mel]

        # Linearly interpolate across unvoiced (f0 == 0) frames between voiced anchors
        nonzero_ids = np.where(f0 > 0)[0]
        voiced_frame_count = int(len(nonzero_ids))
        if len(nonzero_ids) > 1:
            interp_fn = interp1d(
                nonzero_ids,
                f0[nonzero_ids],
                fill_value=(f0[nonzero_ids[0]], f0[nonzero_ids[-1]]),
                bounds_error=False,
            )
            f0_interp = interp_fn(np.arange(t_mel))
        elif len(nonzero_ids) == 1:
            f0_interp = np.full(t_mel, f0[nonzero_ids[0]], dtype=np.float64)
        else:
            f0_interp = np.zeros(t_mel, dtype=np.float64)

        # 6. Compute phoneme-level averaged pitch and energy
        n_phones = len(res.durations)
        pitch_phoneme = np.zeros(n_phones, dtype=np.float32)
        energy_phoneme = np.zeros(n_phones, dtype=np.float32)
        pos = 0
        for i, d in enumerate(res.durations):
            if d > 0:
                pitch_phoneme[i] = float(np.mean(f0_interp[pos : pos + d]))
                energy_phoneme[i] = float(np.mean(energy[pos : pos + d]))
            pos += d

        mel_arr = mel.T.astype(np.float32)  # [T_mel, 80]
        pitch_frame_arr = f0_interp.astype(np.float32)  # [T_mel]
        energy_frame_arr = energy.astype(np.float32)  # [T_mel]
        durations_arr = np.asarray(res.durations, dtype=np.int32)  # [L_phone]

        for name, arr in (
            ("mel", mel_arr),
            ("pitch_frame", pitch_frame_arr),
            ("energy_frame", energy_frame_arr),
            ("pitch_phoneme", pitch_phoneme),
            ("energy_phoneme", energy_phoneme),
            ("durations", durations_arr),
        ):
            if not np.isfinite(arr).all():
                raise ValueError(f"Non-finite values (NaN/Inf) detected in {name} for {tg_path}")

        return {
            "mel": mel_arr,
            "pitch_frame": pitch_frame_arr,
            "energy_frame": energy_frame_arr,
            "raw_f0": f0.astype(np.float32),
            "voiced_frames": voiced_frame_count,
            "pitch_phoneme": pitch_phoneme,
            "energy_phoneme": energy_phoneme,
            "durations": durations_arr,
            "tokens": res.tokens,
            "fastspeech2_text": res.fastspeech2_text,
            "discrepancy": res.discrepancy,
            "reconciliation_applied": res.reconciliation_applied,
            "t_mel": t_mel,
            "trimmed_duration_sec": float((e_sample - s_sample) / float(self.sr)),
        }


def validate_utterance_feature_files(
    out_dir: Path,
    uid: str,
    spk_str: str,
    fastspeech2_text: str,
) -> tuple[bool, str | None, dict[str, Any] | None]:
    """Validate that all saved `.npy` feature files for an utterance exist and satisfy all invariants."""
    mel_path = out_dir / "mel" / f"{spk_str}-mel-{uid}.npy"
    pitch_path = out_dir / "pitch" / f"{spk_str}-pitch-{uid}.npy"
    energy_path = out_dir / "energy" / f"{spk_str}-energy-{uid}.npy"
    dur_path = out_dir / "duration" / f"{spk_str}-duration-{uid}.npy"
    p_ph_path = out_dir / "pitch_phoneme" / f"{spk_str}-pitch-{uid}.npy"
    e_ph_path = out_dir / "energy_phoneme" / f"{spk_str}-energy-{uid}.npy"

    for p in (mel_path, pitch_path, energy_path, dur_path, p_ph_path, e_ph_path):
        if not p.is_file():
            return False, f"missing_file:{p.name}", None

    try:
        mel = np.load(mel_path)
        pitch = np.load(pitch_path)
        energy = np.load(energy_path)
        dur = np.load(dur_path)
        p_ph = np.load(p_ph_path)
        e_ph = np.load(e_ph_path)
    except Exception as e:
        return False, f"npy_load_error:{e}", None

    for label, arr in (
        ("mel", mel),
        ("pitch", pitch),
        ("energy", energy),
        ("duration", dur),
        ("pitch_phoneme", p_ph),
        ("energy_phoneme", e_ph),
    ):
        if arr.size == 0:
            return False, f"empty_array:{label}", None
        if not np.isfinite(arr).all():
            return False, f"nan_or_inf_in_{label}", None

    if mel.ndim != 2 or mel.shape[1] != N_MEL:
        return False, f"invalid_mel_shape:{mel.shape}", None
    t_mel = int(mel.shape[0])

    if pitch.ndim != 1 or len(pitch) != t_mel:
        return False, f"pitch_len_mismatch:{len(pitch)}_vs_{t_mel}", None
    if energy.ndim != 1 or len(energy) != t_mel:
        return False, f"energy_len_mismatch:{len(energy)}_vs_{t_mel}", None
    if dur.ndim != 1:
        return False, f"invalid_duration_ndim:{dur.ndim}", None
    if not (dur >= 1).all():
        return False, f"non_positive_duration_min:{int(dur.min())}", None
    if int(np.sum(dur)) != t_mel:
        return False, f"duration_sum_mismatch:{int(np.sum(dur))}_vs_{t_mel}", None

    seq = text_to_sequence(fastspeech2_text)
    if len(seq) != len(dur):
        return False, f"phoneme_seq_len_mismatch:{len(seq)}_vs_{len(dur)}", None
    if any(tok_id <= 0 or tok_id >= VOCAB_SIZE or tok_id == UNK_ID for tok_id in seq):
        return False, "invalid_or_unk_token_id_in_sequence", None
    if p_ph.ndim != 1 or len(p_ph) != len(dur):
        return False, f"pitch_phoneme_len_mismatch:{len(p_ph)}_vs_{len(dur)}", None
    if e_ph.ndim != 1 or len(e_ph) != len(dur):
        return False, f"energy_phoneme_len_mismatch:{len(e_ph)}_vs_{len(dur)}", None

    meta = {
        "t_mel": t_mel,
        "n_phones": int(len(dur)),
        "pitch_min": float(pitch.min()),
        "pitch_max": float(pitch.max()),
        "energy_min": float(energy.min()),
        "energy_max": float(energy.max()),
    }
    return True, None, meta


def _load_option_1b_split_manifest(split_txt_path: Path, split_name: str) -> list[dict[str, Any]]:
    """Parse Option 1B `train.txt` or `val.txt` (`uid|speaker_id|raw|norm|dur|gender`)."""
    if split_name not in ("train", "val"):
        raise ValueError(
            f"Forbidden split_name {split_name!r}: only 'train' and 'val' may be loaded."
        )
    if "test" in split_txt_path.name.lower():
        raise ValueError(
            f"Refusing to access held-out test manifest: {split_txt_path}"
        )

    entries: list[dict[str, Any]] = []
    with open(split_txt_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line_s = line.strip()
            if not line_s:
                continue
            parts = line_s.split("|")
            if len(parts) != 6:
                raise ValueError(f"Malformed line {idx} in {split_txt_path}: {line_s}")
            uid, spk_id_str, raw_text, norm_text, dur_str, gender_str = parts
            spk_id = int(spk_id_str)
            spk_str = f"spk_{spk_id:03d}"
            if spk_str not in SPEAKERS_MAP:
                raise ValueError(f"Unknown speaker {spk_str} for {uid}")
            clean_text = normalize_text(raw_text)
            entries.append(
                {
                    "id": uid,
                    "split": split_name,
                    "speaker_id": spk_id,
                    "speaker_str": spk_str,
                    "gender": gender_str,
                    "raw_text": raw_text,
                    "normalized_text": clean_text,
                    "manifest_norm_text": norm_text,
                    "manifest_duration_sec": float(dur_str),
                }
            )
    return entries


def _load_parquet_audio_for_ids(
    target_ids: set[str],
    forbidden_ids: set[str] | None = None,
) -> dict[str, bytes]:
    """Load raw WAV bytes from the 4 Parquet shards for `target_ids` only."""
    if not target_ids:
        return {}
    if forbidden_ids and (target_ids & forbidden_ids):
        overlap = sorted(target_ids & forbidden_ids)
        raise ValueError(
            f"Attempted to load audio for forbidden/excluded IDs: {overlap}"
        )

    with open(TEXT_MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    uid_to_row_idx = {
        os.path.splitext(r["path"])[0]: int(r["row_idx"]) for r in manifest
    }
    shard_paths = sorted(glob.glob(PARQUET_GLOB))
    shards = [pq.ParquetFile(p) for p in shard_paths]

    rg_groups: dict[tuple[int, int], list[tuple[str, int]]] = {}
    for uid in target_ids:
        if uid not in uid_to_row_idx:
            raise KeyError(f"Utterance ID {uid} not found in {TEXT_MANIFEST_PATH}")
        r_idx = uid_to_row_idx[uid]
        s_idx = r_idx // 2144
        l_idx = r_idx % 2144
        rg_idx = l_idx // 100
        rg_off = l_idx % 100
        rg_groups.setdefault((s_idx, rg_idx), []).append((uid, rg_off))

    audio_map: dict[str, bytes] = {}
    for (s_idx, rg_idx), items in sorted(rg_groups.items()):
        t = shards[s_idx].read_row_group(rg_idx, columns=["audio"])
        audio_col = t["audio"]
        for uid, rg_off in items:
            audio_map[uid] = audio_col[rg_off].as_py()["bytes"]

    return audio_map


def run_telugu_option_1b_preprocessing(
    out_dir: Path = DEFAULT_OUT_DIR,
    split_dir: Path = OPTION_1B_SPLIT_DIR,
    textgrid_root: Path = TEXTGRID_ROOT_DIR,
    batch_size: int = 250,
) -> dict[str, Any]:
    """Execute the resumable Telugu Option 1B acoustic feature extraction pipeline.

    Args:
        out_dir: Output directory (`preprocessed_data/Telugu_Option_1B`).
        split_dir: Option 1B split directory containing `train.txt` and `val.txt`.
        textgrid_root: Root directory containing `train/*.TextGrid` and `val/*.TextGrid`.
        batch_size: Number of utterances per Parquet row-group read batch.

    Returns:
        Summary dictionary with counts, durations, normalization stats, and validation metrics.
    """
    t0 = time.perf_counter()

    # 1. Prepare output subdirectories and speakers.json
    for sub in ("mel", "pitch", "energy", "duration", "pitch_phoneme", "energy_phoneme"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    speakers_json_path = out_dir / "speakers.json"
    _atomic_write_json(speakers_json_path, SPEAKERS_MAP)

    # 2. Load Option 1B train and validation manifests (NEVER load test.txt)
    train_entries = _load_option_1b_split_manifest(split_dir / "train.txt", "train")
    val_entries = _load_option_1b_split_manifest(split_dir / "val.txt", "val")

    extractor = TeluguFeatureExtractor()
    stats_json_path = out_dir / "stats.json"
    raw_cache_dir = out_dir / ".raw_train_cache"

    failures_and_exclusions: list[dict[str, Any]] = []

    # Check if valid stats.json already exists from a completed training pass
    existing_stats: dict[str, Any] | None = None
    if stats_json_path.is_file():
        try:
            loaded_stats = json.loads(stats_json_path.read_text(encoding="utf-8"))
            if (
                "pitch" in loaded_stats
                and "energy" in loaded_stats
                and "phoneme_level" in loaded_stats
                and len(loaded_stats["pitch"]) == 4
                and len(loaded_stats["energy"]) == 4
                and float(loaded_stats["pitch"][3]) > 0
                and float(loaded_stats["energy"][3]) > 0
            ):
                existing_stats = loaded_stats
        except Exception:
            existing_stats = None

    # ------------------------------------------------------------------
    # PHASE A: Training Split Extraction & Training-Only Normalization
    # ------------------------------------------------------------------
    train_out_lines: list[str] = []
    train_resumed_count = 0
    train_extracted_count = 0
    train_reconciled_count = 0
    train_discrepancies: list[int] = []
    train_total_raw_audio_sec = 0.0
    train_total_trimmed_audio_sec = 0.0
    train_total_mel_frames = 0
    train_total_tokens = 0
    train_female_succeeded = 0
    train_male_succeeded = 0

    # First filter out the explicitly excluded failed alignments and check resume status
    active_train_entries: list[dict[str, Any]] = []
    for entry in train_entries:
        uid = entry["id"]
        tg_path = textgrid_root / "train" / f"{uid}.TextGrid"
        if uid in EXCLUDED_FAILED_IDS:
            failures_and_exclusions.append(
                {
                    "id": uid,
                    "split": "train",
                    "speaker_id": entry["speaker_id"],
                    "gender": entry["gender"],
                    "reason": "strictly_excluded_failed_alignment_upstream_truncated_audio",
                    "textgrid_exists": tg_path.exists(),
                }
            )
            continue
        if not tg_path.exists():
            failures_and_exclusions.append(
                {
                    "id": uid,
                    "split": "train",
                    "speaker_id": entry["speaker_id"],
                    "gender": entry["gender"],
                    "reason": "missing_textgrid",
                    "textgrid_exists": False,
                }
            )
            continue
        active_train_entries.append(entry)

    # Check if all active training entries are already extracted and validated on disk
    resumed_train_records: dict[str, dict[str, Any]] = {}
    pending_train_entries: list[dict[str, Any]] = []

    if existing_stats is not None:
        for entry in active_train_entries:
            uid = entry["id"]
            spk_str = entry["speaker_str"]
            tg_path = textgrid_root / "train" / f"{uid}.TextGrid"
            try:
                dur_path = out_dir / "duration" / f"{spk_str}-duration-{uid}.npy"
                target_mel = int(np.sum(np.load(dur_path))) if dur_path.is_file() else None
                dur_res = extract_telugu_durations_from_textgrid(
                    tg_path, target_mel_length=target_mel
                )
                ok, err, meta = validate_utterance_feature_files(
                    out_dir, uid, spk_str, dur_res.fastspeech2_text
                )
                if ok and meta is not None:
                    resumed_train_records[uid] = {
                        "line": f"{uid}|{spk_str}|{dur_res.fastspeech2_text}|{entry['normalized_text']}",
                        "t_mel": meta["t_mel"],
                        "n_phones": meta["n_phones"],
                        "trimmed_duration_sec": float(dur_res.end_time - dur_res.start_time),
                        "discrepancy": dur_res.discrepancy,
                        "reconciliation_applied": dur_res.reconciliation_applied,
                        "pitch_min": meta["pitch_min"],
                        "pitch_max": meta["pitch_max"],
                        "energy_min": meta["energy_min"],
                        "energy_max": meta["energy_max"],
                    }
                else:
                    pending_train_entries.append(entry)
            except Exception:
                pending_train_entries.append(entry)
    else:
        pending_train_entries = list(active_train_entries)

    if not pending_train_entries and existing_stats is not None:
        # All training utterances are already extracted, normalized, and validated
        for entry in active_train_entries:
            uid = entry["id"]
            rec = resumed_train_records[uid]
            train_out_lines.append(rec["line"])
            train_resumed_count += 1
            if rec["reconciliation_applied"]:
                train_reconciled_count += 1
            train_discrepancies.append(rec["discrepancy"])
            train_total_raw_audio_sec += entry["manifest_duration_sec"]
            train_total_trimmed_audio_sec += rec["trimmed_duration_sec"]
            train_total_mel_frames += rec["t_mel"]
            train_total_tokens += rec["n_phones"]
            if entry["gender"] == "female":
                train_female_succeeded += 1
            else:
                train_male_succeeded += 1

        pitch_min, pitch_max, pitch_frame_mean, pitch_frame_std = [
            float(x) for x in existing_stats["pitch"]
        ]
        energy_min, energy_max, energy_frame_mean, energy_frame_std = [
            float(x) for x in existing_stats["energy"]
        ]
        pitch_mean, pitch_std = [
            float(x) for x in existing_stats["phoneme_level"]["pitch"]
        ]
        energy_mean, energy_std = [
            float(x) for x in existing_stats["phoneme_level"]["energy"]
        ]
        stats_data = existing_stats
    elif existing_stats is not None and len(resumed_train_records) > 0:
        # Partial resume when stats.json already exists: preserve valid files and extract only missing ones
        pitch_min, pitch_max, pitch_frame_mean, pitch_frame_std = [
            float(x) for x in existing_stats["pitch"]
        ]
        energy_min, energy_max, energy_frame_mean, energy_frame_std = [
            float(x) for x in existing_stats["energy"]
        ]
        pitch_mean, pitch_std = [
            float(x) for x in existing_stats["phoneme_level"]["pitch"]
        ]
        energy_mean, energy_std = [
            float(x) for x in existing_stats["phoneme_level"]["energy"]
        ]
        stats_data = existing_stats

        for b_start in range(0, len(pending_train_entries), batch_size):
            batch = pending_train_entries[b_start : b_start + batch_size]
            audio_map = _load_parquet_audio_for_ids(
                {e["id"] for e in batch},
                forbidden_ids=set(EXCLUDED_FAILED_IDS),
            )
            for entry in batch:
                uid = entry["id"]
                spk_str = entry["speaker_str"]
                tg_path = textgrid_root / "train" / f"{uid}.TextGrid"
                try:
                    wav_data, orig_sr = sf.read(io.BytesIO(audio_map[uid]), dtype="float32")
                    resampled_wav, _ = resample_audio(wav_data, orig_sr, SAMPLING_RATE)
                    res = extractor.extract_from_waveform_and_textgrid(resampled_wav, tg_path)
                except Exception as e:
                    failures_and_exclusions.append(
                        {
                            "id": uid,
                            "split": "train",
                            "speaker_id": entry["speaker_id"],
                            "gender": entry["gender"],
                            "reason": f"feature_extraction_error:{e}",
                            "textgrid_exists": tg_path.exists(),
                        }
                    )
                    continue

                p_norm = ((res["pitch_frame"] - pitch_frame_mean) / pitch_frame_std).astype(np.float32)
                e_norm = ((res["energy_frame"] - energy_frame_mean) / energy_frame_std).astype(np.float32)
                p_ph_norm = ((res["pitch_phoneme"] - pitch_mean) / pitch_std).astype(np.float32)
                e_ph_norm = ((res["energy_phoneme"] - energy_mean) / energy_std).astype(np.float32)

                _atomic_save_npy(out_dir / "mel" / f"{spk_str}-mel-{uid}.npy", res["mel"])
                _atomic_save_npy(out_dir / "duration" / f"{spk_str}-duration-{uid}.npy", res["durations"])
                _atomic_save_npy(out_dir / "pitch" / f"{spk_str}-pitch-{uid}.npy", p_norm)
                _atomic_save_npy(out_dir / "energy" / f"{spk_str}-energy-{uid}.npy", e_norm)
                _atomic_save_npy(out_dir / "pitch_phoneme" / f"{spk_str}-pitch-{uid}.npy", p_ph_norm)
                _atomic_save_npy(out_dir / "energy_phoneme" / f"{spk_str}-energy-{uid}.npy", e_ph_norm)

                ok, err, meta = validate_utterance_feature_files(
                    out_dir, uid, spk_str, res["fastspeech2_text"]
                )
                if not ok or meta is None:
                    raise RuntimeError(f"Post-write validation failed for {uid}: {err}")

                resumed_train_records[uid] = {
                    "line": f"{uid}|{spk_str}|{res['fastspeech2_text']}|{entry['normalized_text']}",
                    "t_mel": meta["t_mel"],
                    "n_phones": meta["n_phones"],
                    "trimmed_duration_sec": res["trimmed_duration_sec"],
                    "discrepancy": res["discrepancy"],
                    "reconciliation_applied": res["reconciliation_applied"],
                    "pitch_min": meta["pitch_min"],
                    "pitch_max": meta["pitch_max"],
                    "energy_min": meta["energy_min"],
                    "energy_max": meta["energy_max"],
                    "extracted_now": True,
                }

        for entry in active_train_entries:
            uid = entry["id"]
            if uid not in resumed_train_records:
                continue
            rec = resumed_train_records[uid]
            train_out_lines.append(rec["line"])
            if rec.get("extracted_now"):
                train_extracted_count += 1
            else:
                train_resumed_count += 1
            if rec["reconciliation_applied"]:
                train_reconciled_count += 1
            train_discrepancies.append(rec["discrepancy"])
            train_total_raw_audio_sec += entry["manifest_duration_sec"]
            train_total_trimmed_audio_sec += rec["trimmed_duration_sec"]
            train_total_mel_frames += rec["t_mel"]
            train_total_tokens += rec["n_phones"]
            if entry["gender"] == "female":
                train_female_succeeded += 1
            else:
                train_male_succeeded += 1
    else:
        # Full training pass (with mid-pass-1 raw cache resume support)
        raw_cache_dir.mkdir(parents=True, exist_ok=True)
        train_raw_results: dict[str, dict[str, Any]] = {}
        train_pitch_pool: list[float] = []
        train_energy_pool: list[float] = []
        train_pitch_frame_pool: list[float] = []
        train_energy_frame_pool: list[float] = []

        # Check which entries already have valid raw cached arrays + mel + duration
        to_extract_train: list[dict[str, Any]] = []
        for entry in active_train_entries:
            uid = entry["id"]
            spk_str = entry["speaker_str"]
            cache_npz = raw_cache_dir / f"{uid}.npz"
            mel_path = out_dir / "mel" / f"{spk_str}-mel-{uid}.npy"
            dur_path = out_dir / "duration" / f"{spk_str}-duration-{uid}.npy"
            if cache_npz.is_file() and mel_path.is_file() and dur_path.is_file():
                try:
                    c = np.load(cache_npz, allow_pickle=False)
                    train_raw_results[uid] = {
                        "pitch_frame": c["pitch_frame"],
                        "energy_frame": c["energy_frame"],
                        "pitch_phoneme": c["pitch_phoneme"],
                        "energy_phoneme": c["energy_phoneme"],
                        "fastspeech2_text": str(c["fastspeech2_text"]),
                        "discrepancy": int(c["discrepancy"]),
                        "reconciliation_applied": bool(c["reconciliation_applied"]),
                        "t_mel": int(c["t_mel"]),
                        "trimmed_duration_sec": float(c["trimmed_duration_sec"]),
                        "resumed_from_cache": True,
                    }
                    continue
                except Exception:
                    pass
            to_extract_train.append(entry)

        for b_start in range(0, len(to_extract_train), batch_size):
            batch = to_extract_train[b_start : b_start + batch_size]
            audio_map = _load_parquet_audio_for_ids(
                {e["id"] for e in batch},
                forbidden_ids=set(EXCLUDED_FAILED_IDS),
            )
            for entry in batch:
                uid = entry["id"]
                spk_str = entry["speaker_str"]
                tg_path = textgrid_root / "train" / f"{uid}.TextGrid"
                try:
                    wav_data, orig_sr = sf.read(io.BytesIO(audio_map[uid]), dtype="float32")
                    resampled_wav, _ = resample_audio(wav_data, orig_sr, SAMPLING_RATE)
                    res = extractor.extract_from_waveform_and_textgrid(resampled_wav, tg_path)
                except Exception as e:
                    failures_and_exclusions.append(
                        {
                            "id": uid,
                            "split": "train",
                            "speaker_id": entry["speaker_id"],
                            "gender": entry["gender"],
                            "reason": f"feature_extraction_error:{e}",
                            "textgrid_exists": tg_path.exists(),
                        }
                    )
                    continue

                # Save mel and duration immediately (independent of pitch/energy z-score stats)
                _atomic_save_npy(out_dir / "mel" / f"{spk_str}-mel-{uid}.npy", res["mel"])
                _atomic_save_npy(out_dir / "duration" / f"{spk_str}-duration-{uid}.npy", res["durations"])

                # Cache raw pitch/energy for crash-safe mid-pass resume
                np.savez(
                    raw_cache_dir / f"{uid}.npz",
                    pitch_frame=res["pitch_frame"],
                    energy_frame=res["energy_frame"],
                    pitch_phoneme=res["pitch_phoneme"],
                    energy_phoneme=res["energy_phoneme"],
                    fastspeech2_text=np.array(res["fastspeech2_text"]),
                    discrepancy=np.array(res["discrepancy"], dtype=np.int32),
                    reconciliation_applied=np.array(res["reconciliation_applied"]),
                    t_mel=np.array(res["t_mel"], dtype=np.int32),
                    trimmed_duration_sec=np.array(res["trimmed_duration_sec"], dtype=np.float64),
                )
                train_raw_results[uid] = {
                    "pitch_frame": res["pitch_frame"],
                    "energy_frame": res["energy_frame"],
                    "pitch_phoneme": res["pitch_phoneme"],
                    "energy_phoneme": res["energy_phoneme"],
                    "fastspeech2_text": res["fastspeech2_text"],
                    "discrepancy": res["discrepancy"],
                    "reconciliation_applied": res["reconciliation_applied"],
                    "t_mel": res["t_mel"],
                    "trimmed_duration_sec": res["trimmed_duration_sec"],
                    "resumed_from_cache": False,
                }

        # Accumulate training pools in deterministic manifest order
        for entry in active_train_entries:
            uid = entry["id"]
            if uid not in train_raw_results:
                continue
            r = train_raw_results[uid]
            train_pitch_pool.extend(remove_outliers(r["pitch_phoneme"]))
            train_energy_pool.extend(remove_outliers(r["energy_phoneme"]))
            train_pitch_frame_pool.extend(remove_outliers(r["pitch_frame"]))
            train_energy_frame_pool.extend(remove_outliers(r["energy_frame"]))

        # Fit StandardScalers strictly on training pools
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        pitch_scaler.fit(np.asarray(train_pitch_pool, dtype=np.float64).reshape(-1, 1))
        energy_scaler.fit(np.asarray(train_energy_pool, dtype=np.float64).reshape(-1, 1))
        pitch_mean = float(pitch_scaler.mean_[0])
        pitch_std = float(pitch_scaler.scale_[0])
        energy_mean = float(energy_scaler.mean_[0])
        energy_std = float(energy_scaler.scale_[0])

        pitch_frame_scaler = StandardScaler()
        energy_frame_scaler = StandardScaler()
        pitch_frame_scaler.fit(np.asarray(train_pitch_frame_pool, dtype=np.float64).reshape(-1, 1))
        energy_frame_scaler.fit(np.asarray(train_energy_frame_pool, dtype=np.float64).reshape(-1, 1))
        pitch_frame_mean = float(pitch_frame_scaler.mean_[0])
        pitch_frame_std = float(pitch_frame_scaler.scale_[0])
        energy_frame_mean = float(energy_frame_scaler.mean_[0])
        energy_frame_std = float(energy_frame_scaler.scale_[0])

        pitch_min, pitch_max = float("inf"), float("-inf")
        energy_min, energy_max = float("inf"), float("-inf")

        # Write normalized training pitch & energy files and validate every utterance
        for entry in active_train_entries:
            uid = entry["id"]
            if uid not in train_raw_results:
                continue
            spk_str = entry["speaker_str"]
            r = train_raw_results[uid]

            p_norm = ((r["pitch_frame"] - pitch_frame_mean) / pitch_frame_std).astype(np.float32)
            e_norm = ((r["energy_frame"] - energy_frame_mean) / energy_frame_std).astype(np.float32)
            p_ph_norm = ((r["pitch_phoneme"] - pitch_mean) / pitch_std).astype(np.float32)
            e_ph_norm = ((r["energy_phoneme"] - energy_mean) / energy_std).astype(np.float32)

            _atomic_save_npy(out_dir / "pitch" / f"{spk_str}-pitch-{uid}.npy", p_norm)
            _atomic_save_npy(out_dir / "energy" / f"{spk_str}-energy-{uid}.npy", e_norm)
            _atomic_save_npy(out_dir / "pitch_phoneme" / f"{spk_str}-pitch-{uid}.npy", p_ph_norm)
            _atomic_save_npy(out_dir / "energy_phoneme" / f"{spk_str}-energy-{uid}.npy", e_ph_norm)

            ok, err, meta = validate_utterance_feature_files(
                out_dir, uid, spk_str, r["fastspeech2_text"]
            )
            if not ok or meta is None:
                raise RuntimeError(f"Post-write validation failed for train utterance {uid}: {err}")

            pitch_min = min(pitch_min, float(p_norm.min()))
            pitch_max = max(pitch_max, float(p_norm.max()))
            energy_min = min(energy_min, float(e_norm.min()))
            energy_max = max(energy_max, float(e_norm.max()))

            train_out_lines.append(
                f"{uid}|{spk_str}|{r['fastspeech2_text']}|{entry['normalized_text']}"
            )
            if r.get("resumed_from_cache"):
                train_resumed_count += 1
            else:
                train_extracted_count += 1
            if r["reconciliation_applied"]:
                train_reconciled_count += 1
            train_discrepancies.append(r["discrepancy"])
            train_total_raw_audio_sec += entry["manifest_duration_sec"]
            train_total_trimmed_audio_sec += r["trimmed_duration_sec"]
            train_total_mel_frames += meta["t_mel"]
            train_total_tokens += meta["n_phones"]
            if entry["gender"] == "female":
                train_female_succeeded += 1
            else:
                train_male_succeeded += 1

        stats_data = {
            "pitch": [
                float(pitch_min),
                float(pitch_max),
                float(pitch_frame_mean),
                float(pitch_frame_std),
            ],
            "energy": [
                float(energy_min),
                float(energy_max),
                float(energy_frame_mean),
                float(energy_frame_std),
            ],
            "phoneme_level": {
                "pitch": [float(pitch_mean), float(pitch_std)],
                "energy": [float(energy_mean), float(energy_std)],
            },
        }
        _atomic_write_json(stats_json_path, stats_data)

        # Clean up temporary raw cache once stats.json and normalized arrays are verified
        import shutil

        shutil.rmtree(raw_cache_dir, ignore_errors=True)

    # Write train.txt atomically
    train_txt_path = out_dir / "train.txt"
    tmp_train = train_txt_path.with_suffix(".txt.tmp")
    with open(tmp_train, "w", encoding="utf-8") as f:
        for line in train_out_lines:
            f.write(line + "\n")
    os.replace(tmp_train, train_txt_path)

    # ------------------------------------------------------------------
    # PHASE B: Validation Split Extraction (Frozen Training Stats Only)
    # ------------------------------------------------------------------
    val_out_lines: list[str] = []
    val_resumed_count = 0
    val_extracted_count = 0
    val_reconciled_count = 0
    val_discrepancies: list[int] = []
    val_total_raw_audio_sec = 0.0
    val_total_trimmed_audio_sec = 0.0
    val_total_mel_frames = 0
    val_total_tokens = 0
    val_female_succeeded = 0
    val_male_succeeded = 0

    active_val_entries: list[dict[str, Any]] = []
    for entry in val_entries:
        uid = entry["id"]
        tg_path = textgrid_root / "val" / f"{uid}.TextGrid"
        if uid in EXCLUDED_FAILED_IDS:
            failures_and_exclusions.append(
                {
                    "id": uid,
                    "split": "val",
                    "speaker_id": entry["speaker_id"],
                    "gender": entry["gender"],
                    "reason": "strictly_excluded_failed_alignment_upstream_truncated_audio",
                    "textgrid_exists": tg_path.exists(),
                }
            )
            continue
        if not tg_path.exists():
            failures_and_exclusions.append(
                {
                    "id": uid,
                    "split": "val",
                    "speaker_id": entry["speaker_id"],
                    "gender": entry["gender"],
                    "reason": "missing_textgrid",
                    "textgrid_exists": False,
                }
            )
            continue
        active_val_entries.append(entry)

    resumed_val_records: dict[str, dict[str, Any]] = {}
    pending_val_entries: list[dict[str, Any]] = []

    for entry in active_val_entries:
        uid = entry["id"]
        spk_str = entry["speaker_str"]
        tg_path = textgrid_root / "val" / f"{uid}.TextGrid"
        try:
            dur_path = out_dir / "duration" / f"{spk_str}-duration-{uid}.npy"
            target_mel = int(np.sum(np.load(dur_path))) if dur_path.is_file() else None
            dur_res = extract_telugu_durations_from_textgrid(
                tg_path, target_mel_length=target_mel
            )
            ok, err, meta = validate_utterance_feature_files(
                out_dir, uid, spk_str, dur_res.fastspeech2_text
            )
            if ok and meta is not None:
                resumed_val_records[uid] = {
                    "line": f"{uid}|{spk_str}|{dur_res.fastspeech2_text}|{entry['normalized_text']}",
                    "t_mel": meta["t_mel"],
                    "n_phones": meta["n_phones"],
                    "trimmed_duration_sec": float(dur_res.end_time - dur_res.start_time),
                    "discrepancy": dur_res.discrepancy,
                    "reconciliation_applied": dur_res.reconciliation_applied,
                    "extracted_now": False,
                }
            else:
                pending_val_entries.append(entry)
        except Exception:
            pending_val_entries.append(entry)

    for b_start in range(0, len(pending_val_entries), batch_size):
        batch = pending_val_entries[b_start : b_start + batch_size]
        audio_map = _load_parquet_audio_for_ids(
            {e["id"] for e in batch},
            forbidden_ids=set(EXCLUDED_FAILED_IDS),
        )
        for entry in batch:
            uid = entry["id"]
            spk_str = entry["speaker_str"]
            tg_path = textgrid_root / "val" / f"{uid}.TextGrid"
            try:
                wav_data, orig_sr = sf.read(io.BytesIO(audio_map[uid]), dtype="float32")
                resampled_wav, _ = resample_audio(wav_data, orig_sr, SAMPLING_RATE)
                res = extractor.extract_from_waveform_and_textgrid(resampled_wav, tg_path)
            except Exception as e:
                failures_and_exclusions.append(
                    {
                        "id": uid,
                        "split": "val",
                        "speaker_id": entry["speaker_id"],
                        "gender": entry["gender"],
                        "reason": f"feature_extraction_error:{e}",
                        "textgrid_exists": tg_path.exists(),
                    }
                )
                continue

            p_norm = ((res["pitch_frame"] - pitch_frame_mean) / pitch_frame_std).astype(np.float32)
            e_norm = ((res["energy_frame"] - energy_frame_mean) / energy_frame_std).astype(np.float32)
            p_ph_norm = ((res["pitch_phoneme"] - pitch_mean) / pitch_std).astype(np.float32)
            e_ph_norm = ((res["energy_phoneme"] - energy_mean) / energy_std).astype(np.float32)

            _atomic_save_npy(out_dir / "mel" / f"{spk_str}-mel-{uid}.npy", res["mel"])
            _atomic_save_npy(out_dir / "duration" / f"{spk_str}-duration-{uid}.npy", res["durations"])
            _atomic_save_npy(out_dir / "pitch" / f"{spk_str}-pitch-{uid}.npy", p_norm)
            _atomic_save_npy(out_dir / "energy" / f"{spk_str}-energy-{uid}.npy", e_norm)
            _atomic_save_npy(out_dir / "pitch_phoneme" / f"{spk_str}-pitch-{uid}.npy", p_ph_norm)
            _atomic_save_npy(out_dir / "energy_phoneme" / f"{spk_str}-energy-{uid}.npy", e_ph_norm)

            ok, err, meta = validate_utterance_feature_files(
                out_dir, uid, spk_str, res["fastspeech2_text"]
            )
            if not ok or meta is None:
                raise RuntimeError(f"Post-write validation failed for val utterance {uid}: {err}")

            resumed_val_records[uid] = {
                "line": f"{uid}|{spk_str}|{res['fastspeech2_text']}|{entry['normalized_text']}",
                "t_mel": meta["t_mel"],
                "n_phones": meta["n_phones"],
                "trimmed_duration_sec": res["trimmed_duration_sec"],
                "discrepancy": res["discrepancy"],
                "reconciliation_applied": res["reconciliation_applied"],
                "extracted_now": True,
            }

    for entry in active_val_entries:
        uid = entry["id"]
        if uid not in resumed_val_records:
            continue
        rec = resumed_val_records[uid]
        val_out_lines.append(rec["line"])
        if rec.get("extracted_now"):
            val_extracted_count += 1
        else:
            val_resumed_count += 1
        if rec["reconciliation_applied"]:
            val_reconciled_count += 1
        val_discrepancies.append(rec["discrepancy"])
        val_total_raw_audio_sec += entry["manifest_duration_sec"]
        val_total_trimmed_audio_sec += rec["trimmed_duration_sec"]
        val_total_mel_frames += rec["t_mel"]
        val_total_tokens += rec["n_phones"]
        if entry["gender"] == "female":
            val_female_succeeded += 1
        else:
            val_male_succeeded += 1

    # Write val.txt atomically
    val_txt_path = out_dir / "val.txt"
    tmp_val = val_txt_path.with_suffix(".txt.tmp")
    with open(tmp_val, "w", encoding="utf-8") as f:
        for line in val_out_lines:
            f.write(line + "\n")
    os.replace(tmp_val, val_txt_path)

    # Save failures/exclusions and summary JSON
    failures_path = out_dir / "extraction_failures.json"
    _atomic_write_json(failures_path, failures_and_exclusions)

    elapsed_sec = time.perf_counter() - t0
    summary = {
        "dataset": "Telugu_Option_1B",
        "output_dir": str(out_dir),
        "acoustic_parameters": {
            "sampling_rate_hz": SAMPLING_RATE,
            "n_fft": N_FFT,
            "win_length": WIN_LENGTH,
            "hop_length": HOP_LENGTH,
            "n_mel_channels": N_MEL,
            "mel_fmin_hz": MEL_FMIN,
            "mel_fmax_hz": MEL_FMAX,
            "f0_extractor": "pyworld_dio_stonemask_linear_unvoiced_interpolation",
            "energy_extractor": "stft_magnitude_l2_norm",
            "normalization_fit_split": "train_only_iqr_outlier_filtered",
        },
        "test_split_accessed": False,
        "excluded_failed_alignment_ids": sorted(EXCLUDED_FAILED_IDS),
        "splits": {
            "train": {
                "manifest_utterances": len(train_entries),
                "processed_utterances": len(train_out_lines),
                "newly_extracted_utterances": train_extracted_count,
                "resumed_valid_utterances": train_resumed_count,
                "excluded_or_failed_utterances": len(train_entries) - len(train_out_lines),
                "female_utterances": train_female_succeeded,
                "male_utterances": train_male_succeeded,
                "total_raw_audio_seconds": round(train_total_raw_audio_sec, 4),
                "total_raw_audio_hours": round(train_total_raw_audio_sec / 3600.0, 6),
                "total_trimmed_audio_seconds": round(train_total_trimmed_audio_sec, 4),
                "total_trimmed_audio_hours": round(train_total_trimmed_audio_sec / 3600.0, 6),
                "total_mel_frames": train_total_mel_frames,
                "total_phoneme_tokens": train_total_tokens,
                "reconciled_utterances": train_reconciled_count,
                "min_frame_discrepancy_before_reconciliation": int(min(train_discrepancies)) if train_discrepancies else 0,
                "max_frame_discrepancy_before_reconciliation": int(max(train_discrepancies)) if train_discrepancies else 0,
            },
            "val": {
                "manifest_utterances": len(val_entries),
                "processed_utterances": len(val_out_lines),
                "newly_extracted_utterances": val_extracted_count,
                "resumed_valid_utterances": val_resumed_count,
                "excluded_or_failed_utterances": len(val_entries) - len(val_out_lines),
                "female_utterances": val_female_succeeded,
                "male_utterances": val_male_succeeded,
                "total_raw_audio_seconds": round(val_total_raw_audio_sec, 4),
                "total_raw_audio_hours": round(val_total_raw_audio_sec / 3600.0, 6),
                "total_trimmed_audio_seconds": round(val_total_trimmed_audio_sec, 4),
                "total_trimmed_audio_hours": round(val_total_trimmed_audio_sec / 3600.0, 6),
                "total_mel_frames": val_total_mel_frames,
                "total_phoneme_tokens": val_total_tokens,
                "reconciled_utterances": val_reconciled_count,
                "min_frame_discrepancy_before_reconciliation": int(min(val_discrepancies)) if val_discrepancies else 0,
                "max_frame_discrepancy_before_reconciliation": int(max(val_discrepancies)) if val_discrepancies else 0,
            },
        },
        "total_processed_utterances": len(train_out_lines) + len(val_out_lines),
        "total_excluded_or_failed_utterances": len(failures_and_exclusions),
        "stats": stats_data,
        "failures_and_exclusions": failures_and_exclusions,
        "elapsed_wall_sec": round(elapsed_sec, 3),
    }
    _atomic_write_json(out_dir / "extraction_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reproducible, resumable acoustic feature extraction for Telugu Option 1B."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for preprocessed Telugu Option 1B features.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=250,
        help="Number of utterances per Parquet row-group read batch.",
    )
    args = parser.parse_args()
    print("=" * 72)
    print("TELUGU OPTION 1B FASTSPEECH 2 ACOUSTIC FEATURE EXTRACTION PIPELINE")
    print("=" * 72)
    summary = run_telugu_option_1b_preprocessing(
        out_dir=Path(args.out_dir),
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
