"""Acoustic feature extraction and dataset serialization for Svarah FastSpeech 2.

Extracts 80-channel mel-spectrograms, frame-level and phoneme-level pitch (F0 via
PyWorld DIO+Stonemask), frame-level and phoneme-level energy (STFT L2 norm),
reconciles duration targets against mel sequence length, computes normalization
statistics strictly on training data, and serializes the complete dataset for FastSpeech 2.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))
import time
from typing import Any
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import soundfile as sf
import torch
from tqdm import tqdm

from common.paths import (
    DATA_DIR,
    ENGLISH_DIR,
    PROJECT_ROOT,
    get_preprocessed_dir,
    setup_project_paths,
)

setup_project_paths()

from English.alignment.duration import (
    extract_durations_from_textgrid,
    validate_symbols_against_vocabulary,
)

TRAIN_CORPUS_DIR = DATA_DIR / "svarah_train" / "corpus"
TRAIN_TG_DIR = DATA_DIR / "svarah_train" / "textgrids"
VAL_CORPUS_DIR = DATA_DIR / "svarah_val" / "corpus"
VAL_TG_DIR = DATA_DIR / "svarah_val" / "textgrids"

OUT_DIR = get_preprocessed_dir("english")
MANIFEST_PATH = ENGLISH_DIR / "metadata" / "full_manifest.json"
SPEAKER_PROFILES_PATH = ENGLISH_DIR / "metadata" / "speaker_profiles.json"

# FastSpeech 2 & HiFi-GAN acoustic parameters
SAMPLING_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MEL = 80
MEL_FMIN = 0.0
MEL_FMAX = 8000.0


def remove_outliers(values: np.ndarray) -> np.ndarray:
    """Filter out extreme values using IQR rule before fitting scalers."""
    v = np.array(values, dtype=np.float64)
    v_nonzero = v[v > 0]
    if len(v_nonzero) < 4:
        return v
    p25 = np.percentile(v_nonzero, 25)
    p75 = np.percentile(v_nonzero, 75)
    iqr = p75 - p25
    lower = p25 - 1.5 * iqr
    upper = p75 + 1.5 * iqr
    return v[(v >= lower) & (v <= upper)]


class FeatureExtractor:
    """Extracts mel-spectrogram, pitch, energy and reconciles durations."""

    def __init__(self, sample_rate: int = SAMPLING_RATE, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH, win_length: int = WIN_LENGTH, n_mels: int = N_MEL, fmin: float = MEL_FMIN, fmax: float = MEL_FMAX):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop = hop_length
        self.win = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        # Mel filterbank basis
        self.mel_basis = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        self.window = torch.hann_window(self.win)

    def extract_features_single(
        self,
        wav_path: str | Path,
        tg_path: str | Path,
    ) -> dict[str, Any] | None:
        """Extract acoustic features and reconcile durations for a single utterance."""
        if not os.path.exists(wav_path) or not os.path.exists(tg_path):
            return None

        # 1. Initial duration extraction to determine trimmed boundaries
        res_init = extract_durations_from_textgrid(
            tg_path,
            sampling_rate=self.sr,
            hop_length=self.hop,
            trim_silence=True,
        )
        if len(res_init.phones) == 0 or res_init.start_time >= res_init.end_time:
            return None

        # 2. Load audio and slice to trimmed boundary
        audio, orig_sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        s_sample = int(self.sr * res_init.start_time)
        e_sample = int(self.sr * res_init.end_time)
        audio_trimmed = audio[s_sample:e_sample]

        if len(audio_trimmed) < self.n_fft:
            return None

        # 3. Compute STFT, Mel-Spectrogram and Energy
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
        mag = torch.abs(stft).numpy()  # [n_fft//2 + 1, T]
        energy = np.linalg.norm(mag, axis=0)  # [T]
        mel = np.log(np.clip(self.mel_basis @ mag, a_min=1e-5, a_max=None))  # [80, T]
        t_mel = mel.shape[1]

        # 4. Reconcile durations with actual mel length
        res = extract_durations_from_textgrid(
            tg_path,
            target_mel_length=t_mel,
            sampling_rate=self.sr,
            hop_length=self.hop,
            trim_silence=True,
        )
        if res.total_frames != t_mel:
            return None

        # 5. Extract fundamental frequency (F0) using PyWorld DIO + Stonemask
        _f0, t_f0 = pw.dio(
            audio_trimmed.astype(np.float64),
            self.sr,
            frame_period=self.hop / self.sr * 1000,
        )
        f0 = pw.stonemask(audio_trimmed.astype(np.float64), _f0, t_f0, self.sr)

        # Align F0 frame length to t_mel
        if len(f0) < t_mel:
            f0 = np.pad(f0, (0, t_mel - len(f0)), mode="edge")
        else:
            f0 = f0[:t_mel]

        # Interpolate across unvoiced regions for continuous contour
        nonzero_ids = np.where(f0 > 0)[0]
        if len(nonzero_ids) > 1:
            interp_fn = interp1d(
                nonzero_ids,
                f0[nonzero_ids],
                fill_value=(f0[nonzero_ids[0]], f0[nonzero_ids[-1]]),
                bounds_error=False,
            )
            f0_interp = interp_fn(np.arange(t_mel))
        elif len(nonzero_ids) == 1:
            f0_interp = np.full(t_mel, f0[nonzero_ids[0]])
        else:
            f0_interp = np.zeros(t_mel)

        # 6. Compute phoneme-level averaged pitch and energy
        pitch_phoneme = np.zeros(len(res.durations), dtype=np.float32)
        energy_phoneme = np.zeros(len(res.durations), dtype=np.float32)
        pos = 0
        for i, d in enumerate(res.durations):
            if d > 0:
                pitch_phoneme[i] = np.mean(f0_interp[pos : pos + d])
                energy_phoneme[i] = np.mean(energy[pos : pos + d])
            else:
                pitch_phoneme[i] = 0.0
                energy_phoneme[i] = 0.0
            pos += d

        return {
            "mel": mel.T.astype(np.float32),  # [T, 80]
            "energy_frame": energy.astype(np.float32),  # [T]
            "pitch_frame": f0_interp.astype(np.float32),  # [T]
            "raw_f0": f0.astype(np.float32),
            "energy_phoneme": energy_phoneme,  # [L_phone]
            "pitch_phoneme": pitch_phoneme,  # [L_phone]
            "durations": np.array(res.durations, dtype=np.int32),  # [L_phone]
            "phones": res.phones,
            "fastspeech2_text": res.fastspeech2_text,
            "symbols": res.symbols,
            "discrepancy": res.discrepancy,
            "reconciliation_applied": res.reconciliation_applied,
            "t_mel": t_mel,
        }


def run_full_preprocessing():
    """Execute complete feature extraction and normalization pipeline."""
    t_start = time.time()
    print("=" * 70)
    print("Starting Svarah FastSpeech 2 Feature Preprocessing Pipeline")
    print("=" * 70)

    # 1. Prepare output directories
    for sub in ["mel", "pitch", "energy", "duration", "pitch_phoneme", "energy_phoneme"]:
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    # 2. Speaker mapping
    with open(SPEAKER_PROFILES_PATH, encoding="utf-8") as f:
        profiles_data = json.load(f)

    # Map profile strings 'spk_000' ... 'spk_114' to int ID 0..114
    speakers_map: dict[str, int] = {}
    for p in profiles_data["profiles"]:
        spk_str = f"spk_{p['speaker_profile_id']:03d}"
        speakers_map[spk_str] = p["speaker_profile_id"]

    speakers_json_path = OUT_DIR / "speakers.json"
    with open(speakers_json_path, "w", encoding="utf-8") as f:
        json.dump(speakers_map, f, indent=2)
    print(f"Serialized speaker mapping ({len(speakers_map)} profiles) to {speakers_json_path}")

    # 3. Load manifest
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest_list = json.load(f)
    manifest = {r["utterance_id"]: r for r in manifest_list}

    extractor = FeatureExtractor()

    # -------------------------------------------------------------
    # Step A: Process Training Data (Fit Scalers strictly on Train)
    # -------------------------------------------------------------
    print("\n--- Processing Training Data (3,215 utterances) ---")
    train_tg_files = sorted(TRAIN_TG_DIR.glob("*/*.TextGrid"))
    print(f"Found {len(train_tg_files)} training TextGrids.")

    train_out_lines: list[str] = []
    train_pitch_pool: list[float] = []
    train_energy_pool: list[float] = []
    train_pitch_frame_pool: list[float] = []
    train_energy_frame_pool: list[float] = []

    train_discrepancies: list[int] = []
    train_reconciled_count = 0
    train_failed: list[tuple[str, str]] = []

    for tg_path in tqdm(train_tg_files, desc="Train Features"):
        uid = tg_path.stem
        spk_str = tg_path.parent.name
        wav_path = TRAIN_CORPUS_DIR / spk_str / f"{uid}.wav"

        res = extractor.extract_features_single(wav_path, tg_path)
        if res is None:
            train_failed.append((uid, "Feature extraction failed"))
            continue

        raw_text = manifest.get(uid, {}).get("raw_text", "")
        dataset_line = f"{uid}|{spk_str}|{res['fastspeech2_text']}|{raw_text}"
        train_out_lines.append(dataset_line)

        # Track reconciliation
        if res["reconciliation_applied"]:
            train_reconciled_count += 1
        train_discrepancies.append(res["discrepancy"])

        # Collect outlier-filtered values for pitch & energy
        # (Both frame-level and phoneme-level)
        train_pitch_pool.extend(remove_outliers(res["pitch_phoneme"]))
        train_energy_pool.extend(remove_outliers(res["energy_phoneme"]))
        train_pitch_frame_pool.extend(remove_outliers(res["pitch_frame"]))
        train_energy_frame_pool.extend(remove_outliers(res["energy_frame"]))

        # Save initial arrays before global normalization
        base_name = f"{spk_str}-mel-{uid}.npy"
        np.save(OUT_DIR / "mel" / base_name, res["mel"])
        np.save(OUT_DIR / "duration" / f"{spk_str}-duration-{uid}.npy", res["durations"])
        np.save(OUT_DIR / "pitch" / f"{spk_str}-pitch-{uid}.npy", res["pitch_frame"])
        np.save(OUT_DIR / "energy" / f"{spk_str}-energy-{uid}.npy", res["energy_frame"])
        np.save(OUT_DIR / "pitch_phoneme" / f"{spk_str}-pitch-{uid}.npy", res["pitch_phoneme"])
        np.save(OUT_DIR / "energy_phoneme" / f"{spk_str}-energy-{uid}.npy", res["energy_phoneme"])

    print(f"Successfully processed {len(train_out_lines)} / {len(train_tg_files)} training utterances.")
    print(f"Reconciled frames: {train_reconciled_count} / {len(train_out_lines)} (discrepancy range: {min(train_discrepancies)} to {max(train_discrepancies)})")

    # Fit Normalization Statistics ONLY ON TRAINING POOL
    print("\nComputing normalization statistics strictly on training data...")
    pitch_scaler = StandardScaler()
    energy_scaler = StandardScaler()
    pitch_scaler.fit(np.array(train_pitch_pool).reshape(-1, 1))
    energy_scaler.fit(np.array(train_energy_pool).reshape(-1, 1))

    pitch_mean = float(pitch_scaler.mean_[0])
    pitch_std = float(pitch_scaler.scale_[0])
    energy_mean = float(energy_scaler.mean_[0])
    energy_std = float(energy_scaler.scale_[0])

    # Frame-level scalers as well
    pitch_frame_scaler = StandardScaler()
    energy_frame_scaler = StandardScaler()
    pitch_frame_scaler.fit(np.array(train_pitch_frame_pool).reshape(-1, 1))
    energy_frame_scaler.fit(np.array(train_energy_frame_pool).reshape(-1, 1))
    pitch_frame_mean = float(pitch_frame_scaler.mean_[0])
    pitch_frame_std = float(pitch_frame_scaler.scale_[0])
    energy_frame_mean = float(energy_frame_scaler.mean_[0])
    energy_frame_std = float(energy_frame_scaler.scale_[0])

    print(f"Phoneme-level stats: Pitch mean={pitch_mean:.2f}, std={pitch_std:.2f} | Energy mean={energy_mean:.2f}, std={energy_std:.2f}")
    print(f"Frame-level stats:   Pitch mean={pitch_frame_mean:.2f}, std={pitch_frame_std:.2f} | Energy mean={energy_frame_mean:.2f}, std={energy_frame_std:.2f}")

    # Normalize training files on disk and find min/max
    pitch_min, pitch_max = np.inf, -np.inf
    energy_min, energy_max = np.inf, -np.inf

    for line in train_out_lines:
        uid, spk_str, _, _ = line.split("|")
        # Normalize frame-level
        p_path = OUT_DIR / "pitch" / f"{spk_str}-pitch-{uid}.npy"
        e_path = OUT_DIR / "energy" / f"{spk_str}-energy-{uid}.npy"
        p_norm = (np.load(p_path) - pitch_frame_mean) / pitch_frame_std
        e_norm = (np.load(e_path) - energy_frame_mean) / energy_frame_std
        np.save(p_path, p_norm.astype(np.float32))
        np.save(e_path, e_norm.astype(np.float32))

        pitch_min = min(pitch_min, float(p_norm.min()))
        pitch_max = max(pitch_max, float(p_norm.max()))
        energy_min = min(energy_min, float(e_norm.min()))
        energy_max = max(energy_max, float(e_norm.max()))

        # Normalize phoneme-level
        p_ph_path = OUT_DIR / "pitch_phoneme" / f"{spk_str}-pitch-{uid}.npy"
        e_ph_path = OUT_DIR / "energy_phoneme" / f"{spk_str}-energy-{uid}.npy"
        p_ph_norm = (np.load(p_ph_path) - pitch_mean) / pitch_std
        e_ph_norm = (np.load(e_ph_path) - energy_mean) / energy_std
        np.save(p_ph_path, p_ph_norm.astype(np.float32))
        np.save(e_ph_path, e_ph_norm.astype(np.float32))

    # Save stats.json
    stats_data = {
        "pitch": [pitch_min, pitch_max, pitch_frame_mean, pitch_frame_std],
        "energy": [energy_min, energy_max, energy_frame_mean, energy_frame_std],
        "phoneme_level": {
            "pitch": [float(pitch_mean), float(pitch_std)],
            "energy": [float(energy_mean), float(energy_std)],
        },
    }
    stats_json_path = OUT_DIR / "stats.json"
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)
    print(f"Serialized normalization statistics to {stats_json_path}")

    # Write train.txt
    train_txt_path = OUT_DIR / "train.txt"
    with open(train_txt_path, "w", encoding="utf-8") as f:
        for line in train_out_lines:
            f.write(line + "\n")
    print(f"Wrote {len(train_out_lines)} lines to {train_txt_path}")

    # -------------------------------------------------------------
    # Step B: Process Validation Data (Apply Frozen Training Stats)
    # -------------------------------------------------------------
    print("\n--- Processing Validation Data (665 utterances) ---")
    val_tg_files = sorted(VAL_TG_DIR.glob("*/*.TextGrid"))
    print(f"Found {len(val_tg_files)} validation TextGrids.")

    val_out_lines: list[str] = []
    val_discrepancies: list[int] = []
    val_reconciled_count = 0
    val_failed: list[tuple[str, str]] = []

    for tg_path in tqdm(val_tg_files, desc="Val Features"):
        uid = tg_path.stem
        spk_str = tg_path.parent.name
        wav_path = VAL_CORPUS_DIR / spk_str / f"{uid}.wav"

        res = extractor.extract_features_single(wav_path, tg_path)
        if res is None:
            val_failed.append((uid, "Feature extraction failed"))
            continue

        raw_text = manifest.get(uid, {}).get("raw_text", "")
        dataset_line = f"{uid}|{spk_str}|{res['fastspeech2_text']}|{raw_text}"
        val_out_lines.append(dataset_line)

        if res["reconciliation_applied"]:
            val_reconciled_count += 1
        val_discrepancies.append(res["discrepancy"])

        # Save arrays with TRAINING-DERIVED NORMALIZATION
        np.save(OUT_DIR / "mel" / f"{spk_str}-mel-{uid}.npy", res["mel"])
        np.save(OUT_DIR / "duration" / f"{spk_str}-duration-{uid}.npy", res["durations"])

        p_norm = (res["pitch_frame"] - pitch_frame_mean) / pitch_frame_std
        e_norm = (res["energy_frame"] - energy_frame_mean) / energy_frame_std
        np.save(OUT_DIR / "pitch" / f"{spk_str}-pitch-{uid}.npy", p_norm.astype(np.float32))
        np.save(OUT_DIR / "energy" / f"{spk_str}-energy-{uid}.npy", e_norm.astype(np.float32))

        p_ph_norm = (res["pitch_phoneme"] - pitch_mean) / pitch_std
        e_ph_norm = (res["energy_phoneme"] - energy_mean) / energy_std
        np.save(OUT_DIR / "pitch_phoneme" / f"{spk_str}-pitch-{uid}.npy", p_ph_norm.astype(np.float32))
        np.save(OUT_DIR / "energy_phoneme" / f"{spk_str}-energy-{uid}.npy", e_ph_norm.astype(np.float32))

    # Write val.txt
    val_txt_path = OUT_DIR / "val.txt"
    with open(val_txt_path, "w", encoding="utf-8") as f:
        for line in val_out_lines:
            f.write(line + "\n")
    print(f"Wrote {len(val_out_lines)} lines to {val_txt_path}")

    dt_total = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"Preprocessing Pipeline Completed in {dt_total:.2f} seconds!")
    print(f"Train samples saved: {len(train_out_lines)} / {len(train_tg_files)}")
    print(f"Val samples saved:   {len(val_out_lines)} / {len(val_tg_files)}")
    print(f"Reconciled files:    {train_reconciled_count} train, {val_reconciled_count} val")
    print("=" * 70)


if __name__ == "__main__":
    run_full_preprocessing()
