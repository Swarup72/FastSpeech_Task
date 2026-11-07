"""Download and prepare the 100 pilot training utterances for MFA alignment."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio.functional as AF
from huggingface_hub import HfFileSystem

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from common.paths import DATA_DIR as ROOT_DATA_DIR, ENGLISH_DIR, PROJECT_ROOT

PILOT_JSON = ENGLISH_DIR / "metadata" / "pilot_train_100.json"
DATA_DIR = ROOT_DATA_DIR / "svarah_pilot_100"
RAW_DIR = DATA_DIR / "raw_audio"
CORPUS_DIR = DATA_DIR / "corpus"

RAW_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

with open(PILOT_JSON, encoding="utf-8") as f:
    pilot_records = json.load(f)

target_uids = {r["utterance_id"]: r for r in pilot_records}
print(f"Loaded {len(target_uids)} target pilot training utterances.")

# Connect to Hugging Face Hub
fs = HfFileSystem()
REPO = "datasets/ai4bharat/Svarah@ebbf7777fe771490696a3f7b007097606fa8c924"
SHARD = "data/test-00000-of-00003.parquet"
ROW_GROUPS = [0, 1, 2, 3]

print(f"Reading row groups {ROW_GROUPS} from {SHARD}...")
pf = pq.ParquetFile(fs.open(f"{REPO}/{SHARD}", "rb", block_size=2**20))
tbl = pf.read_row_groups(ROW_GROUPS)

df_rows = tbl.drop(["audio_filepath"]).to_pandas()
df_rows["audio_bytes"] = [a["bytes"] for a in tbl.column("audio_filepath").to_pylist()]
df_rows["audio_path"] = [a["path"] for a in tbl.column("audio_filepath").to_pylist()]
df_rows["utterance_id"] = df_rows["audio_path"].str.replace(".wav", "", regex=False)

print(f"Read {len(df_rows)} rows from shard.")

saved_count = 0
resample_manifest = []

for _, row in df_rows.iterrows():
    uid = row["utterance_id"]
    if uid not in target_uids:
        continue

    meta = target_uids[uid]
    prof_str = f"spk_{meta['speaker_profile_id']:03d}"
    spk_dir = CORPUS_DIR / prof_str
    spk_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save raw audio
    raw_path = RAW_DIR / f"{uid}.wav"
    raw_path.write_bytes(row["audio_bytes"])

    # 2. Inspect and resample from 16,000 Hz to 22,050 Hz
    x, sr = sf.read(str(raw_path), dtype="float32", always_2d=True)
    mono = x.mean(axis=1)

    # Resample using windowed-sinc Kaiser filter (22050 Hz)
    w_t = torch.from_numpy(mono)
    y_t = AF.resample(
        w_t,
        orig_freq=sr,
        new_freq=22050,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )
    y_np = y_t.numpy()

    # Peak normalization to prevent clipping on 16-bit PCM write
    peak = np.abs(y_np).max()
    if peak > 0.95:
        y_np = y_np * (0.95 / peak)

    # Save resampled WAV into MFA corpus directory
    out_wav = spk_dir / f"{uid}.wav"
    sf.write(str(out_wav), y_np, 22050, subtype="PCM_16")

    # 3. Write normalized .lab transcript file
    out_lab = spk_dir / f"{uid}.lab"
    out_lab.write_text(meta["normalized_text"].strip(), encoding="utf-8")

    resample_manifest.append({
        "utterance_id": uid,
        "speaker_profile_id": meta["speaker_profile_id"],
        "raw_duration": float(row["duration"]),
        "resampled_duration": round(len(y_np) / 22050.0, 4),
        "raw_path": str(raw_path.relative_to(PROJECT_ROOT)),
        "corpus_wav_path": str(out_wav.relative_to(PROJECT_ROOT)),
        "corpus_lab_path": str(out_lab.relative_to(PROJECT_ROOT)),
        "normalized_text": meta["normalized_text"],
    })
    saved_count += 1

print(f"\nSuccessfully prepared {saved_count} pilot training utterances in {CORPUS_DIR}")
assert saved_count == len(target_uids), f"Expected {len(target_uids)} utterances, saved {saved_count}"

manifest_df = pd.DataFrame(resample_manifest)
manifest_df.to_csv(DATA_DIR / "prepared_manifest.csv", index=False)
print(f"Saved preparation manifest to {DATA_DIR / 'prepared_manifest.csv'}")

