"""Extract and prepare audio and transcripts for full Svarah train and val sets."""

from __future__ import annotations

import glob
import io
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio.functional as AF
from tqdm import tqdm

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from common.paths import DATA_DIR, ENGLISH_DIR, PROJECT_ROOT

MANIFEST_PATH = ENGLISH_DIR / "metadata" / "full_manifest.json"

TRAIN_CORPUS_DIR = DATA_DIR / "svarah_train" / "corpus"
VAL_CORPUS_DIR = DATA_DIR / "svarah_val" / "corpus"

TARGET_SR = 22050


def prepare_full_corpora() -> dict[str, int]:
    """Read cached parquet shards and extract audio and lab files for train & val."""
    t0 = time.time()
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    # Filter to train and val only
    train_entries: dict[str, dict] = {}
    val_entries: dict[str, dict] = {}

    for entry in manifest:
        split = entry.get("split")
        status = entry.get("status")
        uid = entry.get("utterance_id")
        if status != "valid":
            continue
        if split == "train":
            train_entries[uid] = entry
        elif split == "val":
            val_entries[uid] = entry

    print(f"Target utterances: {len(train_entries)} train, {len(val_entries)} val")

    # Group by shard
    shard_map: dict[str, dict[str, dict]] = {
        "test-00000-of-00003.parquet": {},
        "test-00001-of-00003.parquet": {},
        "test-00002-of-00003.parquet": {},
    }

    for uid, entry in {**train_entries, **val_entries}.items():
        s = entry["source"]["shard"]
        if s in shard_map:
            shard_map[s][uid] = entry
        else:
            raise ValueError(f"Unknown shard {s} for utterance {uid}")

    counts = {"train": 0, "val": 0}

    # Locate cached shards on disk
    cache_glob = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--ai4bharat--Svarah/snapshots/*/data/*.parquet"
    )
    cached_files = {os.path.basename(p): p for p in glob.glob(cache_glob)}

    for shard_name, entries in shard_map.items():
        if shard_name not in cached_files:
            raise FileNotFoundError(f"Cached shard {shard_name} not found in {cache_glob}")

        shard_path = cached_files[shard_name]
        print(f"\nProcessing shard {shard_name} ({len(entries)} target utterances)...")

        pf = pq.ParquetFile(shard_path)
        num_row_groups = pf.metadata.num_row_groups

        for rg_idx in range(num_row_groups):
            tbl = pf.read_row_group(rg_idx, columns=["audio_filepath"])
            col = tbl.column("audio_filepath")

            for i in range(tbl.num_rows):
                item = col[i].as_py()
                path_str = item["path"]
                uid = path_str.replace(".wav", "")

                if uid not in entries:
                    continue

                meta = entries[uid]
                split = meta["split"]
                prof_id = meta["speaker_profile_id"]
                spk_str = f"spk_{prof_id:03d}"
                raw_bytes = item["bytes"]

                # Determine target directory
                out_dir = TRAIN_CORPUS_DIR if split == "train" else VAL_CORPUS_DIR
                spk_dir = out_dir / spk_str
                spk_dir.mkdir(parents=True, exist_ok=True)

                out_wav = spk_dir / f"{uid}.wav"
                out_lab = spk_dir / f"{uid}.lab"

                # 1. Read and resample audio
                audio, orig_sr = sf.read(
                    io.BytesIO(raw_bytes), dtype="float32", always_2d=True
                )
                mono = torch.from_numpy(audio.mean(axis=1))

                if orig_sr != TARGET_SR:
                    y_t = AF.resample(
                        mono,
                        orig_freq=orig_sr,
                        new_freq=TARGET_SR,
                        lowpass_filter_width=64,
                        rolloff=0.9475937167399596,
                        resampling_method="sinc_interp_kaiser",
                        beta=14.769656459379492,
                    )
                    y_np = y_t.numpy()
                else:
                    y_np = mono.numpy()

                # Peak normalization to prevent clipping
                peak = np.abs(y_np).max()
                if peak > 0.95:
                    y_np = y_np * (0.95 / peak)

                # Write WAV (22050 Hz 16-bit PCM)
                sf.write(str(out_wav), y_np, TARGET_SR, subtype="PCM_16")

                # 2. Write normalized transcript
                norm_text = meta["normalized_text"].strip()
                out_lab.write_text(norm_text, encoding="utf-8")

                counts[split] += 1

    dt = time.time() - t0
    print(f"\nFinished preparing corpora in {dt:.2f}s!")
    print(f"Total extracted: {counts['train']} train, {counts['val']} val")
    return counts


if __name__ == "__main__":
    prepare_full_corpora()
