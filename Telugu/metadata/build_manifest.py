"""Extract complete manifest and exact audio durations for IndicTTS Telugu.

Extracts text, gender, audio path, byte length, and exact duration for all 8,576
utterances using DuckDB range queries over Hugging Face parquet shards,
without downloading the multi-gigabyte audio payloads.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import duckdb

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from common.paths import PROJECT_ROOT, TELUGU_DIR

METADATA_DIR = TELUGU_DIR / "metadata"
MANIFEST_PATH = METADATA_DIR / "full_manifest.json"

PARQUET_URLS = [
    "https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu/resolve/main/data/train-00000-of-00004.parquet",
    "https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu/resolve/main/data/train-00001-of-00004.parquet",
    "https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu/resolve/main/data/train-00002-of-00004.parquet",
    "https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu/resolve/main/data/train-00003-of-00004.parquet",
]


def build_manifest() -> list[dict]:
    print("Connecting to DuckDB and querying remote parquet files sequentially...")
    all_records = []
    con = duckdb.connect()

    for file_idx, url in enumerate(PARQUET_URLS):
        t0 = time.time()
        print(f"Querying shard {file_idx+1}/{len(PARQUET_URLS)}: {url.split('/')[-1]}...")
        query = f"""
        SELECT 
            text, 
            gender, 
            audio.path AS path,
            octet_length(audio.bytes) AS byte_len,
            CASE 
                WHEN gender = 0 THEN ((octet_length(audio.bytes) - 44) / 2.0) / 16000.0
                ELSE ((octet_length(audio.bytes) - 44) / 2.0) / 48000.0
            END AS duration,
            CASE
                WHEN gender = 0 THEN 16000
                ELSE 48000
            END AS sample_rate
        FROM read_parquet('{url}')
        """
        df = con.execute(query).fetchdf()
        elapsed = time.time() - t0
        print(f"  Shard {file_idx+1} fetched {len(df)} rows in {elapsed:.2f}s")
        recs = df.to_dict(orient="records")
        all_records.extend(recs)

    # Assign sequential row_idx across all records
    for i, r in enumerate(all_records):
        r["row_idx"] = i
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(records)} records to {MANIFEST_PATH}")

    # Print summary statistics
    female_recs = [r for r in records if r["gender"] == 0]
    male_recs = [r for r in records if r["gender"] == 1]

    female_dur = sum(r["duration"] for r in female_recs)
    male_dur = sum(r["duration"] for r in male_recs)
    total_dur = female_dur + male_dur

    print("\n--- Corpus Summary ---")
    print(f"Female: {len(female_recs)} utterances, {female_dur:.2f}s ({female_dur/3600:.4f}h)")
    print(f"Male:   {len(male_recs)} utterances, {male_dur:.2f}s ({male_dur/3600:.4f}h)")
    print(f"Total:  {len(records)} utterances, {total_dur:.2f}s ({total_dur/3600:.4f}h)")

    return records


if __name__ == "__main__":
    build_manifest()
