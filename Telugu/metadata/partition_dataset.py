"""Deterministic, leakage-safe dataset partitioning for IndicTTS Telugu.

Generates:
- Telugu/metadata/train.txt (5.00 hours total: ~2.50h Female + ~2.50h Male)
- Telugu/metadata/val.txt   (1.00 hour total: ~0.50h Female + ~0.50h Male)
- Telugu/metadata/test.txt  (Remaining clean held-out test data)
- Telugu/metadata/split_summary.json (Audit log with exclusion reasons & stats)

Constraints:
1. Strict duration filtering: 1.0s <= duration <= 15.0s to ensure FastSpeech 2
   stability and prevent GPU memory exhaustion.
2. Cross-speaker duplicate title isolation: eliminates shared story headings
   to guarantee zero text leakage across splits.
3. Deterministic pseudo-random seed (seed=42) for exact reproducibility.
4. Preserves original row indices, audio paths, and speaker IDs.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Any

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from common.paths import PROJECT_ROOT, TELUGU_DIR
from Telugu.text.normalizer import normalize_text

METADATA_DIR = TELUGU_DIR / "metadata"
TEXT_MANIFEST_PATH = METADATA_DIR / "text_manifest.json"

TRAIN_TXT = METADATA_DIR / "train.txt"
VAL_TXT = METADATA_DIR / "val.txt"
TEST_TXT = METADATA_DIR / "test.txt"
SUMMARY_JSON = METADATA_DIR / "split_summary.json"

# Calibrated reading speech rates (characters per second)
CHARS_PER_SEC_FEMALE = 8.80
CHARS_PER_SEC_MALE = 15.20

# Target hours per split
TARGET_TRAIN_HOURS_PER_SPEAKER = 2.50
TARGET_VAL_HOURS_PER_SPEAKER = 0.50


def partition_dataset(seed: int = 42) -> dict[str, Any]:
    print(f"Loading manifest from {TEXT_MANIFEST_PATH}...")
    with open(TEXT_MANIFEST_PATH, encoding="utf-8") as f:
        rows: list[dict[str, Any]] = json.load(f)

    # 1. Annotate durations and normalize text
    for r in rows:
        norm_t = normalize_text(r["text"])
        r["norm_text"] = norm_t
        rate = CHARS_PER_SEC_FEMALE if r["gender"] == 0 else CHARS_PER_SEC_MALE
        r["est_dur"] = len(norm_t) / rate

    # 2. Identify duplicates and cross-gender overlaps
    text_to_rows: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        text_to_rows.setdefault(r["norm_text"], []).append(r)

    clean_pool: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for norm_t, group in text_to_rows.items():
        if len(group) > 1:
            genders = set(x["gender"] for x in group)
            if len(genders) > 1:
                for item in group:
                    excluded.append({
                        "path": item["path"],
                        "row_idx": item["row_idx"],
                        "gender": item["gender"],
                        "duration": item["est_dur"],
                        "reason": "cross_gender_duplicate_title",
                        "text": item["text"],
                    })
                continue
            else:
                # Single gender duplicate: keep first, exclude remaining
                for item in group[1:]:
                    excluded.append({
                        "path": item["path"],
                        "row_idx": item["row_idx"],
                        "gender": item["gender"],
                        "duration": item["est_dur"],
                        "reason": "intra_gender_duplicate_transcript",
                        "text": item["text"],
                    })

        item = group[0]
        if item["est_dur"] < 1.0:
            excluded.append({
                "path": item["path"],
                "row_idx": item["row_idx"],
                "gender": item["gender"],
                "duration": item["est_dur"],
                "reason": "too_short_sub_1s",
                "text": item["text"],
            })
        elif item["est_dur"] > 15.0:
            excluded.append({
                "path": item["path"],
                "row_idx": item["row_idx"],
                "gender": item["gender"],
                "duration": item["est_dur"],
                "reason": "too_long_super_15s",
                "text": item["text"],
            })
        else:
            clean_pool.append(item)

    # 3. Partition within each speaker
    rng = random.Random(seed)

    female_pool = [r for r in clean_pool if r["gender"] == 0]
    male_pool = [r for r in clean_pool if r["gender"] == 1]

    rng.shuffle(female_pool)
    rng.shuffle(male_pool)

    def select_split(pool: list[dict[str, Any]], target_seconds: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        selected = []
        cum_dur = 0.0
        remaining = []
        for item in pool:
            if cum_dur + item["est_dur"] <= target_seconds or not selected:
                selected.append(item)
                cum_dur += item["est_dur"]
            else:
                remaining.append(item)
        return selected, remaining

    # Female split
    female_train, female_rem = select_split(female_pool, TARGET_TRAIN_HOURS_PER_SPEAKER * 3600)
    female_val, female_test = select_split(female_rem, TARGET_VAL_HOURS_PER_SPEAKER * 3600)

    # Male split
    male_train, male_rem = select_split(male_pool, TARGET_TRAIN_HOURS_PER_SPEAKER * 3600)
    male_val, male_test = select_split(male_rem, TARGET_VAL_HOURS_PER_SPEAKER * 3600)

    train_set = female_train + male_train
    val_set = female_val + male_val
    test_set = female_test + male_test

    # 4. Leakage Verification
    train_texts = set(r["norm_text"] for r in train_set)
    val_texts = set(r["norm_text"] for r in val_set)
    test_texts = set(r["norm_text"] for r in test_set)

    assert len(train_texts.intersection(val_texts)) == 0, "Leakage between train and val!"
    assert len(train_texts.intersection(test_texts)) == 0, "Leakage between train and test!"
    assert len(val_texts.intersection(test_texts)) == 0, "Leakage between val and test!"

    # 5. Write metadata files
    def write_split(path: Path, items: list[dict[str, Any]]) -> None:
        lines = []
        for r in items:
            basename = Path(r["path"]).stem
            speaker = str(r["gender"])
            raw = r["text"]
            norm = r["norm_text"]
            dur = f"{r['est_dur']:.4f}"
            gender_str = "female" if r["gender"] == 0 else "male"
            lines.append(f"{basename}|{speaker}|{raw}|{norm}|{dur}|{gender_str}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    write_split(TRAIN_TXT, train_set)
    write_split(VAL_TXT, val_set)
    write_split(TEST_TXT, test_set)

    # 6. Build summary
    summary = {
        "random_seed": seed,
        "total_source_utterances": len(rows),
        "clean_pool_utterances": len(clean_pool),
        "excluded_utterances": len(excluded),
        "exclusions_by_reason": {
            reason: sum(1 for e in excluded if e["reason"] == reason)
            for reason in set(e["reason"] for e in excluded)
        },
        "splits": {
            "train": {
                "total_utterances": len(train_set),
                "total_duration_hours": sum(r["est_dur"] for r in train_set) / 3600,
                "female_utterances": len(female_train),
                "female_duration_hours": sum(r["est_dur"] for r in female_train) / 3600,
                "male_utterances": len(male_train),
                "male_duration_hours": sum(r["est_dur"] for r in male_train) / 3600,
            },
            "val": {
                "total_utterances": len(val_set),
                "total_duration_hours": sum(r["est_dur"] for r in val_set) / 3600,
                "female_utterances": len(female_val),
                "female_duration_hours": sum(r["est_dur"] for r in female_val) / 3600,
                "male_utterances": len(male_val),
                "male_duration_hours": sum(r["est_dur"] for r in male_val) / 3600,
            },
            "test": {
                "total_utterances": len(test_set),
                "total_duration_hours": sum(r["est_dur"] for r in test_set) / 3600,
                "female_utterances": len(female_test),
                "female_duration_hours": sum(r["est_dur"] for r in female_test) / 3600,
                "male_utterances": len(male_test),
                "male_duration_hours": sum(r["est_dur"] for r in male_test) / 3600,
            },
        },
        "leakage_verification": {
            "train_val_overlap": len(train_texts.intersection(val_texts)),
            "train_test_overlap": len(train_texts.intersection(test_texts)),
            "val_test_overlap": len(val_texts.intersection(test_texts)),
            "is_leakage_safe": True,
        },
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n--- Partitioning Summary ---")
    print(f"Train: {len(train_set)} utterances, {summary['splits']['train']['total_duration_hours']:.4f}h "
          f"(F: {summary['splits']['train']['female_duration_hours']:.4f}h, M: {summary['splits']['train']['male_duration_hours']:.4f}h)")
    print(f"Val:   {len(val_set)} utterances, {summary['splits']['val']['total_duration_hours']:.4f}h "
          f"(F: {summary['splits']['val']['female_duration_hours']:.4f}h, M: {summary['splits']['val']['male_duration_hours']:.4f}h)")
    print(f"Test:  {len(test_set)} utterances, {summary['splits']['test']['total_duration_hours']:.4f}h "
          f"(F: {summary['splits']['test']['female_duration_hours']:.4f}h, M: {summary['splits']['test']['male_duration_hours']:.4f}h)")
    print("Zero text leakage verified across all splits!")

    return summary


if __name__ == "__main__":
    partition_dataset(seed=42)
