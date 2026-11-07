#!/usr/bin/env python3
"""Phase 12 & Phase 13 Dry-Run Cleanup Verifier for FastSpeech 2 Project.

Strictly Read-Only:
- Lists all approved cleanup candidate paths across Tiers 1 to 5.
- Computes and displays exact byte sizes and SHA-256 checksums.
- Asserts that strictly protected directories (data/, preprocessed_data/,
  original_fastspeech2/preprocessed_data/, .venv/) have ZERO targeted files.
- Asserts that all 62 mandatory assignment deliverables, authoritative checkpoints,
  and explicitly retained files (Telugu_IndicTTS_Features.tar.gz,
  Results/English/audio/diagnostic_test.wav, SOURCE_ARCHIVE_MANIFEST.txt)
  have ZERO targeted files.
- Prints proposed actions and total recoverable disk space without modifying any file.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_BOOTSTRAP_ROOT = Path(__file__).resolve().parent.parent
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from common.paths import DOCUMENTATION_DIR, PROJECT_ROOT

BACKUP_ROOT = Path.home() / "Downloads" / "FastSpeech2_Cleanup_Backup"
MANIFEST_JSON = DOCUMENTATION_DIR / "final_artifact_manifest.json"

PROTECTED_CHECKPOINTS_AND_RETAINED = {
    "checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar",
    "checkpoints/Svarah_profile_stratified_v1/checkpoint_latest.pth.tar",
    "checkpoints/IndicTTS/checkpoint_best.pth.tar",
    "checkpoints/IndicTTS/checkpoint_latest.pth.tar",
    "original_fastspeech2/hifigan/generator_universal.pth.tar",
    "Telugu_IndicTTS_Features.tar.gz",
    "Results/English/audio/diagnostic_test.wav",
    "SOURCE_ARCHIVE_MANIFEST.txt",
}

APPROVED_CANDIDATE_FILES = [
    # Tier 1: Byte-identical duplicate checkpoints & stray pilot OOV text files
    ("Tier 1", "BACKUP_AND_DELETE_DUPLICATE", "output/ckpt/Svarah_profile_stratified_v1/checkpoint_best.pth.tar"),
    ("Tier 1", "BACKUP_AND_DELETE_DUPLICATE", "output/ckpt/Svarah_profile_stratified_v1/checkpoint_latest.pth.tar"),
    ("Tier 1", "BACKUP_AND_DELETE", "--output_path/oov_counts_pilot_dictionary.txt"),
    ("Tier 1", "BACKUP_AND_DELETE", "--output_path/oovs_found_pilot_dictionary.txt"),
    ("Tier 1", "BACKUP_AND_DELETE", "--output_path/utterance_oovs.txt"),
    # Tier 2: Superseded Pilot / Mock Checkpoint Backups
    ("Tier 2", "BACKUP_AND_DELETE", "checkpoints/Svarah_profile_stratified_v1/checkpoint_5000.pth.tar"),
    ("Tier 2", "BACKUP_AND_DELETE", "checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar.mock_bak"),
    ("Tier 2", "BACKUP_AND_DELETE", "checkpoints/Svarah_profile_stratified_v1/checkpoint_best_diagnostic_mock.pth.tar.bak"),
    ("Tier 2", "BACKUP_AND_DELETE", "checkpoints/Svarah_profile_stratified_v1/checkpoint_latest.pth.tar.mock_bak"),
    # Tier 3: Approved Old Deployment Archive Bundles & Redundant HiFi-GAN Zips
    # (Telugu_IndicTTS_Features.tar.gz and generator_universal.pth.tar are KEPT)
    ("Tier 3", "BACKUP_AND_DELETE", "FastSpeech2_Colab_Bundle_profile_stratified_v1.tar.gz"),
    ("Tier 3", "BACKUP_AND_DELETE", "FastSpeech2_Colab_Bundle.tar.gz"),
    ("Tier 3", "BACKUP_AND_DELETE", "English_inference_full_assets.zip"),
    ("Tier 3", "BACKUP_AND_DELETE", "original_fastspeech2/hifigan/generator_universal.pth.tar.zip"),
    ("Tier 3", "BACKUP_AND_DELETE", "original_fastspeech2/hifigan/generator_LJSpeech.pth.tar.zip"),
    ("Tier 3", "BACKUP_AND_DELETE", "FastSpeech2_Source_Code.zip"),
    ("Tier 3", "BACKUP_AND_DELETE", "FastSpeech2_Colab_Code_Update_profile_stratified_v1.tar.gz"),
    ("Tier 3", "BACKUP_AND_DELETE", "Telugu_Colab_Bundle.tar.gz"),
    ("Tier 3", "BACKUP_AND_DELETE", "English_inference_only.zip"),
    # Tier 4: Telugu Intermediate Periodic Checkpoints (Steps 16,000 - 20,000)
    ("Tier 4", "ARCHIVE_TO_BACKUP", "checkpoints/IndicTTS/checkpoint_16000.pth.tar"),
    ("Tier 4", "ARCHIVE_TO_BACKUP", "checkpoints/IndicTTS/checkpoint_17000.pth.tar"),
    ("Tier 4", "ARCHIVE_TO_BACKUP", "checkpoints/IndicTTS/checkpoint_18000.pth.tar"),
    ("Tier 4", "ARCHIVE_TO_BACKUP", "checkpoints/IndicTTS/checkpoint_19000.pth.tar"),
    ("Tier 4", "ARCHIVE_TO_BACKUP", "checkpoints/IndicTTS/checkpoint_20000.pth.tar"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def dir_size_and_count(path: Path) -> tuple[int, int]:
    files = [f for f in path.rglob("*") if f.is_file() and not f.is_symlink()]
    return sum(f.stat().st_size for f in files), len(files)


def main() -> int:
    print("=" * 100)
    print("FASTSPEECH 2 — APPROVED STAGED DRY-RUN CLEANUP VERIFICATION (READ-ONLY MODE)")
    print("=" * 100)
    print(f"Project Root : {PROJECT_ROOT}")
    print(f"Backup Target: {BACKUP_ROOT}")
    print("Mode         : DRY-RUN ONLY (No files will be deleted, moved, or modified)\n")

    mandatory_deliverables: set[str] = set()
    if MANIFEST_JSON.is_file():
        with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
            manifest_items = json.load(f)
            for item in manifest_items:
                mandatory_deliverables.add(item["relative_path"])

    # 1. Verify Safety Guardrails on Candidate Files
    print("[1/3] Verifying Protection Guardrails...")
    for _, _, rel_path in APPROVED_CANDIDATE_FILES:
        for prefix in ("data/", "preprocessed_data/", "original_fastspeech2/preprocessed_data/", ".venv/"):
            if rel_path.startswith(prefix):
                raise RuntimeError(f"SAFETY VIOLATION: Protected path targeted: {rel_path}")
        if rel_path in PROTECTED_CHECKPOINTS_AND_RETAINED:
            raise RuntimeError(f"SAFETY VIOLATION: Protected/retained file targeted: {rel_path}")
        if rel_path in mandatory_deliverables:
            raise RuntimeError(f"SAFETY VIOLATION: Mandatory deliverable targeted: {rel_path}")

    for prot_rel in sorted(PROTECTED_CHECKPOINTS_AND_RETAINED):
        p = PROJECT_ROOT / prot_rel
        if not p.is_file():
            raise RuntimeError(f"Missing protected file: {prot_rel}")
        print(f"  [PROTECTED & VERIFIED] {prot_rel} ({p.stat().st_size:,} B)")

    for pdir in ("data", "preprocessed_data", "original_fastspeech2/preprocessed_data"):
        dp = PROJECT_ROOT / pdir
        sz, cnt = dir_size_and_count(dp)
        print(f"  [PROTECTED & UNTOUCHED] {pdir}/ ({cnt:,} files, {sz / (1024**2):.2f} MiB)")

    print(f"  [PROTECTED & UNTOUCHED] All {len(mandatory_deliverables)} manifest deliverables verified safe.\n")

    # 2. Inspect Individual Approved Candidate Files
    print("[2/3] Approved File Actions & Checksums:")
    print("-" * 100)
    print(f"{'Tier':6s} | {'Action':28s} | {'Size (MiB)':>10s} | {'SHA-256 (First 16)':18s} | {'Relative Path'}")
    print("-" * 100)

    total_bytes = 0
    tier_totals: dict[str, int] = {}

    for tier, action, rel_path in APPROVED_CANDIDATE_FILES:
        p = PROJECT_ROOT / rel_path
        if not p.exists():
            print(f"{tier:6s} | {'ALREADY_MOVED':28s} | {'0.00':>10s} | {'N/A':18s} | {rel_path}")
            continue
        sz = p.stat().st_size
        sha = sha256_file(p)
        total_bytes += sz
        tier_totals[tier] = tier_totals.get(tier, 0) + sz
        print(f"{tier:6s} | {action:28s} | {sz / (1024**2):10.2f} | {sha[:16]:18s} | {rel_path}")

    # Directory-level candidates (Caches, .DS_Store outside protected dirs, original_fastspeech2/.git)
    pycache_dirs = [
        p for p in PROJECT_ROOT.rglob("__pycache__")
        if ".venv" not in p.parts and "data" not in p.parts and "preprocessed_data" not in p.parts
    ]
    pyc_bytes = sum(dir_size_and_count(d)[0] for d in pycache_dirs)
    pytest_dir = PROJECT_ROOT / ".pytest_cache"
    pytest_bytes = dir_size_and_count(pytest_dir)[0] if pytest_dir.exists() else 0
    ds_files = [
        p for p in PROJECT_ROOT.rglob(".DS_Store")
        if ".venv" not in p.parts and "data" not in p.parts and "preprocessed_data" not in p.parts
    ]
    ds_bytes = sum(f.stat().st_size for f in ds_files)
    cache_total = pyc_bytes + pytest_bytes + ds_bytes
    total_bytes += cache_total
    tier_totals["Tier 1"] = tier_totals.get("Tier 1", 0) + cache_total
    print(f"{'Tier 1':6s} | {'DELETE_CACHES':28s} | {cache_total / (1024**2):10.2f} | {f'({len(pycache_dirs)} pycache, {len(ds_files)} DS)':18s} | __pycache__/, .pytest_cache/, .DS_Store")

    nested_git = PROJECT_ROOT / "original_fastspeech2" / ".git"
    if nested_git.is_dir():
        git_bytes, git_cnt = dir_size_and_count(nested_git)
        total_bytes += git_bytes
        tier_totals["Tier 5"] = tier_totals.get("Tier 5", 0) + git_bytes
        print(f"{'Tier 5':6s} | {'MOVE_TO_BACKUP':28s} | {git_bytes / (1024**2):10.2f} | {f'({git_cnt} git files)':18s} | original_fastspeech2/.git/")

    print("-" * 100)
    print("\n[3/3] Summary of Approved Recoverable Space by Tier:")
    for t_name in sorted(tier_totals.keys()):
        t_sz = tier_totals[t_name]
        print(f"  - {t_name}: {t_sz:14,d} bytes ({t_sz / (1024**2):8.2f} MiB / {t_sz / (1024**3):5.2f} GiB)")
    print("=" * 100)
    print(f"TOTAL APPROVED RECOVERABLE SPACE: {total_bytes:,} bytes ({total_bytes / (1024**2):.2f} MiB / {total_bytes / (1024**3):.2f} GiB)")
    print("STATUS: DRY-RUN COMPLETE. ZERO FILES WERE MODIFIED OR DELETED.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
