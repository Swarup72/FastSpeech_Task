"""Utility script to package a clean, minimal Google Drive upload bundle for Colab training.

Excludes:
    - .venv/, __pycache__/, .pytest_cache/, .git/
    - Original raw WAV audio files (data/svarah_train/corpus/*.wav)
    - Telugu pipeline files and MFA models
    - Unused demo/ and LJSpeech sample files in original_fastspeech2/

Includes only what English FastSpeech 2 training requires under FastSpeech2_Training/:
    - project/common/
    - project/config/Svarah/
    - project/English/ (train.py, monitoring.py, __init__.py, text/)
    - project/original_fastspeech2/ (dataset.py, text/, utils/)
    - preprocessed_data/Svarah/ (train.txt, val.txt, speakers.json, stats.json, mel/, pitch/, energy/, duration/)
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path

_BOOTSTRAP_ROOT = Path(__file__).resolve().parent.parent
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from common.paths import PROJECT_ROOT, resolve_project_path

EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".git", ".venv", ".DS_Store"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".wav"}


def _should_include(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDED_DIRS or part.startswith("._"):
            return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return True


def create_colab_bundle(output_archive: Path, include_features: bool = True) -> dict:
    output_archive = output_archive.resolve()
    output_archive.parent.mkdir(parents=True, exist_ok=True)

    project_items = [
        ("common", "FastSpeech2_Training/project/common"),
        ("config/Svarah", "FastSpeech2_Training/project/config/Svarah"),
        ("English/__init__.py", "FastSpeech2_Training/project/English/__init__.py"),
        ("English/train.py", "FastSpeech2_Training/project/English/train.py"),
        ("English/monitoring.py", "FastSpeech2_Training/project/English/monitoring.py"),
        ("English/text", "FastSpeech2_Training/project/English/text"),
        (
            "English/metadata/splits/profile_stratified_v1",
            "FastSpeech2_Training/project/English/metadata/splits/profile_stratified_v1",
        ),
        (
            "original_fastspeech2/dataset.py",
            "FastSpeech2_Training/project/original_fastspeech2/dataset.py",
        ),
        (
            "original_fastspeech2/text",
            "FastSpeech2_Training/project/original_fastspeech2/text",
        ),
        (
            "original_fastspeech2/utils",
            "FastSpeech2_Training/project/original_fastspeech2/utils",
        ),
    ]

    feature_items = [
        ("preprocessed_data/Svarah/train.txt", "FastSpeech2_Training/preprocessed_data/Svarah/train.txt"),
        ("preprocessed_data/Svarah/val.txt", "FastSpeech2_Training/preprocessed_data/Svarah/val.txt"),
        ("preprocessed_data/Svarah/speakers.json", "FastSpeech2_Training/preprocessed_data/Svarah/speakers.json"),
        ("preprocessed_data/Svarah/stats.json", "FastSpeech2_Training/preprocessed_data/Svarah/stats.json"),
        ("preprocessed_data/Svarah/mel", "FastSpeech2_Training/preprocessed_data/Svarah/mel"),
        ("preprocessed_data/Svarah/pitch", "FastSpeech2_Training/preprocessed_data/Svarah/pitch"),
        ("preprocessed_data/Svarah/energy", "FastSpeech2_Training/preprocessed_data/Svarah/energy"),
        ("preprocessed_data/Svarah/duration", "FastSpeech2_Training/preprocessed_data/Svarah/duration"),
    ]

    all_items = list(project_items)
    if include_features:
        all_items.extend(feature_items)

    file_count = 0
    with tarfile.open(output_archive, "w:gz") as tar:
        for rel_src, arc_prefix in all_items:
            src_path = PROJECT_ROOT / rel_src
            if not src_path.exists():
                raise FileNotFoundError(f"Required source path missing: {src_path}")
            if src_path.is_file():
                if _should_include(src_path):
                    tar.add(src_path, arcname=arc_prefix)
                    file_count += 1
            else:
                for root, dirs, files in os.walk(src_path):
                    dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith("._")]
                    for fname in sorted(files):
                        fpath = Path(root) / fname
                        if _should_include(fpath):
                            rel_sub = fpath.relative_to(src_path)
                            arcname = f"{arc_prefix}/{rel_sub.as_posix()}"
                            tar.add(fpath, arcname=arcname)
                            file_count += 1

    size_mb = round(output_archive.stat().st_size / (1024 * 1024), 2)
    return {
        "archive_path": str(output_archive),
        "file_count": file_count,
        "size_mb": size_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create minimal Google Drive bundle for English FastSpeech 2 Colab training"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "FastSpeech2_Colab_Bundle.tar.gz"),
        help="Output .tar.gz path",
    )
    parser.add_argument(
        "--code_only",
        action="store_true",
        help="Package only the lightweight project code (~200 KB) without preprocessed .npy features",
    )
    args = parser.parse_args()
    info = create_colab_bundle(Path(args.output), include_features=not args.code_only)
    print(f"Created Colab bundle: {info['archive_path']} ({info['size_mb']} MB, {info['file_count']} files)")


if __name__ == "__main__":
    main()

