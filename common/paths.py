"""Centralized project root discovery and path management (`common.paths`).

Provides a single source of truth for all filesystem paths across the FastSpeech 2
English (Svarah) and Telugu (IndicTTS) pipelines.

Design guarantees:
1. Never relies on ``os.getcwd()`` or ``Path.cwd()`` for project root discovery.
2. Uses only Python standard library modules (no PyTorch, NumPy, or ML imports).
3. Resolves all relative configuration, dataset, checkpoint, and output paths
   deterministically against the project root while preserving explicit absolute
   paths and Google Colab mount overrides.
4. Validates required input paths with clear error messages and never silently
   creates missing dataset or checkpoint directories.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_SENTINEL_DIRS: tuple[str, ...] = ("common", "config")
_COLAB_FALLBACK_ROOTS: tuple[str, ...] = (
    "/content/FastSpeech2_Project",
    "/content/drive/MyDrive/FastSpeech2_Training/project",
    "/content/drive/MyDrive/FastSpeech2_Project",
)


def _is_valid_project_root(candidate: Path) -> bool:
    """Return True if ``candidate`` contains the core project sentinel directories."""
    if not candidate.is_dir():
        return False
    return all((candidate / name).is_dir() for name in _SENTINEL_DIRS)


def get_project_root() -> Path:
    """Discover and return the absolute project root path, independent of CWD.

    Resolution order:
    1. ``FASTSPEECH2_PROJECT_ROOT`` environment variable (if set and valid).
    2. Upward parent traversal from ``Path(__file__).resolve()`` checking for
       sentinel directories (``common/`` and ``config/``).
    3. Known Google Colab extraction roots (if running in a Colab container).

    Raises:
        FileNotFoundError: If no valid project root can be located.
    """
    env_override = os.environ.get("FASTSPEECH2_PROJECT_ROOT")
    if env_override:
        env_path = Path(env_override).expanduser().resolve()
        if _is_valid_project_root(env_path):
            return env_path
        raise FileNotFoundError(
            f"FASTSPEECH2_PROJECT_ROOT={env_override!r} does not point to a valid "
            f"FastSpeech 2 project root (missing {_SENTINEL_DIRS} in {env_path})."
        )

    here = Path(__file__).resolve()
    for parent in here.parents:
        if _is_valid_project_root(parent):
            return parent

    for colab_root in _COLAB_FALLBACK_ROOTS:
        candidate = Path(colab_root)
        if _is_valid_project_root(candidate):
            return candidate.resolve()

    raise FileNotFoundError(
        f"Unable to locate FastSpeech 2 project root starting from {here}. "
        "Ensure 'common/' and 'config/' exist at the project root or set "
        "FASTSPEECH2_PROJECT_ROOT."
    )


# ==============================================================================
# Canonical Project Directory & File Constants
# ==============================================================================

PROJECT_ROOT: Path = get_project_root()

COMMON_DIR: Path = PROJECT_ROOT / "common"
ENGLISH_DIR: Path = PROJECT_ROOT / "English"
TELUGU_DIR: Path = PROJECT_ROOT / "Telugu"
ORIGINAL_FS2_DIR: Path = PROJECT_ROOT / "original_fastspeech2"
CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATA_DIR: Path = PROJECT_ROOT / "data"
PREPROCESSED_DIR: Path = PROJECT_ROOT / "preprocessed_data"
CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"

VOCODER_DIR: Path = COMMON_DIR / "vocoder"
HIFIGAN_DIR: Path = (
    CHECKPOINTS_DIR / "hifigan"
    if (CHECKPOINTS_DIR / "hifigan").is_dir()
    else ORIGINAL_FS2_DIR / "hifigan"
)
HIFIGAN_CONFIG_PATH: Path = (
    CONFIG_DIR / "hifigan.json"
    if (CONFIG_DIR / "hifigan.json").is_file()
    else (
        VOCODER_DIR / "config.json"
        if (VOCODER_DIR / "config.json").is_file()
        else ORIGINAL_FS2_DIR / "hifigan" / "config.json"
    )
)
HIFIGAN_WEIGHTS_PATH: Path = (
    CHECKPOINTS_DIR / "hifigan" / "generator_universal.pth.tar"
    if (CHECKPOINTS_DIR / "hifigan" / "generator_universal.pth.tar").is_file()
    else ORIGINAL_FS2_DIR / "hifigan" / "generator_universal.pth.tar"
)
HIFIGAN_WEIGHTS_ZIP_PATH: Path = (
    CHECKPOINTS_DIR / "hifigan" / "generator_universal.pth.tar.zip"
    if (CHECKPOINTS_DIR / "hifigan" / "generator_universal.pth.tar.zip").is_file()
    else ORIGINAL_FS2_DIR / "hifigan" / "generator_universal.pth.tar.zip"
)
LEXICON_PATH: Path = (
    ENGLISH_DIR / "lexicon" / "librispeech-lexicon.txt"
    if (ENGLISH_DIR / "lexicon" / "librispeech-lexicon.txt").is_file()
    else ORIGINAL_FS2_DIR / "lexicon" / "librispeech-lexicon.txt"
)

RESULTS_DIR: Path = PROJECT_ROOT / "Results"
ENGLISH_RESULTS_DIR: Path = RESULTS_DIR / "English"
TELUGU_RESULTS_DIR: Path = RESULTS_DIR / "Telugu"

OUTPUT_DIR: Path = PROJECT_ROOT / "output"
REPORT_DIR: Path = PROJECT_ROOT / "Report"
REPORT_FIGURES_DIR: Path = REPORT_DIR / "figures"
DOCUMENTATION_DIR: Path = PROJECT_ROOT / "documentation"
RESEARCH_DIR: Path = PROJECT_ROOT / "research"
SCRATCH_DIR: Path = PROJECT_ROOT / "scratch"
TESTS_DIR: Path = PROJECT_ROOT / "tests"


# ==============================================================================
# Python Import Path Registration & Path Resolution
# ==============================================================================


def setup_project_paths(*, include_original_fs2: bool = True) -> Path:
    """Idempotently register ``PROJECT_ROOT`` (and ``original_fastspeech2/``) on ``sys.path``."""
    root = get_project_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    if include_original_fs2:
        orig_str = str(root / "original_fastspeech2")
        if orig_str not in sys.path:
            # Place right after PROJECT_ROOT so top-level packages take precedence
            insert_idx = 1 if sys.path and sys.path[0] == root_str else 0
            sys.path.insert(insert_idx, orig_str)

    return root


def resolve_project_path(
    path: str | Path,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Resolve a path deterministically against ``PROJECT_ROOT`` (or ``base_dir``).

    - Explicit absolute paths (including Google Colab ``/content/drive/...`` or
      temporary test directories) are preserved and returned with ``os.path.normpath``.
    - Relative paths (e.g. ``"./preprocessed_data/Svarah"``, ``"config/IndicTTS/model.yaml"``)
      are resolved against ``base_dir`` if provided, otherwise ``get_project_root()``.
      Never uses ``os.getcwd()``.
    """
    if path is None or str(path).strip() == "":
        raise ValueError("Path cannot be empty or None.")
    p = Path(path).expanduser()
    if p.is_absolute():
        return Path(os.path.normpath(str(p)))
    anchor = base_dir.resolve() if base_dir is not None else get_project_root()
    return Path(os.path.normpath(str(anchor / p)))


def validate_required_path(
    path: str | Path,
    description: str = "Required path",
    *,
    must_be_file: bool = False,
    must_be_dir: bool = False,
) -> Path:
    """Validate that a required path exists without ever creating missing directories.

    Args:
        path: Path to validate (relative paths are resolved against ``PROJECT_ROOT``).
        description: Human-readable label included in error messages.
        must_be_file: If True, require ``resolved.is_file()``.
        must_be_dir: If True, require ``resolved.is_dir()``.

    Returns:
        Resolved ``Path`` object.

    Raises:
        FileNotFoundError: If the path does not exist or fails type checks.
    """
    resolved = resolve_project_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{description} not found at: {resolved}")
    if must_be_file and not resolved.is_file():
        raise FileNotFoundError(f"{description} is not a regular file: {resolved}")
    if must_be_dir and not resolved.is_dir():
        raise FileNotFoundError(f"{description} is not a directory: {resolved}")
    return resolved


def ensure_output_dir(path: str | Path) -> Path:
    """Resolve and create an output directory when needed.

    Refuses to silently create the raw ``data/`` or ``preprocessed_data/`` roots
    as a safeguard against masking missing dataset errors.
    """
    resolved = resolve_project_path(path)
    root = get_project_root()
    protected_roots = {root / "data", root / "preprocessed_data"}
    if resolved in protected_roots and not resolved.exists():
        raise FileNotFoundError(
            f"Refusing to silently create missing dataset directory: {resolved}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def get_temp_dir(prefix: str = "fastspeech2_") -> Path:
    """Create and return an isolated temporary directory as a resolved ``Path``."""
    return Path(tempfile.mkdtemp(prefix=prefix)).resolve()


# ==============================================================================
# Language-Aware Path Accessors (English Svarah & Telugu IndicTTS)
# ==============================================================================

_ENGLISH_ALIASES = {"en", "eng", "english", "svarah", "svarah_profile_stratified_v1"}
_TELUGU_ALIASES = {
    "te",
    "tel",
    "telugu",
    "indictts",
    "telugu_option_1b",
    "option_1b",
}


def normalize_language(language: str) -> str:
    """Normalize a language or dataset identifier to ``'en'`` or ``'te'``."""
    if not isinstance(language, str) or not language.strip():
        raise ValueError(f"Language identifier must be a non-empty string, got {language!r}.")
    key = language.strip().lower()
    if key in _ENGLISH_ALIASES:
        return "en"
    if key in _TELUGU_ALIASES:
        return "te"
    raise ValueError(
        f"Unsupported language identifier {language!r}. "
        f"Expected one of {sorted(_ENGLISH_ALIASES | _TELUGU_ALIASES)}."
    )


def get_data_dir(
    language: str | None = None,
    *,
    must_exist: bool = False,
) -> Path:
    """Return the raw dataset directory (or language-specific corpus directory)."""
    root = get_project_root() / "data"
    if language is None:
        target = root
    else:
        lang = normalize_language(language)
        if lang == "en":
            target = root / "svarah_train" / "corpus"
        else:
            target = root / "telugu_parquet" / "data"
    if must_exist:
        return validate_required_path(
            target, f"Raw dataset directory ({language or 'root'})", must_be_dir=True
        )
    return target


def get_preprocessed_dir(
    language: str,
    *,
    variant: str | None = None,
    must_exist: bool = False,
) -> Path:
    """Return the preprocessed feature directory for English or Telugu.

    For Telugu:
    - Default (``variant=None`` and ``language`` not ``'telugu_option_1b'``):
      ``preprocessed_data/IndicTTS``
    - ``variant='option_1b'`` or ``language='telugu_option_1b'``:
      ``preprocessed_data/Telugu_Option_1B``
    """
    lang = normalize_language(language)
    prep_root = get_project_root() / "preprocessed_data"
    raw_key = language.strip().lower()
    var_key = (variant or "").strip().lower()

    if lang == "en":
        target = prep_root / "Svarah"
    else:
        if raw_key in ("telugu_option_1b", "option_1b") or var_key in (
            "option_1b",
            "telugu_option_1b",
        ):
            target = prep_root / "Telugu_Option_1B"
        else:
            target = prep_root / "IndicTTS"

    if must_exist:
        return validate_required_path(
            target, f"Preprocessed feature directory ({language})", must_be_dir=True
        )
    return target


def get_checkpoint_dir(
    language: str,
    *,
    must_exist: bool = False,
) -> Path:
    """Return the canonical checkpoint directory for English or Telugu."""
    lang = normalize_language(language)
    ckpt_root = get_project_root() / "checkpoints"
    if lang == "en":
        target = ckpt_root / "Svarah_profile_stratified_v1"
    else:
        target = ckpt_root / "IndicTTS"

    if must_exist:
        return validate_required_path(
            target, f"Checkpoint directory ({language})", must_be_dir=True
        )
    return target


def get_checkpoint_candidate_dirs(
    language: str,
    *,
    extra_dir: str | Path | None = None,
) -> list[Path]:
    """Return ordered candidate directories to search for English or Telugu checkpoints."""
    lang = normalize_language(language)
    root = get_project_root()
    candidates: list[Path] = []
    if extra_dir is not None:
        candidates.append(resolve_project_path(extra_dir))

    if lang == "en":
        candidates.extend(
            [
                root / "checkpoints" / "Svarah_profile_stratified_v1",
                root / "checkpoints" / "Svarah",
                root / "output" / "ckpt" / "Svarah_profile_stratified_v1",
                Path("/content/drive/MyDrive/FastSpeech2_Training/checkpoints/Svarah_profile_stratified_v1"),
                Path("/content/drive/MyDrive/FastSpeech2_Project/checkpoints/Svarah_profile_stratified_v1"),
                Path("/content/drive/MyDrive/FastSpeech2_Training/project/checkpoints/Svarah_profile_stratified_v1"),
            ]
        )
    else:
        candidates.extend(
            [
                root / "checkpoints" / "IndicTTS",
                root / "output" / "ckpt" / "IndicTTS",
                Path("/content/drive/MyDrive/FastSpeech2_Telugu/output/ckpt/IndicTTS"),
                Path("/content/drive/MyDrive/FastSpeech2_Project/output/ckpt/IndicTTS"),
                Path("/content/drive/MyDrive/FastSpeech2_Project/checkpoints/IndicTTS"),
            ]
        )

    deduped: list[Path] = []
    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def get_checkpoint_path(
    language: str,
    kind: str = "best",
    *,
    step: int | None = None,
    must_exist: bool = True,
) -> Path:
    """Return the path to a specific checkpoint file (`best`, `latest`, or `step`)."""
    ckpt_dir = get_checkpoint_dir(language)
    if step is not None:
        target = ckpt_dir / f"checkpoint_{step}.pth.tar"
    else:
        k = kind.strip().lower()
        if k in ("best", "checkpoint_best"):
            target = ckpt_dir / "checkpoint_best.pth.tar"
        elif k in ("latest", "checkpoint_latest"):
            target = ckpt_dir / "checkpoint_latest.pth.tar"
        else:
            target = ckpt_dir / k

    if must_exist:
        return validate_required_path(
            target, f"Model checkpoint ({language}, {step or kind})", must_be_file=True
        )
    return target


def get_result_dir(
    language: str | None = None,
    *,
    subdir: str | None = None,
    create: bool = False,
) -> Path:
    """Return the canonical ``Results/`` directory for English or Telugu."""
    res_root = get_project_root() / "Results"
    if language is None:
        target = res_root
    else:
        lang = normalize_language(language)
        target = res_root / ("English" if lang == "en" else "Telugu")

    if subdir:
        target = target / subdir

    if create:
        return ensure_output_dir(target)
    return target


def get_config_dir(language: str) -> Path:
    """Return the language configuration directory (`config/Svarah` or `config/IndicTTS`)."""
    lang = normalize_language(language)
    return get_project_root() / "config" / ("Svarah" if lang == "en" else "IndicTTS")


def get_config_path(
    language: str,
    config_type: str = "preprocess",
    *,
    must_exist: bool = True,
) -> Path:
    """Return a specific YAML configuration file path for English or Telugu.

    Args:
        language: Language identifier (``'en'``/``'svarah'`` or ``'te'``/``'indictts'``).
        config_type: One of ``'preprocess'``, ``'model'``, ``'train'``,
            ``'train_profile_stratified_v1'``, or a ``.yaml`` filename.
        must_exist: If True, validate that the YAML file exists on disk.
    """
    cfg_dir = get_config_dir(language)
    ctype = config_type.strip()
    if not ctype.endswith(".yaml"):
        ctype = f"{ctype}.yaml"
    target = cfg_dir / ctype
    if must_exist:
        return validate_required_path(
            target, f"Configuration file ({language}/{ctype})", must_be_file=True
        )
    return target


def get_config_paths(
    language: str,
    *,
    stratified: bool = True,
    must_exist: bool = True,
) -> tuple[Path, Path, Path]:
    """Return ``(preprocess_yaml, model_yaml, train_yaml)`` for English or Telugu."""
    lang = normalize_language(language)
    train_name = (
        "train_profile_stratified_v1" if (lang == "en" and stratified) else "train"
    )
    return (
        get_config_path(lang, "preprocess", must_exist=must_exist),
        get_config_path(lang, "model", must_exist=must_exist),
        get_config_path(lang, train_name, must_exist=must_exist),
    )


__all__ = [
    "PROJECT_ROOT",
    "COMMON_DIR",
    "ENGLISH_DIR",
    "TELUGU_DIR",
    "ORIGINAL_FS2_DIR",
    "CONFIG_DIR",
    "DATA_DIR",
    "PREPROCESSED_DIR",
    "CHECKPOINTS_DIR",
    "VOCODER_DIR",
    "HIFIGAN_DIR",
    "HIFIGAN_CONFIG_PATH",
    "HIFIGAN_WEIGHTS_PATH",
    "HIFIGAN_WEIGHTS_ZIP_PATH",
    "LEXICON_PATH",
    "RESULTS_DIR",
    "ENGLISH_RESULTS_DIR",
    "TELUGU_RESULTS_DIR",
    "OUTPUT_DIR",
    "REPORT_DIR",
    "REPORT_FIGURES_DIR",
    "DOCUMENTATION_DIR",
    "RESEARCH_DIR",
    "SCRATCH_DIR",
    "TESTS_DIR",
    "get_project_root",
    "setup_project_paths",
    "resolve_project_path",
    "validate_required_path",
    "ensure_output_dir",
    "get_temp_dir",
    "normalize_language",
    "get_data_dir",
    "get_preprocessed_dir",
    "get_checkpoint_dir",
    "get_checkpoint_candidate_dirs",
    "get_checkpoint_path",
    "get_result_dir",
    "get_config_dir",
    "get_config_path",
    "get_config_paths",
]

