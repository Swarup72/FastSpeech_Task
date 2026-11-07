"""Unit tests for centralized path management (`common.paths`).

Tests:
1. Project root discovery from different working directories (including /tmp).
2. Zero dependency on os.getcwd() / Path.cwd() and zero ML/training imports inside common/paths.py.
3. Relative configuration path resolution against PROJECT_ROOT.
4. Explicit absolute path and Google Colab path handling.
5. Missing checkpoint and dataset error handling (never silently creating missing input dirs).
6. Output directory creation via ensure_output_dir.
7. English (Svarah) and Telugu (IndicTTS / Option 1B) path resolution.
"""

from __future__ import annotations

import ast
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from common.paths import (
    CHECKPOINTS_DIR,
    COMMON_DIR,
    CONFIG_DIR,
    DATA_DIR,
    DOCUMENTATION_DIR,
    ENGLISH_DIR,
    ENGLISH_RESULTS_DIR,
    HIFIGAN_CONFIG_PATH,
    HIFIGAN_DIR,
    HIFIGAN_WEIGHTS_PATH,
    LEXICON_PATH,
    ORIGINAL_FS2_DIR,
    OUTPUT_DIR,
    PREPROCESSED_DIR,
    PROJECT_ROOT,
    REPORT_DIR,
    REPORT_FIGURES_DIR,
    RESEARCH_DIR,
    RESULTS_DIR,
    SCRATCH_DIR,
    TELUGU_DIR,
    TELUGU_RESULTS_DIR,
    TESTS_DIR,
    ensure_output_dir,
    get_checkpoint_candidate_dirs,
    get_checkpoint_dir,
    get_checkpoint_path,
    get_config_dir,
    get_config_path,
    get_config_paths,
    get_data_dir,
    get_preprocessed_dir,
    get_project_root,
    get_result_dir,
    get_temp_dir,
    normalize_language,
    resolve_project_path,
    setup_project_paths,
    validate_required_path,
)


class TestCentralPathManager(unittest.TestCase):
    """Comprehensive unit tests for common.paths."""

    def test_01_ast_purity_no_getcwd_or_ml_imports(self) -> None:
        """Verify common/paths.py uses only stdlib and never calls os.getcwd() or Path.cwd()."""
        paths_file = COMMON_DIR / "paths.py"
        self.assertTrue(paths_file.is_file())
        source = paths_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(paths_file))

        forbidden_imports = {
            "torch",
            "torchaudio",
            "numpy",
            "scipy",
            "librosa",
            "matplotlib",
            "yaml",
            "English",
            "Telugu",
            "original_fastspeech2",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    self.assertNotIn(top, forbidden_imports)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                self.assertNotIn(top, forbidden_imports)
            elif isinstance(node, ast.Attribute):
                self.assertNotIn(
                    node.attr,
                    {"getcwd", "cwd"},
                    f"Forbidden CWD call '{node.attr}' found in common/paths.py",
                )

    def test_02_project_root_discovery_from_different_working_directories(self) -> None:
        """Verify get_project_root() returns the exact same root regardless of os.getcwd()."""
        expected_root = PROJECT_ROOT
        self.assertTrue((expected_root / "common").is_dir())
        self.assertTrue((expected_root / "config").is_dir())

        orig_cwd = os.getcwd()
        with tempfile.TemporaryDirectory(prefix="fs2_cwd_test_") as tmp_dir:
            try:
                os.chdir(tmp_dir)
                self.assertEqual(get_project_root(), expected_root)
                os.chdir(str(expected_root / "English"))
                self.assertEqual(get_project_root(), expected_root)
                os.chdir(str(expected_root / "Telugu" / "text"))
                self.assertEqual(get_project_root(), expected_root)
            finally:
                os.chdir(orig_cwd)

    def test_03_env_override_validation(self) -> None:
        """Verify FASTSPEECH2_PROJECT_ROOT override succeeds on valid root and raises on invalid root."""
        with tempfile.TemporaryDirectory(prefix="fs2_invalid_root_") as invalid_dir:
            with mock.patch.dict(
                os.environ, {"FASTSPEECH2_PROJECT_ROOT": invalid_dir}
            ):
                with self.assertRaises(FileNotFoundError):
                    get_project_root()

            # Create sentinel directories in temp dir and verify override works
            (Path(invalid_dir) / "common").mkdir()
            (Path(invalid_dir) / "config").mkdir()
            with mock.patch.dict(
                os.environ, {"FASTSPEECH2_PROJECT_ROOT": invalid_dir}
            ):
                self.assertEqual(get_project_root(), Path(invalid_dir).resolve())

    def test_04_relative_configuration_path_resolution(self) -> None:
        """Verify relative paths (with or without ./) resolve against PROJECT_ROOT from any CWD."""
        orig_cwd = os.getcwd()
        with tempfile.TemporaryDirectory(prefix="fs2_rel_test_") as tmp_dir:
            try:
                os.chdir(tmp_dir)
                p1 = resolve_project_path("./preprocessed_data/Svarah")
                p2 = resolve_project_path("preprocessed_data/IndicTTS")
                p3 = resolve_project_path("config/Svarah/preprocess.yaml")
                p4 = resolve_project_path("./output/ckpt/IndicTTS")

                self.assertEqual(p1, PROJECT_ROOT / "preprocessed_data" / "Svarah")
                self.assertEqual(p2, PROJECT_ROOT / "preprocessed_data" / "IndicTTS")
                self.assertEqual(p3, PROJECT_ROOT / "config" / "Svarah" / "preprocess.yaml")
                self.assertEqual(p4, PROJECT_ROOT / "output" / "ckpt" / "IndicTTS")
            finally:
                os.chdir(orig_cwd)

        with self.assertRaises(ValueError):
            resolve_project_path("")

    def test_05_explicit_absolute_and_colab_path_handling(self) -> None:
        """Verify explicit absolute paths and Google Colab paths are preserved unaltered."""
        colab_ckpt = "/content/drive/MyDrive/FastSpeech2_Training/checkpoints/Svarah_profile_stratified_v1/checkpoint_best.pth.tar"
        resolved = resolve_project_path(colab_ckpt)
        self.assertEqual(resolved, Path(colab_ckpt))
        self.assertTrue(resolved.is_absolute())

        with tempfile.TemporaryDirectory(prefix="fs2_abs_test_") as tmp_dir:
            abs_p = Path(tmp_dir) / "custom_output" / "sample.wav"
            self.assertEqual(resolve_project_path(abs_p), Path(os.path.normpath(str(abs_p))))

    def test_06_missing_checkpoint_and_dataset_errors(self) -> None:
        """Verify missing dataset/checkpoint paths raise FileNotFoundError and never create directories."""
        missing_dir = PROJECT_ROOT / "checkpoints" / "nonexistent_language_v99"
        self.assertFalse(missing_dir.exists())

        with self.assertRaises(FileNotFoundError):
            validate_required_path(
                missing_dir, "Missing checkpoint dir", must_be_dir=True
            )
        self.assertFalse(
            missing_dir.exists(),
            "validate_required_path must never create a missing directory!",
        )

        with self.assertRaises(FileNotFoundError):
            get_checkpoint_path("en", step=999999, must_exist=True)

        with self.assertRaises(FileNotFoundError):
            get_config_path("te", "nonexistent_config.yaml", must_exist=True)

        # Verify validate_required_path distinguishes file vs directory
        with self.assertRaises(FileNotFoundError):
            validate_required_path(CONFIG_DIR, "Config dir as file", must_be_file=True)
        with self.assertRaises(FileNotFoundError):
            validate_required_path(
                LEXICON_PATH, "Lexicon file as dir", must_be_dir=True
            )

    def test_07_output_directory_creation_and_safety_guard(self) -> None:
        """Verify ensure_output_dir creates nested output directories in temp space."""
        tmp_root = get_temp_dir("fs2_out_test_")
        try:
            nested = tmp_root / "Results" / "English" / "custom_plots"
            self.assertFalse(nested.exists())
            created = ensure_output_dir(nested)
            self.assertEqual(created, nested)
            self.assertTrue(created.is_dir())
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def test_08_english_and_telugu_path_resolution(self) -> None:
        """Verify language-specific accessors for English (Svarah) and Telugu (IndicTTS)."""
        self.assertEqual(normalize_language("en"), "en")
        self.assertEqual(normalize_language("English"), "en")
        self.assertEqual(normalize_language("Svarah"), "en")
        self.assertEqual(normalize_language("te"), "te")
        self.assertEqual(normalize_language("Telugu"), "te")
        self.assertEqual(normalize_language("IndicTTS"), "te")
        self.assertEqual(normalize_language("Telugu_Option_1B"), "te")

        with self.assertRaises(ValueError):
            normalize_language("fr")
        with self.assertRaises(ValueError):
            normalize_language("")

        # Preprocessed directories
        self.assertEqual(
            get_preprocessed_dir("en", must_exist=True),
            PREPROCESSED_DIR / "Svarah",
        )
        self.assertEqual(
            get_preprocessed_dir("te", must_exist=True),
            PREPROCESSED_DIR / "IndicTTS",
        )
        self.assertEqual(
            get_preprocessed_dir("te", variant="option_1b", must_exist=True),
            PREPROCESSED_DIR / "Telugu_Option_1B",
        )
        self.assertEqual(
            get_preprocessed_dir("Telugu_Option_1B", must_exist=True),
            PREPROCESSED_DIR / "Telugu_Option_1B",
        )

        # Checkpoints
        self.assertEqual(
            get_checkpoint_dir("en", must_exist=True),
            CHECKPOINTS_DIR / "Svarah_profile_stratified_v1",
        )
        self.assertEqual(
            get_checkpoint_dir("te", must_exist=True),
            CHECKPOINTS_DIR / "IndicTTS",
        )
        self.assertTrue(get_checkpoint_path("en", "best").is_file())
        self.assertTrue(get_checkpoint_path("en", "latest").is_file())
        self.assertTrue(get_checkpoint_path("te", "best").is_file())
        self.assertTrue(get_checkpoint_path("te", "latest").is_file())

        # Candidate checkpoint directories
        en_cands = get_checkpoint_candidate_dirs("en")
        te_cands = get_checkpoint_candidate_dirs("te")
        self.assertIn(CHECKPOINTS_DIR / "Svarah_profile_stratified_v1", en_cands)
        self.assertIn(CHECKPOINTS_DIR / "IndicTTS", te_cands)

        # Results
        self.assertEqual(get_result_dir("en"), ENGLISH_RESULTS_DIR)
        self.assertEqual(get_result_dir("te"), TELUGU_RESULTS_DIR)
        self.assertEqual(
            get_result_dir("en", subdir="audio"), ENGLISH_RESULTS_DIR / "audio"
        )
        self.assertEqual(
            get_result_dir("te", subdir="plots"), TELUGU_RESULTS_DIR / "plots"
        )

        # Configs
        en_prep, en_mod, en_trn = get_config_paths("en", stratified=True)
        self.assertEqual(en_prep, CONFIG_DIR / "Svarah" / "preprocess.yaml")
        self.assertEqual(en_mod, CONFIG_DIR / "Svarah" / "model.yaml")
        self.assertEqual(
            en_trn, CONFIG_DIR / "Svarah" / "train_profile_stratified_v1.yaml"
        )

        te_prep, te_mod, te_trn = get_config_paths("te")
        self.assertEqual(te_prep, CONFIG_DIR / "IndicTTS" / "preprocess.yaml")
        self.assertEqual(te_mod, CONFIG_DIR / "IndicTTS" / "model.yaml")
        self.assertEqual(te_trn, CONFIG_DIR / "IndicTTS" / "train.yaml")

        # Constants existence check
        for p in (
            COMMON_DIR,
            ENGLISH_DIR,
            TELUGU_DIR,
            ORIGINAL_FS2_DIR,
            CONFIG_DIR,
            DATA_DIR,
            PREPROCESSED_DIR,
            CHECKPOINTS_DIR,
            HIFIGAN_DIR,
            HIFIGAN_CONFIG_PATH,
            HIFIGAN_WEIGHTS_PATH,
            LEXICON_PATH,
            RESULTS_DIR,
            OUTPUT_DIR,
            REPORT_DIR,
            REPORT_FIGURES_DIR,
            DOCUMENTATION_DIR,
            RESEARCH_DIR,
            SCRATCH_DIR,
            TESTS_DIR,
        ):
            self.assertTrue(p.exists(), f"Expected canonical path to exist: {p}")

        # setup_project_paths idempotency
        root = setup_project_paths(include_original_fs2=True)
        self.assertEqual(root, PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()

