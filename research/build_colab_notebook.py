"""Generator script to build and validate English_FastSpeech2_Training.ipynb."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

_BOOTSTRAP_ROOT = Path(__file__).resolve().parent.parent
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from common.paths import PROJECT_ROOT

NOTEBOOK_PATH = PROJECT_ROOT / "English_FastSpeech2_Training.ipynb"


def _md_cell(source: str) -> dict:
    lines = [line + "\n" for line in source.strip().splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def _code_cell(source: str) -> dict:
    # Validate Python syntax before creating cell
    ast.parse(source)
    lines = [line + "\n" for line in source.strip().splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def build_notebook() -> dict:
    cells = []

    # ------------------------------------------------------------------
    # Title & Overview
    # ------------------------------------------------------------------
    cells.append(
        _md_cell(
            """
# English FastSpeech 2 (Svarah `profile_stratified_v1`) — Google Colab GPU Training, Live Monitoring & Checkpoint Recovery

This notebook trains and monitors the **English FastSpeech 2** Text-to-Speech model on the preprocessed **Svarah** dataset using the **`profile_stratified_v1`** speaker-profile-stratified split (`3,208` training utterances, `664` validation utterances, `88/88` active demographic speaker profiles shared across train and validation, `115` total speaker table entries, `360` vocabulary symbols).

### Recommended Google Drive Layout
```text
MyDrive/
└── FastSpeech2_Training/
    ├── project/
    │   ├── common/
    │   ├── config/Svarah/
    │   │   └── train_profile_stratified_v1.yaml
    │   ├── English/
    │   │   └── metadata/splits/profile_stratified_v1/
    │   │       ├── train.txt
    │   │       └── val.txt
    │   └── original_fastspeech2/
    ├── preprocessed_data/
    │   └── Svarah/
    ├── checkpoints/
    │   └── Svarah_profile_stratified_v1/
    └── logs/
        └── Svarah_profile_stratified_v1/
```

### Key Features
- **Speaker-Profile-Stratified Split (`profile_stratified_v1`)**: Ensures 100% (`88/88`) of validation speaker profiles are trained in `train.txt` while enforcing zero `user_prefix`, `recording_id`, and normalized transcript leakage.
- **Isolated Step-0 Experiment Directories**: Writes checkpoints and logs to `checkpoints/Svarah_profile_stratified_v1` and `logs/Svarah_profile_stratified_v1`, preserving the baseline experiment untouched.
- **Google Drive Persistence**: Checkpoints and logs are written directly to Google Drive using atomic `.tmp` + `os.replace` operations.
- **Post-Save Checkpoint Verification**: Every saved checkpoint is immediately verified before pruning older periodic checkpoints (`max_to_keep = 5`).
- **Strict Best-Checkpoint Protection**: `checkpoint_best.pth.tar` is never overwritten unless validation total loss strictly improves.
- **Live 5-Graph Dashboard**: Real-time inline HTML telemetry and 5 Matplotlib subplots updated every `log_step = 100` steps.
"""
        )
    )

    # ------------------------------------------------------------------
    # Section 1: Google Drive Integration & Path Configuration
    # ------------------------------------------------------------------
    cells.append(
        _md_cell(
            """
## Section 1 — Google Drive Integration & Workspace Configuration

Mounts Google Drive (when running on Google Colab) and configures persistent paths for the project codebase, preprocessed Svarah features, `profile_stratified_v1` split manifests, checkpoints (`checkpoints/Svarah_profile_stratified_v1`), and training logs (`logs/Svarah_profile_stratified_v1`).
"""
        )
    )

    cells.append(
        _code_cell(
            """
import os
import sys
from pathlib import Path

# Detect Google Colab environment
try:
    import google.colab  # type: ignore
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive  # type: ignore
    MOUNT_POINT = Path("/content/drive")
    if not (MOUNT_POINT / "MyDrive").exists():
        print("Mounting Google Drive...")
        drive.mount(str(MOUNT_POINT))
    else:
        print("Google Drive is already mounted.")
    DRIVE_BASE = MOUNT_POINT / "MyDrive" / "FastSpeech2_Training"
    PROJECT_ROOT = DRIVE_BASE / "project"
    PREPROCESSED_DIR = DRIVE_BASE / "preprocessed_data" / "Svarah"
    CHECKPOINT_DIR = DRIVE_BASE / "checkpoints" / "Svarah_profile_stratified_v1"
    LOG_DIR = DRIVE_BASE / "logs" / "Svarah_profile_stratified_v1"
else:
    print("Running outside Google Colab (local diagnostic/validation mode).")
    PROJECT_ROOT = Path.cwd().resolve()
    if not (PROJECT_ROOT / "common").exists() and (PROJECT_ROOT.parent / "common").exists():
        PROJECT_ROOT = PROJECT_ROOT.parent
    DRIVE_BASE = PROJECT_ROOT / "colab_drive_sim"
    PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed_data" / "Svarah"
    CHECKPOINT_DIR = DRIVE_BASE / "checkpoints" / "Svarah_profile_stratified_v1"
    LOG_DIR = DRIVE_BASE / "logs" / "Svarah_profile_stratified_v1"

SPLIT_DIR = PROJECT_ROOT / "English" / "metadata" / "splits" / "profile_stratified_v1"
TRAIN_MANIFEST = SPLIT_DIR / "train.txt"
VAL_MANIFEST = SPLIT_DIR / "val.txt"

# Create persistent checkpoint and log directories
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure Python import paths
for p in (PROJECT_ROOT, PROJECT_ROOT / "original_fastspeech2"):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

print("=" * 72)
print(f"Environment       : {'Google Colab' if IN_COLAB else 'Local Host'}")
print(f"PROJECT_ROOT      : {PROJECT_ROOT} (exists={PROJECT_ROOT.exists()})")
print(f"PREPROCESSED_DIR  : {PREPROCESSED_DIR} (exists={PREPROCESSED_DIR.exists()})")
print(f"SPLIT_DIR         : {SPLIT_DIR} (exists={SPLIT_DIR.exists()})")
print(f"TRAIN_MANIFEST    : {TRAIN_MANIFEST} (exists={TRAIN_MANIFEST.exists()})")
print(f"VAL_MANIFEST      : {VAL_MANIFEST} (exists={VAL_MANIFEST.exists()})")
print(f"CHECKPOINT_DIR    : {CHECKPOINT_DIR} (exists={CHECKPOINT_DIR.exists()})")
print(f"LOG_DIR           : {LOG_DIR} (exists={LOG_DIR.exists()})")
print("=" * 72)

for required_path, desc in [
    (PROJECT_ROOT / "common", "common/ package"),
    (PROJECT_ROOT / "English", "English/ package"),
    (PROJECT_ROOT / "original_fastspeech2" / "dataset.py", "original_fastspeech2/dataset.py"),
    (PROJECT_ROOT / "config" / "Svarah" / "preprocess.yaml", "config/Svarah/preprocess.yaml"),
    (PROJECT_ROOT / "config" / "Svarah" / "model.yaml", "config/Svarah/model.yaml"),
    (
        PROJECT_ROOT / "config" / "Svarah" / "train_profile_stratified_v1.yaml",
        "config/Svarah/train_profile_stratified_v1.yaml",
    ),
    (TRAIN_MANIFEST, "Svarah profile_stratified_v1 train.txt manifest"),
    (VAL_MANIFEST, "Svarah profile_stratified_v1 val.txt manifest"),
    (PREPROCESSED_DIR / "speakers.json", "Svarah speakers.json"),
    (PREPROCESSED_DIR / "stats.json", "Svarah stats.json"),
]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"Missing {desc} at: {required_path}\\n"
            "Please verify your Google Drive folder layout matches MyDrive/FastSpeech2_Training/."
        )
print("All project and Svarah profile_stratified_v1 dataset paths verified successfully!")
"""
        )
    )

    # ------------------------------------------------------------------
    # Section 2: Dependency Inspection & Installation
    # ------------------------------------------------------------------
    cells.append(
        _md_cell(
            """
## Section 2 — Dependency Verification & Minimal Installation

Checks installed versions of core libraries (`torch`, `numpy`, `scipy`, `yaml`, `matplotlib`, `tqdm`, `psutil`) and installs only lightweight text-processing utilities (`unidecode`, `inflect`) if missing, avoiding any disruptive upgrades to Colab's pre-installed PyTorch/CUDA stack.
"""
        )
    )

    cells.append(
        _code_cell(
            """
import importlib
import subprocess
import sys

CORE_PACKAGES = ["torch", "numpy", "scipy", "yaml", "matplotlib", "tqdm"]
OPTIONAL_PACKAGES = ["psutil", "IPython"]
LIGHTWEIGHT_TEXT_DEPS = [
    ("unidecode", "Unidecode"),
    ("inflect", "inflect"),
]

print("--- Checking Core Pre-installed Packages ---")
for pkg in CORE_PACKAGES:
    mod = importlib.import_module(pkg)
    ver = getattr(mod, "__version__", "builtin")
    print(f"  [OK] {pkg:12s} : {ver}")

for opt_pkg in OPTIONAL_PACKAGES:
    try:
        mod = importlib.import_module(opt_pkg)
        ver = getattr(mod, "__version__", "available")
        print(f"  [OK] {opt_pkg:12s} : {ver} (optional telemetry/display)")
    except ImportError:
        print(f"  [INFO] {opt_pkg:10s} : not installed (using stdlib fallback)")

print("\\n--- Checking Text Cleaner Dependencies ---")
for mod_name, pip_name in LIGHTWEIGHT_TEXT_DEPS:
    try:
        mod = importlib.import_module(mod_name)
        print(f"  [OK] {mod_name:12s} : already installed")
    except ImportError:
        print(f"  [INSTALLING] {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", pip_name])
        # Install pydantic/typeguard/more-itertools only if inflect requires them
        try:
            importlib.import_module(mod_name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
        print(f"  [OK] {mod_name:12s} : installed successfully")
"""
        )
    )

    # ------------------------------------------------------------------
    # Section 3: GPU & Hardware Configuration
    # ------------------------------------------------------------------
    cells.append(
        _md_cell(
            """
## Section 3 — GPU & Hardware Configuration

Detects CUDA availability, reports GPU model (target: NVIDIA Tesla T4 on Colab), VRAM capacity, PyTorch version, and CUDA runtime version.
- Set `ALLOW_CPU_FALLBACK = False` (default) during actual training on Colab so the notebook halts immediately if a GPU runtime is not attached.
- Set `ALLOW_CPU_FALLBACK = True` only for small diagnostic/smoke tests when a GPU is unavailable.
"""
        )
    )

    cells.append(
        _code_cell(
            """
import torch
from English.monitoring import SystemMonitor

# Set to True ONLY for small diagnostic checks without a GPU
ALLOW_CPU_FALLBACK = not IN_COLAB

cuda_available = torch.cuda.is_available()
print("=" * 72)
print(f"PyTorch Version   : {torch.__version__}")
print(f"CUDA Available    : {cuda_available}")
print(f"CUDA Runtime Ver  : {torch.version.cuda if cuda_available else 'N/A'}")

if cuda_available:
    DEVICE = torch.device("cuda")
    sys_metrics = SystemMonitor.collect(DEVICE)
    print(f"Active Device     : {DEVICE} ({sys_metrics.gpu_name})")
    print(f"Total GPU Memory  : {sys_metrics.gpu_mem_total_gb:.2f} GB")
    if "T4" in sys_metrics.gpu_name:
        print("Target GPU Check  : NVIDIA T4 detected.")
    else:
        print(f"Target GPU Note   : Running on {sys_metrics.gpu_name} (CUDA compatible).")
else:
    if not ALLOW_CPU_FALLBACK:
        raise RuntimeError(
            "CUDA GPU is NOT available in this Colab runtime!\\n"
            "Please go to Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU, "
            "or set ALLOW_CPU_FALLBACK = True for small diagnostic tests only."
        )
    DEVICE = torch.device("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")
    sys_metrics = SystemMonitor.collect(DEVICE)
    print(f"WARNING: CUDA not available. Diagnostic fallback device selected: {DEVICE} ({sys_metrics.gpu_name})")

print(f"System RAM        : {sys_metrics.cpu_ram_used_gb:.2f} / {sys_metrics.cpu_ram_total_gb:.2f} GB ({sys_metrics.cpu_ram_pct:.1f}%)")
print("=" * 72)
"""
        )
    )

    # ------------------------------------------------------------------
    # Section 4: Preprocessed Dataset Validation
    # ------------------------------------------------------------------
    cells.append(
        _md_cell(
            """
## Section 4 — Preprocessed Svarah Dataset Validation (`profile_stratified_v1`)

Before initializing full training, this cell validates:
1. Profile-stratified training (`English/metadata/splits/profile_stratified_v1/train.txt`: 3,208 utterances) and validation (`English/metadata/splits/profile_stratified_v1/val.txt`: 664 utterances) manifests (`88/88` active speaker profiles shared across splits).
2. Presence of all `.npy` feature files (`mel`, `pitch`, `energy`, `duration`) for every utterance.
3. Vocabulary size (`360` symbols) and speaker profile mapping (`115` profiles in `speakers.json`).
4. Actual training and validation batch collation via `DataLoader`.
5. Absence of `NaN` or `Inf` values in any tensor.
6. Exact frame-level alignment: `sum(duration) == mel_frames == pitch_frames == energy_frames`.
"""
        )
    )

    cells.append(
        _code_cell(
            """
import json
import yaml
from torch.utils.data import DataLoader
from common.config import FastSpeech2Config
from dataset import Dataset
from English.monitoring import validate_svarah_dataset

PREPROCESS_YAML = PROJECT_ROOT / "config" / "Svarah" / "preprocess.yaml"
MODEL_YAML = PROJECT_ROOT / "config" / "Svarah" / "model.yaml"
TRAIN_YAML = PROJECT_ROOT / "config" / "Svarah" / "train_profile_stratified_v1.yaml"
SPLIT_DIR = PROJECT_ROOT / "English" / "metadata" / "splits" / "profile_stratified_v1"
TRAIN_MANIFEST = SPLIT_DIR / "train.txt"
VAL_MANIFEST = SPLIT_DIR / "val.txt"

with open(PREPROCESS_YAML, encoding="utf-8") as f:
    prep_dict = yaml.safe_load(f)
with open(MODEL_YAML, encoding="utf-8") as f:
    model_dict = yaml.safe_load(f)
with open(TRAIN_YAML, encoding="utf-8") as f:
    train_dict = yaml.safe_load(f)

# Point configs to active Drive / workspace paths and profile_stratified_v1 manifests
prep_dict["path"]["preprocessed_path"] = str(PREPROCESSED_DIR)
train_dict["path"]["ckpt_path"] = str(CHECKPOINT_DIR)
train_dict["path"]["log_path"] = str(LOG_DIR)
train_dict["path"]["split_dir"] = str(SPLIT_DIR)
train_dict["path"]["train_manifest"] = str(TRAIN_MANIFEST)
train_dict["path"]["val_manifest"] = str(VAL_MANIFEST)

with open(PREPROCESSED_DIR / "stats.json", encoding="utf-8") as f:
    model_dict["stats"] = json.load(f)

batch_size = train_dict["optimizer"]["batch_size"]
group_size = 4

train_dataset = Dataset(str(TRAIN_MANIFEST), prep_dict, train_dict, sort=True, drop_last=True)
val_dataset = Dataset(str(VAL_MANIFEST), prep_dict, train_dict, sort=False, drop_last=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size * group_size,
    shuffle=True,
    collate_fn=train_dataset.collate_fn,
    num_workers=2 if DEVICE.type == "cuda" else 0,
    pin_memory=(DEVICE.type == "cuda"),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size * group_size,
    shuffle=False,
    collate_fn=val_dataset.collate_fn,
    num_workers=0,
)

dataset_report = validate_svarah_dataset(
    preprocessed_path=PREPROCESSED_DIR,
    model_config=model_dict,
    train_loader=train_loader,
    val_loader=val_loader,
    expected_vocab_size=360,
    max_file_checks=None,  # Verify all 3,872 utterances
    train_manifest=TRAIN_MANIFEST,
    val_manifest=VAL_MANIFEST,
)

print("Dataset Validation Report:")
print(json.dumps(dataset_report, indent=2))
"""
        )
    )

    # ------------------------------------------------------------------
    # Section 5: Model & Loss Smoke Test
    # ------------------------------------------------------------------
    cells.append(
        _md_cell(
            """
## Section 5 — FastSpeech 2 Model & Multi-Task Loss Smoke Test

Initializes the English `FastSpeech2` model and `FastSpeech2Loss` on the target device, executes a real training batch forward/backward pass and a validation forward pass, verifies finite losses and gradients across all parameters, and clears gradients without stepping the optimizer.
"""
        )
    )

    cells.append(
        _code_cell(
            """
from common.model.fastspeech2 import FastSpeech2
from common.model.loss import FastSpeech2Loss
from English.monitoring import run_model_smoke_test

fs2_cfg = FastSpeech2Config.from_dicts(
    preprocess_dict=prep_dict,
    model_dict=model_dict,
    train_dict=train_dict,
)

smoke_model = FastSpeech2(fs2_cfg.preprocess, fs2_cfg.model).to(DEVICE)
smoke_loss_fn = FastSpeech2Loss(fs2_cfg.preprocess, fs2_cfg.model).to(DEVICE)

smoke_report = run_model_smoke_test(
    model=smoke_model,
    loss_fn=smoke_loss_fn,
    train_loader=train_loader,
    val_loader=val_loader,
    device=DEVICE,
)

print("Model Smoke Test Report:")
print(json.dumps(smoke_report, indent=2))

# Free temporary smoke-test model memory
del smoke_model, smoke_loss_fn
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""
        )
    )

    # ------------------------------------------------------------------
    # Section 6: Training Configuration & Checkpoint Status Inspection
    # ------------------------------------------------------------------
    cells.append(
        _md_cell(
            """
## Section 6 — Audited Training Configuration & Checkpoint Status Inspection

Configures the monitoring and checkpoint intervals for Google Colab resilience while preserving all audited model, optimizer, and Noam scheduler hyperparameters:
- **Checkpoint saving interval (`SAVE_STEP`)**: `1,000` steps
- **Validation interval (`VAL_STEP`)**: `1,000` steps
- **Live dashboard logging interval (`LOG_STEP`)**: `100` steps
- **Maximum retained periodic checkpoints (`MAX_TO_KEEP`)**: `5` (plus `checkpoint_latest.pth.tar` and `checkpoint_best.pth.tar`)

This cell also inspects `CHECKPOINT_DIR` (`checkpoints/Svarah_profile_stratified_v1`) on Google Drive and reports whether the experiment directory is clean for a fresh Step-0 launch.
"""
        )
    )

    cells.append(
        _code_cell(
            """
from common.training.checkpoint import CheckpointManager
from English.monitoring import verify_checkpoint_file

# Monitoring and Checkpoint Intervals (Part B & Part C)
SAVE_STEP = 1000
VAL_STEP = 1000
LOG_STEP = 100
MAX_TO_KEEP = 5
TOTAL_STEP = train_dict["step"]["total_step"]  # 900,000 default

ckpt_mgr = CheckpointManager(checkpoint_dir=CHECKPOINT_DIR, max_to_keep=MAX_TO_KEEP)

print("=" * 72)
print("AUDITED FASTSPEECH 2 TRAINING CONFIGURATION (profile_stratified_v1)")
print("=" * 72)
print(f"Train Config YAML     : {TRAIN_YAML}")
print(f"Train Manifest        : {TRAIN_MANIFEST}")
print(f"Val Manifest          : {VAL_MANIFEST}")
print(f"Checkpoint Dir        : {CHECKPOINT_DIR}")
print(f"Log Dir               : {LOG_DIR}")
print(f"Batch Size            : {train_dict['optimizer']['batch_size']} (Grad Acc: {train_dict['optimizer']['grad_acc_step']})")
print(f"Optimizer             : Adam (betas={train_dict['optimizer']['betas']}, eps={train_dict['optimizer']['eps']})")
print(f"Noam Warmup Steps     : {train_dict['optimizer']['warm_up_step']} (init_lr={train_dict['optimizer']['init_lr']})")
print(f"Anneal Steps          : {train_dict['optimizer']['anneal_steps']} (rate={train_dict['optimizer']['anneal_rate']})")
print(f"Grad Clip Threshold   : {train_dict['optimizer']['grad_clip_thresh']}")
print(f"Total Steps           : {TOTAL_STEP:,}")
print(f"Save Interval         : Every {SAVE_STEP:,} steps (Max Retained: {MAX_TO_KEEP})")
print(f"Validation Interval   : Every {VAL_STEP:,} steps")
print(f"Live Log Interval     : Every {LOG_STEP:,} steps")
print("-" * 72)

try:
    resolved_ckpt = ckpt_mgr.resolve_checkpoint_path(None)
    ckpt_meta = verify_checkpoint_file(resolved_ckpt)
    print("CHECKPOINT STATUS     : EXISTING VALID CHECKPOINT FOUND IN EXPERIMENT DIR")
    print(f"  Path                : {ckpt_meta['path']} ({ckpt_meta['size_mb']} MB)")
    print(f"  Saved Step          : {ckpt_meta['global_step']:,} (Next step: {ckpt_meta['global_step'] + 1:,})")
    print(f"  Saved Epoch         : {ckpt_meta['epoch']}")
    print(f"  Best Val Loss       : {ckpt_meta['best_val_loss']:.4f}")
except FileNotFoundError:
    print("CHECKPOINT STATUS     : NO CHECKPOINT FOUND (WILL START FRESH FROM STEP 0)")
print("=" * 72)
"""
        )
    )

    # ------------------------------------------------------------------
    # Section 7: Controlled Dry-Run Verification
    # ------------------------------------------------------------------
    cells.append(
        _md_cell(
            """
## Section 7 — Controlled Dry-Run Verification (1 Train Step + 1 Val Step + Verified Save)

Executes a single-step dry run using an isolated temporary subfolder to verify end-to-end `profile_stratified_v1` manifest loading, checkpoint saving, post-save load verification, CSV logging, and 5-graph plot rendering without polluting `CHECKPOINT_DIR` or `LOG_DIR`.
"""
        )
    )

    cells.append(
        _code_cell(
            """
import shutil
import tempfile
from English.train import parse_args, run_training

dry_tmp_dir = Path(tempfile.mkdtemp(prefix="colab_dry_run_check_"))
try:
    dry_args = parse_args([
        "--preprocess_config", str(PREPROCESS_YAML),
        "--model_config", str(MODEL_YAML),
        "--train_config", str(TRAIN_YAML),
        "--preprocessed_dir", str(PREPROCESSED_DIR),
        "--split_dir", str(SPLIT_DIR),
        "--train_manifest", str(TRAIN_MANIFEST),
        "--val_manifest", str(VAL_MANIFEST),
        "--checkpoint_dir", str(dry_tmp_dir / "ckpt"),
        "--log_dir", str(dry_tmp_dir / "logs"),
        "--device", str(DEVICE),
        "--save_step", str(SAVE_STEP),
        "--val_step", str(VAL_STEP),
        "--log_step", str(LOG_STEP),
        "--max_to_keep", str(MAX_TO_KEEP),
        "--dry_run",
    ])
    dry_result = run_training(dry_args)
    print("Controlled Dry-Run Result:", dry_result)
finally:
    shutil.rmtree(dry_tmp_dir, ignore_errors=True)
"""
        )
    )

    # ------------------------------------------------------------------
    # Section 8: Full Training Execution with Live Monitoring Dashboard
    # ------------------------------------------------------------------
    cells.append(
        _md_cell(
            """
## Section 8 — Launch Fresh Step-0 English FastSpeech 2 Training (`profile_stratified_v1` & Live Dashboard)

**Instructions to Start Fresh Training:**
1. Confirm Sections 1–7 above have passed without errors.
2. Change `START_FULL_TRAINING = False` to `START_FULL_TRAINING = True` in the cell below.
3. Run the cell.
   - Training starts fresh from **Step 0** using `config/Svarah/train_profile_stratified_v1.yaml` and the `English/metadata/splits/profile_stratified_v1/{train.txt,val.txt}` manifests, saving checkpoints to `checkpoints/Svarah_profile_stratified_v1` and logs to `logs/Svarah_profile_stratified_v1`.
   - To stop safely at any time, click the **Stop (Interrupt execution)** button in Colab; the loop catches `KeyboardInterrupt` and writes a verified emergency checkpoint to Google Drive.
"""
        )
    )

    cells.append(
        _code_cell(
            """
from English.train import parse_args, run_training

# SAFETY GATE: Set START_FULL_TRAINING = True when ready to launch fresh Step-0 training on Colab GPU
START_FULL_TRAINING = False

# Optional override for total steps (e.g. set to a smaller number for a bounded session, or None for full 900,000)
CUSTOM_TOTAL_STEP = None

if not START_FULL_TRAINING:
    print(
        "Training launch gate is currently set to START_FULL_TRAINING = False.\\n"
        "All pre-flight validations (Drive paths, GPU check, Svarah profile_stratified_v1 dataset check, model smoke test, and dry run) "
        "are ready.\\nSet START_FULL_TRAINING = True and re-run this cell to start fresh Step-0 training."
    )
else:
    if not torch.cuda.is_available() and not ALLOW_CPU_FALLBACK:
        raise RuntimeError("Refusing to launch full training without a CUDA GPU.")

    cli_args = [
        "--preprocess_config", str(PREPROCESS_YAML),
        "--model_config", str(MODEL_YAML),
        "--train_config", str(TRAIN_YAML),
        "--preprocessed_dir", str(PREPROCESSED_DIR),
        "--split_dir", str(SPLIT_DIR),
        "--train_manifest", str(TRAIN_MANIFEST),
        "--val_manifest", str(VAL_MANIFEST),
        "--checkpoint_dir", str(CHECKPOINT_DIR),
        "--log_dir", str(LOG_DIR),
        "--device", str(DEVICE),
        "--save_step", str(SAVE_STEP),
        "--val_step", str(VAL_STEP),
        "--log_step", str(LOG_STEP),
        "--max_to_keep", str(MAX_TO_KEEP),
    ]
    if CUSTOM_TOTAL_STEP is not None:
        cli_args.extend(["--total_step", str(CUSTOM_TOTAL_STEP)])

    train_args = parse_args(cli_args)
    training_summary = run_training(train_args)
"""
        )
    )

    # ------------------------------------------------------------------
    # Section 9: Post-Run / Recovery Inspection & Summary Viewer
    # ------------------------------------------------------------------
    cells.append(
        _md_cell(
            """
## Section 9 — Post-Run / Disconnection Inspection & Loss Curve Viewer

Run this cell at any time (after training finishes, or upon reconnecting after a Colab timeout) to:
1. Inspect `training_status.json` and `training_summary.json`.
2. List all verified checkpoints currently stored in `CHECKPOINT_DIR`.
3. Reconstruct and display all 5 loss/learning-rate graphs from `training_history.csv` and `validation_history.csv`.
"""
        )
    )

    cells.append(
        _code_cell(
            """
import json
from common.training.checkpoint import CheckpointManager
from English.monitoring import LiveDashboard, PersistentLogger, verify_checkpoint_file

try:
    from IPython.display import Image, Markdown, display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

logger_inspect = PersistentLogger(log_dir=LOG_DIR)
ckpt_mgr_inspect = CheckpointManager(checkpoint_dir=CHECKPOINT_DIR, max_to_keep=MAX_TO_KEEP)

status_file = LOG_DIR / "training_status.json"
summary_md_file = LOG_DIR / "training_summary.md"

print("=" * 72)
print("PERSISTENT DRIVE STATUS & CHECKPOINT INVENTORY")
print("=" * 72)

if status_file.exists():
    with open(status_file, encoding="utf-8") as f:
        status_data = json.load(f)
    print(json.dumps(status_data, indent=2))
else:
    print("No training_status.json found yet (training has not started).")

print("\\n--- Checkpoints on Google Drive ---")
for label, path in [("LATEST", ckpt_mgr_inspect.latest_path), ("BEST", ckpt_mgr_inspect.best_path)]:
    if path.exists():
        meta = verify_checkpoint_file(path)
        print(f"  [{label:6s}] {path.name:28s} | step={meta['global_step']:,} | epoch={meta['epoch']} | best_val={meta['best_val_loss']:.4f} | {meta['size_mb']} MB")
    else:
        print(f"  [{label:6s}] Not created yet")

periodic = ckpt_mgr_inspect.find_all_step_checkpoints()
print(f"  [PERIODIC ({len(periodic)}/{MAX_TO_KEEP})]: {[p.name for _, p in periodic]}")

# Reconstruct graphs if any history exists
if logger_inspect.train_csv_path.exists():
    logger_inspect.restore_and_deduplicate(max_valid_step=10**9)
    if logger_inspect.train_records or logger_inspect.val_records:
        dash_inspect = LiveDashboard(
            logger=logger_inspect,
            total_steps=TOTAL_STEP,
            total_epochs=4500,
            enable_notebook_display=False,
        )
        dash_inspect.render_plots()
        if HAS_IPYTHON and dash_inspect.dashboard_png_path.exists():
            display(Image(filename=str(dash_inspect.dashboard_png_path)))

if summary_md_file.exists():
    if HAS_IPYTHON:
        display(Markdown(summary_md_file.read_text(encoding="utf-8")))
    else:
        print(summary_md_file.read_text(encoding="utf-8"))
"""
        )
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook


def main() -> None:
    nb = build_notebook()
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Successfully generated and syntax-validated: {NOTEBOOK_PATH}")
    print(f"Total cells: {len(nb['cells'])}")


if __name__ == "__main__":
    main()
