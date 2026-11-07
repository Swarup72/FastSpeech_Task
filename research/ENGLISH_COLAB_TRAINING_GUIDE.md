# English FastSpeech 2 (Svarah) — Google Colab GPU Training, Live Monitoring & Checkpoint Recovery Guide

**Project Root**: `/Users/swaruppal/FastSpeech2_Project`  
**Primary Notebook**: `English_FastSpeech2_Training.ipynb`  
**Training Runner**: `English/train.py`  
**Monitoring & Recovery Engine**: `English/monitoring.py`  
**Bundle Packaging Utility**: `English/prepare_colab_bundle.py`  

---

## 1. Google Drive Directory Structure

All persistent files (project code, preprocessed Svarah `.npy` features, checkpoints, and training logs) are stored under a single root folder in your Google Drive:

```text
MyDrive/
└── FastSpeech2_Training/
    ├── project/
    │   ├── common/
    │   ├── config/
    │   │   └── Svarah/
    │   │       ├── model.yaml
    │   │       ├── preprocess.yaml
    │   │       └── train.yaml
    │   ├── English/
    │   │   ├── __init__.py
    │   │   ├── monitoring.py
    │   │   ├── prepare_colab_bundle.py
    │   │   ├── train.py
    │   │   └── text/
    │   └── original_fastspeech2/
    │       ├── dataset.py
    │       ├── text/
    │       └── utils/
    ├── preprocessed_data/
    │   └── Svarah/
    │       ├── train.txt          (3,208 utterances)
    │       ├── val.txt            (664 utterances)
    │       ├── speakers.json      (115 speaker profiles)
    │       ├── stats.json         (pitch & energy normalization stats)
    │       ├── duration/          (3,874 .npy files)
    │       ├── energy/            (3,874 .npy files)
    │       ├── mel/               (3,874 .npy files)
    │       └── pitch/             (3,874 .npy files)
    ├── checkpoints/
    │   └── Svarah/
    │       ├── checkpoint_latest.pth.tar
    │       ├── checkpoint_best.pth.tar
    │       └── checkpoint_{step}.pth.tar  (at most 5 retained)
    └── logs/
        └── Svarah/
            ├── training_history.csv
            ├── validation_history.csv
            ├── training_status.json
            ├── training.log
            ├── training_summary.json
            ├── training_summary.md
            └── plots/
                ├── loss_curves_dashboard.png
                ├── graph1_train_total_loss.png
                ├── graph2_val_total_loss.png
                ├── graph3_train_loss_components.png
                ├── graph4_val_loss_components.png
                └── graph5_learning_rate.png
```

> **Important Note on Excluded Files**: Do **not** upload `.venv/`, `__pycache__/`, `data/svarah_train/corpus/*.wav` (raw audio), `Telugu/`, or `original_fastspeech2/demo/`. Training requires only the preprocessed `.npy` arrays and the lightweight Python modules (~200 KB of code).

---

## 2. How to Upload or Sync the Project to Google Drive

Uploading 15,000+ small `.npy` files individually via a browser is slow. Instead, package the exact required directories into a single `.tar.gz` archive locally, upload that single file to Google Drive, and extract it inside Colab in ~10 seconds.

### Step 2.1 — Create the Clean Archive Locally (on your Mac)
Run the included packaging script from `/Users/swaruppal/FastSpeech2_Project`:

```bash
.venv/bin/python English/prepare_colab_bundle.py --output FastSpeech2_Colab_Bundle.tar.gz
```

This creates `FastSpeech2_Colab_Bundle.tar.gz` (~380 MB compressed) containing **only** `FastSpeech2_Training/project/...` and `FastSpeech2_Training/preprocessed_data/Svarah/...`.

### Step 2.2 — Upload to Google Drive
1. Open [Google Drive](https://drive.google.com/).
2. Upload `FastSpeech2_Colab_Bundle.tar.gz` and `English_FastSpeech2_Training.ipynb` to the root of your `MyDrive`.
3. Once inside Colab (after mounting Google Drive in Section 1), run a one-time extraction command in a scratch cell if `MyDrive/FastSpeech2_Training` does not exist yet:
   ```python
   !tar -xzf /content/drive/MyDrive/FastSpeech2_Colab_Bundle.tar.gz -C /content/drive/MyDrive/
   ```

*(Optional Speed Optimization for Colab I/O)*: If reading `.npy` files over the Google Drive FUSE mount feels slow on epoch 1, you can extract `preprocessed_data/Svarah` to local Colab SSD (`/content/preprocessed_data/Svarah`) while keeping `CHECKPOINT_DIR` and `LOG_DIR` on `/content/drive/MyDrive/FastSpeech2_Training/`.

---

## 3. How to Open the Notebook & Enable GPU Runtime in Colab

1. In Google Drive, double-click **`English_FastSpeech2_Training.ipynb`** and select **Open with -> Google Colaboratory** (or open [colab.research.google.com](https://colab.research.google.com/) and select **File -> Upload notebook**).
2. In the top menu bar of Google Colab, click **Runtime -> Change runtime type**.
3. Set:
   - **Runtime type**: `Python 3`
   - **Hardware accelerator**: **`T4 GPU`** (or another available CUDA GPU)
4. Click **Save**.

---

## 4. Executing the Notebook Section by Section

Run the cells sequentially from top to bottom:

| Section | Purpose | Expected Output |
| :--- | :--- | :--- |
| **Section 1 — Google Drive Integration** | Mounts `/content/drive`, creates `checkpoints/Svarah` and `logs/Svarah`, and verifies all required project and manifest files exist. | `All project and Svarah dataset paths verified successfully!` |
| **Section 2 — Dependency Verification** | Verifies pre-installed `torch`, `numpy`, `scipy`, `yaml`, `matplotlib`, `tqdm`, `psutil` and installs `Unidecode` and `inflect` if missing. | `[OK] torch ...`, `[OK] unidecode ...`, `[OK] inflect ...` |
| **Section 3 — GPU Configuration** | Verifies `torch.cuda.is_available() == True`, displays GPU model (`Tesla T4`) and VRAM (~15.0 GB). Fails clearly if GPU is missing and `ALLOW_CPU_FALLBACK = False`. | `Active Device: cuda (Tesla T4)`, `Target GPU Check: NVIDIA T4 detected.` |
| **Section 4 — Dataset Validation** | Checks all 3,208 train and 664 val utterances, verifies `.npy` paths, checks `n_src_vocab == 360` and `115` speakers, collates real batches, and checks `sum(duration) == mel_lens` and zero `NaN`/`Inf`. | `"status": "PASSED"`, `"train_utterances": 3208`, `"val_utterances": 664` |
| **Section 5 — Model Smoke Test** | Runs a real training batch forward/backward pass and a validation forward pass on GPU, verifies finite losses and gradients across all 35,188,545 parameters, and clears gradients without stepping the optimizer. | `"status": "PASSED"`, finite `train_losses` and `val_losses` |
| **Section 6 — Configuration & Checkpoint Inspection** | Displays audited settings (`save_step=1000`, `val_step=1000`, `log_step=100`, `max_to_keep=5`) and inspects `CHECKPOINT_DIR` to report whether the run will start fresh or resume. | Either `NO CHECKPOINT FOUND (WILL START FRESH)` or `EXISTING VALID CHECKPOINT FOUND (WILL RESUME)` |
| **Section 7 — Controlled Dry-Run** | Executes 1 train step, 1 val step, and 1 verified atomic checkpoint save in a temporary directory to confirm end-to-end readiness. | `Controlled Dry-Run Result: {'status': 'DRY_RUN_COMPLETED', 'step': 1}` |
| **Section 8 — Launch Full Training** | Protected by `START_FULL_TRAINING = False`. Set `START_FULL_TRAINING = True` to launch or resume training with the live dashboard. | Live updating HTML table + 5-panel Matplotlib figure every 100 steps |
| **Section 9 — Post-Run / Recovery Viewer** | Displays `training_status.json`, lists all checkpoints on Drive, renders the 5 loss graphs from CSV history, and displays `training_summary.md`. | Full inventory of checkpoints, status JSON, and 5-panel plot |

---

## 5. How to Read the Live Monitoring Dashboard

During Section 8 execution, the output cell refreshes every **100 steps** (`LOG_STEP = 100`) and displays:

1. **Top Progress Banner & Bar**:
   - Shows whether the session started fresh or resumed (`FRESH RUN` vs `RESUMED @ STEP N`), current `Epoch / Total Epochs`, `Global Step / Total Steps`, and completion percentage.
2. **Telemetry Strip**:
   - **Learning Rate**: Current Noam schedule learning rate (warms up linearly until step 4,000 at `9.88e-4`, then decays as $\text{step}^{-0.5}$).
   - **Throughput**: Training batches (steps) per second.
   - **Elapsed & ETA**: Wall-clock time elapsed in the current session and estimated time remaining.
   - **GPU & RAM**: GPU model, allocated/reserved/total VRAM in GB, GPU utilization %, and host system RAM usage.
3. **Training vs. Validation Loss Comparison Table**:
   - **Row 1 (Training)**: Latest step's `Total Loss`, `Mel Loss`, `PostNet Mel Loss`, `Pitch Loss`, `Energy Loss`, `Duration Loss`, and `Grad Norm`.
   - **Row 2 (Validation)**: Updated every **1,000 steps** (`VAL_STEP = 1000`), showing sample-weighted validation losses across all 664 validation utterances, the historical `Best Validation Loss` and step, and a green **`IMPROVED (Best)`** badge whenever validation loss reaches a new low.
4. **Live 5-Graph Matplotlib Figure**:
   - **Graph 1**: Training Total Loss vs. Global Step (raw un-smoothed measurements).
   - **Graph 2**: Validation Total Loss vs. Global Step (with green star marking the best validation checkpoint).
   - **Graph 3**: Individual Training Loss Components (`Mel`, `PostNet Mel`, `Pitch`, `Energy`, `Duration`).
   - **Graph 4**: Individual Validation Loss Components.
   - **Graph 5**: Learning Rate vs. Global Step.

---

## 6. How to Identify Checkpoints & Resume After Disconnection

### Checkpoint Files in `MyDrive/FastSpeech2_Training/checkpoints/Svarah/`
- **`checkpoint_latest.pth.tar`**: Always points to the most recently verified checkpoint (updated every 1,000 steps and on safe interruption).
- **`checkpoint_best.pth.tar`**: Preserves the checkpoint with the strictly lowest validation total loss (`val_loss.total < prev_best_val_loss`). Never overwritten by a worse validation step.
- **`checkpoint_{step}.pth.tar`**: Up to **5** recent periodic checkpoints (e.g. `checkpoint_6000.pth.tar` through `checkpoint_10000.pth.tar`). Older periodic files are pruned only **after** the newly saved checkpoint is verified via `torch.load`.

### How to Resume After a Google Colab Disconnect or Timeout
1. Reconnect to a GPU runtime in Google Colab.
2. Run **Sections 1, 2, 3, 4, and 6** (you can skip Sections 5 and 7 on resume if desired, or run all cells).
3. In **Section 6**, verify the output states:
   ```text
   CHECKPOINT STATUS : EXISTING VALID CHECKPOINT FOUND (WILL RESUME)
   ```
4. In **Section 8**, ensure `START_FULL_TRAINING = True` and run the cell.
5. `English/train.py` with `--resume` will automatically:
   - Load `checkpoint_latest.pth.tar`.
   - Restore model weights, Adam optimizer moments, Noam scheduler `current_step`, `epoch`, `best_val_loss`, and RNG states.
   - Load `training_history.csv` and `validation_history.csv`, deduplicate any steps beyond the checkpoint step, and reconstruct all 5 graphs seamlessly.

---

## 7. How to Download Final Checkpoints and Loss Graphs

All outputs are already saved in your Google Drive under `MyDrive/FastSpeech2_Training/`:
- **Best Model Checkpoint for Synthesis**: `MyDrive/FastSpeech2_Training/checkpoints/Svarah/checkpoint_best.pth.tar`
- **Latest Model Checkpoint**: `MyDrive/FastSpeech2_Training/checkpoints/Svarah/checkpoint_latest.pth.tar`
- **Combined 5-Graph Dashboard PNG**: `MyDrive/FastSpeech2_Training/logs/Svarah/plots/loss_curves_dashboard.png`
- **Individual Graphs 1–5**: `MyDrive/FastSpeech2_Training/logs/Svarah/plots/graph1_train_total_loss.png` ... `graph5_learning_rate.png`
- **Raw CSV Logs & Summary**: `MyDrive/FastSpeech2_Training/logs/Svarah/training_history.csv`, `validation_history.csv`, `training_summary.md`

You can download them directly from the Google Drive web UI or run the following snippet in a Colab cell:
```python
from google.colab import files
files.download("/content/drive/MyDrive/FastSpeech2_Training/logs/Svarah/plots/loss_curves_dashboard.png")
files.download("/content/drive/MyDrive/FastSpeech2_Training/logs/Svarah/training_summary.md")
```

---

## 8. Common Troubleshooting Steps

1. **`RuntimeError: CUDA GPU is NOT available in this Colab runtime!`**:
   - Go to **Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU**. If Colab free tier temporarily denies GPU allocation due to quota limits, wait or switch accounts; do **not** run full training on CPU.
2. **`FileNotFoundError: Missing Svarah train.txt manifest`**:
   - Ensure `FastSpeech2_Colab_Bundle.tar.gz` was extracted directly into `/content/drive/MyDrive/` so that `/content/drive/MyDrive/FastSpeech2_Training/preprocessed_data/Svarah/train.txt` exists.
3. **Slow Data Loading on Epoch 1 in Google Colab**:
   - Reading thousands of small `.npy` files over Google Drive FUSE takes longer on the very first epoch while Colab caches them. For maximum speed, copy or extract `preprocessed_data/Svarah` to `/content/preprocessed_data/Svarah` (local VM disk) and set `PREPROCESSED_DIR = Path("/content/preprocessed_data/Svarah")` in Section 1, while keeping `CHECKPOINT_DIR` and `LOG_DIR` on `/content/drive/MyDrive/...`.
4. **CUDA Out of Memory (`CUDA_OUT_OF_MEMORY`)**:
   - The default `batch_size = 16` uses ~4.5–6.0 GB VRAM on a 15 GB T4 GPU. If another process is consuming GPU memory, click **Runtime -> Restart session** and re-run.

