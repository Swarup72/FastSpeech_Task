# English Svarah Dataset Validation Performance Report

**Date:** 2026-11-05  
**Target Function:** [`validate_svarah_dataset()`](file:///Users/swaruppal/FastSpeech2_Project/English/monitoring.py#L159-L360) in [`English/monitoring.py`](file:///Users/swaruppal/FastSpeech2_Project/English/monitoring.py)  
**Target Dataset:** Svarah English (`3,208` training utterances + `664` validation utterances = `3,872` utterances; `3,872 × 4 = 15,488` `.npy` feature files across `mel/`, `pitch/`, `energy/`, and `duration/`)

---

## 1. Identified Bottlenecks

When `PREPROCESSED_DIR` resides on a Google Colab Google Drive FUSE mount (`/content/drive/MyDrive/...`), [`validate_svarah_dataset()`](file:///Users/swaruppal/FastSpeech2_Project/English/monitoring.py#L159-L360) in Section 4 encounters three distinct filesystem and I/O bottlenecks:

### Bottleneck A — `15,488` Individual Per-File `Path.exists()` (`os.stat`) Syscalls Across 4 Directories
- **What the previous code did**: For each of the `3,208` lines in `train.txt` and `664` lines in `val.txt` (`3,872` utterances total), `validate_svarah_dataset()` constructed four `pathlib.Path` objects (`mel_f`, `pitch_f`, `energy_f`, `dur_f`) and executed `fpath.exists()` (`os.stat()`) four times per utterance — totaling **`15,488` individual `stat()` system calls** across `mel/`, `pitch/`, `energy/`, and `duration/`.
- **Why this is slow on Google Drive FUSE**: Google Colab mounts Google Drive via a user-space FUSE daemon (`drive-filter`). Issuing `15,488` individual `LOOKUP`/`GETATTR` RPCs across 4 directories containing `3,872` entries each incurs `~1–5 ms` of FUSE IPC/metadata overhead per uncached file lookup (`~15–75+ seconds`), whereas reading the directory listing of each of the 4 feature directories once (`os.listdir()`) requires only **4 bulk `READDIR` calls**.

### Bottleneck B — Multi-Worker `DataLoader` Spawning & Over-Prefetching (`1,024` `.npy` Files) During `_inspect_batch(train_loader, "train")`
- **What happens in Section 4**: In the Colab notebook, `train_loader` is configured with `batch_size = 16 * 4 = 64` and `num_workers = 2` (when `DEVICE.type == "cuda"` on an L4/T4 GPU).
- **Why `next(iter(train_loader))` is slow**:
  1. Calling `iter(train_loader)` spawns **2 worker subprocesses** (`~2.1–2.3s` process creation + PyTorch IPC initialization overhead even on local SSD).
  2. With PyTorch's default `prefetch_factor = 2`, the 2 workers immediately begin prefetching **`2 workers × 2 batches = 4` outer batches of `64` utterances = `256` randomly shuffled utterances = `1,024` `.npy` file reads** (`256` mel + `256` pitch + `256` energy + `256` duration) plus `256` phoneme conversions concurrently over Google Drive FUSE.
  3. On cold Google Drive FUSE—where the Colab L4 diagnostic showed individual `.npy` reads take **`~14.5 ms/file` on average** (`17.91s` for `3,872` mel files [`4.6 ms/file`], `88.38s` for `3,872` pitch files [`22.8 ms/file`], `56.31s` for `3,872` energy files [`14.5 ms/file`], `62.04s` for `3,872` duration files [`16.0 ms/file`], totaling `224.64s` for all `15,488` files)—concurrently reading `1,024` uncached random `.npy` files for `train_loader` plus `256` `.npy` files for `val_loader` (`64 × 4`) plus `120` `.npy` files for the deep array check (`30 × 4`) = **`1,400` `.npy` file opens/reads** takes **`~20–25 seconds` of FUSE read latency alone**, even though `_inspect_batch` only inspects the first batch (`64` samples) and immediately discards the other `3` prefetched batches (`768` `.npy` files) when the temporary iterator exits.

### Bottleneck C — Repeated Filesystem Operations Across Section 4 & Section 5
Tracing every filesystem call across Section 4 (`Dataset` initialization + `validate_svarah_dataset()`) and Section 5 (`run_model_smoke_test()`) identified four categories of repeated filesystem operations:

| Repeated Filesystem Operation | Count Before Optimization | Where It Occurs | Impact |
| :--- | :---: | :--- | :--- |
| **Per-file `Path.exists()` (`os.stat`) calls in `mel/`, `pitch/`, `energy/`, `duration/`** | **`15,488` `stat()` calls** | [`validate_svarah_dataset()`](file:///Users/swaruppal/FastSpeech2_Project/English/monitoring.py#L224-L266) manifest loop | **High on Google Drive FUSE** (`15,488` individual metadata lookups instead of `4` directory scans). **Eliminated (`15,488 → 0` per-file `stat` calls when files exist).** |
| **Re-reading `idx 0..14` of `val.txt` (`60` `.npy` files)** | `2×` (`60` `.npy` files read twice) | Stage 3 deep array check (`idx < 15`) loads `val[0..14]`, then Stage 5 `next(iter(val_loader))` (`shuffle=False`) loads `val[0..63]` | Low once in OS page cache (`~1.5 ms/file`, as shown by `20` Mel files loading in `0.03s`), since Stage 3 warms the Linux page cache for `val[0..14]`. |
| **Re-reading `train.txt`, `val.txt`, `speakers.json`, `stats.json`** | `2–3×` each (`10` file opens total) | `Dataset("train.txt")`, `Dataset("val.txt")`, and `validate_svarah_dataset()` | Negligible (`< 6 ms` total across all JSON and manifest reads). |
| **Re-iterating `train_loader` and `val_loader` in Section 5 (`run_model_smoke_test`)** | `2×` (`Section 4` then `Section 5`) | `validate_svarah_dataset()` followed by `run_model_smoke_test()` | High on Drive FUSE if `train_loader` has `num_workers=2` and `PREPROCESSED_DIR` is on FUSE instead of local `/content` SSD. |

---

## 2. Measured Stage Timings

Using the new high-resolution `time.perf_counter()` instrumentation added to [`validate_svarah_dataset()`](file:///Users/swaruppal/FastSpeech2_Project/English/monitoring.py#L159-L360), here is the exact stage-by-stage timing breakdown across all `3,872` Svarah utterances (`15,488` feature files):

| Validation Stage | Filesystem Operations Performed | Before Optimization (Local SSD) | After Optimization (Local SSD, Warm Cache) | After Optimization (Local SSD, Cold Dir Scan) | Colab Google Drive FUSE Behaviour |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1. Metadata & Manifest Reading** (`metadata_and_manifests_sec`) | Check & read `speakers.json`, `stats.json`, `train.txt` (`3,208` lines), `val.txt` (`664` lines) | `3.56 ms` | **`0.90 ms`** | **`0.90 ms`** | `< 50 ms` (4 sequential text/JSON files) |
| **2. Directory Index Scan** (`directory_index_scan_sec`) | `4 × os.listdir()` on `mel/`, `pitch/`, `energy/`, `duration/` (`3,872` entries each) | N/A *(scanned per-file)* | **`8.08 ms`** | **`128.00 ms`** | **4 bulk `READDIR` FUSE calls** (`~0.1–0.4s` total) |
| **3. Feature Existence Checks** (`feature_existence_checks_sec`) | Verify all 4 features (`mel`, `pitch`, `energy`, `duration`) for all `3,872` utterances (`15,488` checks) | `80.59 ms` (`15,488` `os.stat` syscalls) | **`4.44 ms`** (`15,488` O(1) set checks, `0` `stat` syscalls) | **`4.60 ms`** (`15,488` O(1) set checks, `0` `stat` syscalls) | **`~5 ms` in-memory set checks** (saves `15,488` individual FUSE `stat` calls = **`~15–60s` saved**) |
| **4. Deep Array Consistency Checks** (`deep_array_checks_sec`) | `np.load()` + `np.isfinite()` + frame alignment on 15 train + 15 val utterances (`120` `.npy` files) | `29.04 ms` | **`32.20 ms`** | **`32.20 ms`** | `120` `.npy` reads (`~1.5–2.0s` on Drive FUSE; `< 0.05s` on `/content` local SSD) |
| **5A. Train `DataLoader` Init & 1st Batch (`num_workers=0`)** (`train_dataloader_total_sec`) | `iter(train_loader)` + `next()` (`64` utterances = `256` `.npy` files + `64` G2P + batch tensor checks) | `110.97 ms` | **`171.20 ms`** (`init=7.2ms`, `batch=164.1ms`) | **`171.20 ms`** | `256` `.npy` reads (`~3.5–4.0s` on Drive FUSE; `< 0.2s` on `/content` local SSD) |
| **5B. Train `DataLoader` Init & 1st Batch (`num_workers=2`)** | Spawns 2 workers + prefetches `4 × 64 = 256` utterances (`1,024` `.npy` files) + validates 1st batch | `2,291.99 ms` (**`20.6×` slower**) | `2,291.99 ms` (if `num_workers=2` passed) | `2,291.99 ms` | Spawns 2 workers + `1,024` concurrent random FUSE `.npy` reads (`~15–20s` on Drive FUSE) |
| **6. Val `DataLoader` Init & 1st Batch (`num_workers=0`)** (`val_dataloader_total_sec`) | `iter(val_loader)` + `next()` (`64` utterances = `256` `.npy` files, `60` already cached in Stage 4) | `101.58 ms` | **`70.60 ms`** (`init=0.0ms`, `batch=70.6ms`) | **`70.60 ms`** | `196` uncached + `60` cached `.npy` reads (`~2.5–3.0s` on Drive FUSE; `< 0.1s` on `/content` local SSD) |
| **Total (`num_workers=0`)** (`total_validation_sec`) | Full `validate_svarah_dataset()` execution | `592.92 ms` | **`287.42 ms`** | **`408.00 ms`** | **`~6–9s` on Drive FUSE** (down from `~30–90s+`), or **`< 0.5s` on `/content` local SSD** |

---

## 3. Implemented Optimization & Correctness Guarantees

In [`English/monitoring.py`](file:///Users/swaruppal/FastSpeech2_Project/English/monitoring.py#L159-L360), we implemented a minimal, 100% backward-compatible improvement to [`validate_svarah_dataset()`](file:///Users/swaruppal/FastSpeech2_Project/English/monitoring.py#L159-L360):

1. **Single-Pass Directory Indexing (`4 × os.listdir`) with Fallback `Path.exists()` Verification**:
   - Scans `mel/`, `pitch/`, `energy/`, and `duration/` once before the manifest loop (`mel_entries`, `pitch_entries`, `energy_entries`, `dur_entries`).
   - For every manifest line (`3,208` train + `664` val = `3,872` utterances), checks all four expected filenames (`{spk}-mel-{basename}.npy`, `{spk}-pitch-{basename}.npy`, `{spk}-energy-{basename}.npy`, `{spk}-duration-{basename}.npy`) against the corresponding directory set (`15,488` checks).
   - If any filename is not in the directory set, falls back to `fpath.exists()` and raises the exact same `FileNotFoundError(f"Missing feature file: {fpath}")`.
2. **Stage-Level Timing Instrumentation (`timings_sec` + Console Summary)**:
   - Separately measures and reports:
     - `metadata_and_manifests_sec`
     - `directory_index_scan_sec`
     - `feature_existence_checks_sec`
     - `deep_array_checks_sec`
     - `train_dataloader_iter_init_sec`, `train_dataloader_first_batch_sec`, `train_dataloader_total_sec`
     - `val_dataloader_iter_init_sec`, `val_dataloader_first_batch_sec`, `val_dataloader_total_sec`
     - `total_validation_sec`
   - Prints a single structured timing line:
     ```text
     [Dataset Validation Timings] manifests=0.001s | dir_scan=0.128s | existence(15488 files)=0.005s | deep_arrays(30 utts)=0.032s | train_loader=0.171s | val_loader=0.071s | total=0.408s
     ```
3. **Strict Correctness & Backward-Compatibility Guarantees**:
   - Function signature `validate_svarah_dataset(preprocessed_path, model_config, train_loader, val_loader, expected_vocab_size=360, max_file_checks=None)` is **unchanged**.
   - Every existing return dictionary key (`status`, `train_utterances`, `val_utterances`, `num_speakers`, `vocab_size`, `deep_checked_arrays`, `train_batch`, `val_batch`) is **unchanged**, with `checked_feature_files` (`15488`) and `timings_sec` added non-destructively.
   - Zero manifest entries are skipped (`3,208 + 664 = 3,872`).
   - All `15,488` feature-file existence checks (`4` per utterance) are performed in the exact same order.
   - All numerical finiteness (`np.isfinite`, `torch.isfinite`) and frame-alignment (`mel_frames == sum(duration) == pitch_frames == energy_frames` and batch `sum(durations, axis=1) == mel_lens`) checks are preserved verbatim.

---

## 4. Recommended Optimizations for Google Drive Access on Colab

Because Google Drive FUSE (`/content/drive/MyDrive/...`) has a `~5–20 ms` per-file open/read latency (which caused loading all `15,488` `.npy` files manually to take `17.91s + 88.38s + 56.31s + 62.04s = 224.64s`, and would cause every training epoch of `3,208 × 4 = 12,832` random `.npy` reads to spend `~180s` waiting on Drive FUSE I/O!):

1. **Stage `preprocessed_data/Svarah` to Colab's Local NVMe SSD (`/content/FastSpeech2_Project/preprocessed_data/Svarah`) While Keeping Checkpoints & Logs on Google Drive**:
   - Copying or extracting `preprocessed_data/Svarah` (`~398 MB` compressed / `15,488` files) once at the start of the Colab session from Google Drive to `/content/...` (local SSD) takes **~4–8 seconds** sequential archive read, after which:
     - `validate_svarah_dataset()` runs in **`~0.4 seconds`** (instead of `30–90+ seconds`).
     - Training epochs load `12,832` `.npy` files from local NVMe SSD in **`< 1.5 seconds/epoch`** instead of **`~180 seconds/epoch`** over Google Drive FUSE!
   - Meanwhile, `CHECKPOINT_DIR` (`/content/drive/MyDrive/FastSpeech2_Training/checkpoints/Svarah`) and `LOG_DIR` (`/content/drive/MyDrive/FastSpeech2_Training/logs/Svarah`) **remain directly on Google Drive** for 100% persistent checkpoint/log recovery.
2. **Avoid Full-Corpus Manual `np.load()` Scans on Google Drive FUSE**:
   - Manually calling `np.load()` on all `15,488` `.npy` files directly over `/content/drive/...` takes **`224.64 seconds`** (`3.75 minutes`) solely due to FUSE per-file open latency. Rely on `validate_svarah_dataset()`'s `4`-directory index scan (`15,488` existence checks in `~5 ms`) + deep array checks (`120` files) + batch checks, or run full-corpus `.npy` scans against `/content/...` local SSD where all `15,488` `.npy` files load in **`~2.8 seconds`**.
3. **Pass `num_workers=0` (or Use Local SSD) During Single-Batch Diagnostic Validation**:
   - When `PREPROCESSED_DIR` is on Google Drive FUSE, a `DataLoader` with `num_workers=2` spawns 2 subprocesses and prefetches `4` outer batches (`256` utterances = `1,024` random `.npy` files) on `next(iter(train_loader))`, adding `~15–20s` of wasted FUSE reads for 3 prefetched batches that are immediately discarded.

---

## 5. Regression Test Results

Added `test_07_validate_svarah_dataset_timings_and_regression_guarantees` in [`tests/test_colab_monitoring_and_notebook.py`](file:///Users/swaruppal/FastSpeech2_Project/tests/test_colab_monitoring_and_notebook.py#L410-L501) verifying:
- All `3,208` training and `664` validation utterances (`15,488` feature files) are checked (`checked_feature_files == 15488`).
- All 11 stage timing keys are present and non-negative in `report["timings_sec"]`.
- Missing file detection raises `FileNotFoundError` for each of `mel`, `pitch`, `energy`, and `duration`.
- `NaN`/`Inf` detection in `.npy` arrays raises `ValueError`.
- Frame alignment mismatch (`sum(duration) != mel_frames`) raises `ValueError`.

Full regression suite status: **207 / 207 tests passing (`OK`)**.

