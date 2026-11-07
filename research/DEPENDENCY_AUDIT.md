# FastSpeech 2 Dependency Compatibility Audit (`DEPENDENCY_AUDIT.md`)

## 1. Overview of Original Dependency Specification

- **`requirements.txt`**: Present in `original_fastspeech2/requirements.txt` (17 pinned packages from mid-2020, targeting Python 3.7/3.8 on x86_64 Linux + CUDA).
- **`setup.py` / `pyproject.toml`**: Not present (`original_fastspeech2` is a script-based repository, not an installable package).
- **External Binary Dependency**: **Montreal Forced Aligner (MFA)** is referenced in `README.md` (`./montreal-forced-aligner/bin/mfa_align`), plus **HiFi-GAN** (`hifigan/` bundled in repo) and **MelGAN** (`torch.hub.load("descriptinc/melgan-neurips", ...)`).

---

## 2. Dependency Compatibility Table (macOS Apple Silicon M2 / Python 3.10+ `arm64`)

| Package | Required version (`requirements.txt`) | Python compatibility (Pinned vs. Modern) | macOS ARM (`arm64`) compatibility | Classification / Stage | Status & Recommended Version |
|---|---|---|---|---|---|
| `torch` | `1.7.0` | Pinned `1.7.0` supports Python 3.6–3.8 only; fails on Python >= 3.10 | **Incompatible (`1.7.0`)**: No `macosx_11_0_arm64` wheel; no MPS backend (MPS added in PyTorch 1.12). Modern `torch>=2.0` has native `arm64` + MPS support. | (1) Inspection, (3) Training, (4) Inference | **Upgrade Required** -> `torch>=2.2.0` (native `arm64` + MPS) |
| `numpy` | `1.19.0` | Pinned `1.19.0` supports Python 3.6–3.8; fails to build on Python >= 3.10 | **Incompatible (`1.19.0`)**: No `arm64` wheels; Cython/C API build failure. Note: `numpy>=2.0` breaks ABI with many audio/C extensions (observed in system Anaconda `2.2.6`). | (1) Inspection, (2) Preprocessing, (3) Training, (4) Inference | **Upgrade Required (Pin `<2.0`)** -> `numpy>=1.24.0,<2.0.0` (e.g., `1.26.4`) |
| `PyYAML` | `5.4.1` | Pinned `5.4.1` fails `pip` build under Cython >= 3.0 on Python 3.10+ (`AttributeError: cython_sources`) | **Incompatible (`5.4.1`)**: Build fails on modern `pip`. `PyYAML>=6.0.1` provides prebuilt `arm64` wheels and supports `yaml.FullLoader`. | (1) Inspection, (2) Preprocessing, (3) Training, (4) Inference | **Upgrade Required** -> `PyYAML>=6.0.1` |
| `scipy` | `1.5.0` | Pinned `1.5.0` supports Python 3.6–3.8 only | **Incompatible (`1.5.0`)**: No `arm64` wheels; requires Fortran (`gfortran`) compilation from source. `scipy>=1.10` provides native `arm64` wheels. | (2) Preprocessing, (4) Inference (`wavfile.write`, `get_window`, `interp1d`) | **Upgrade Required** -> `scipy>=1.11.0,<1.15.0` |
| `librosa` | `0.7.2` | Pinned `0.7.2` depends on `numba==0.48` & `scipy==1.5.0`; uses deprecated `np.complex`/`np.float` removed in NumPy >= 1.24 | **Incompatible (`0.7.2`)**: Fails via `numba==0.48`. However, `librosa>=0.10.0` changed `librosa.filters.mel()` and `librosa.load()` from positional to keyword-only arguments, requiring a 2-line adapter in `audio/stft.py` and `preprocessor/`. | (2) Preprocessing (`audio/stft.py`, `preprocessor/*.py`) | **Upgrade + Code Adapter** -> `librosa==0.9.2` (drop-in with `numpy<1.24`) or `librosa>=0.10.2` (with keyword-arg fix) |
| `numba` | `0.48` | Pinned `0.48` supports Python <= 3.8 only (requires `llvmlite 0.31`) | **Incompatible (`0.48`)**: `llvmlite 0.31` does not support Apple Silicon M1/M2 LLVM targets. `numba>=0.57` has native `arm64` wheels. | (2) Preprocessing (transitive dependency of `librosa`) | **Upgrade Required** -> `numba>=0.58.0` |
| `pyworld` | `0.2.10` | Pinned `0.2.10` Cython bindings fail to compile against Python 3.10+ / NumPy >= 1.24 | **Problematic (`0.2.10`)**: Requires C++ compiler + Cython build from sdist on macOS `arm64`. `pyworld>=0.3.4` supports Python 3.10+ and modern Cython. | (2) Preprocessing only (`pw.dio`, `pw.stonemask` in `preprocessor.py`) | **Upgrade Required (Preprocessing only)** -> `pyworld>=0.3.4` |
| `tgt` | `1.4.4` | Pure Python (`tgt 1.4.4` or `1.4.5`); compatible with Python 3.10–3.12 | **Compatible**: Pure Python package for reading Praat `.TextGrid` files. | (2) Preprocessing only (`preprocessor.py`) | **Compatible** -> `tgt>=1.4.4` |
| `scikit-learn` | `0.23.2` | Pinned `0.23.2` supports Python <= 3.8 only | **Incompatible (`0.23.2`)**: No `arm64` wheels. Only `StandardScaler` is used in `preprocessor.py`. `scikit-learn>=1.3` has native `arm64` wheels. | (2) Preprocessing only (`StandardScaler` in `preprocessor.py`) | **Upgrade Required** -> `scikit-learn>=1.3.0` |
| `soundfile` | `0.10.3.post1` | Wrapper over `libsndfile` | **Incompatible (`0.10.3.post1`)**: Did not bundle `arm64` `libsndfile` dylib. `soundfile>=0.12.1` bundles native `arm64` `libsndfile`. | (2) Preprocessing (used by `librosa`) | **Upgrade Required** -> `soundfile>=0.12.1` |
| `matplotlib` | `3.2.2` | Pinned `3.2.2` supports Python <= 3.8 | **Incompatible (`3.2.2`)**: No `arm64` wheel. `matplotlib>=3.7` provides native `arm64` wheels. | (3) Training, (4) Inference (`plot_mel` in `utils/tools.py`) | **Upgrade Required** -> `matplotlib>=3.7.0` |
| `tqdm` | `4.46.1` | Pure Python; works on Python 3.10+ | **Compatible**: Pure Python progress bar. | (1) Inspection, (2) Preprocessing, (3) Training | **Compatible** -> `tqdm>=4.66.0` |
| `unidecode` | `1.1.1` | Pure Python; works on Python 3.10+ | **Compatible**: Pure Python ASCII transliterator (used for English only; must NOT be used on Telugu script). | (2) Preprocessing, (3) Training, (4) Inference (`text/cleaners.py`) | **Compatible** -> `unidecode>=1.1.1` |
| `inflect` | `4.1.0` | Pure Python; `4.1.0` works on Python 3.10 | **Compatible**: Pure Python English number-to-word converter (`text/numbers.py`). | (2) Preprocessing, (3) Training, (4) Inference (English text frontend) | **Compatible** -> `inflect>=4.1.0,<7.0.0` |
| `g2p-en` | `2.1.0` | Pure Python (`g2p-en==2.1.0`), depends on `nltk`, `inflect`, `numpy`, `scipy` | **Compatible with Caveat**: Pure Python, but on first call downloads NLTK `averaged_perceptron_tagger` & `cmudict` (note: modern NLTK >= 3.8.2 renamed tagger resource to `averaged_perceptron_tagger_eng`). | (4) Inference (English OOV words in `synthesize.py`) | **Compatible** -> `g2p-en==2.1.0` |
| `pypinyin` | `0.39.0` | Pure Python | **Compatible**: Used only for Mandarin (`AISHELL3`) in `synthesize.py` (imported at top level of `synthesize.py`). | (5) Optional / Mandarin only (imported in `synthesize.py`) | **Optional** (or stub/remove import if not running Mandarin) |
| `tensorboard` | `2.2.2` | Pinned `2.2.2` requires `protobuf<4` and old `grpcio` without `arm64` wheels | **Incompatible (`2.2.2`)**: `grpcio` in 2020 lacks `arm64` wheels. `tensorboard>=2.14` works natively on `arm64`. | (3) Training only (`train.py` `SummaryWriter`) | **Colab Primary / Optional Locally** -> `tensorboard>=2.14.0` |
| **Montreal Forced Aligner (MFA)** | External binary (`mfa_align`) | External Conda/Kaldi C++ suite | **Heavy / Problematic Locally on 8 GB RAM**: Requires `kaldi`, `openfst`, `baumwelch` via `conda-forge` and consumes several GB of RAM/disk. | (2) Preprocessing (forced alignment `.TextGrid` generation) | **Run in Google Colab / Conda-Forge** (do not install in local `.venv`) |

---

## 3. Classification by Workflow Stage

1. **Required for Repository Inspection & Model Instantiation Check (Local Phase F)**:
   - `torch` (native `arm64` with MPS backend)
   - `numpy` (`<2.0`)
   - `PyYAML`
   - `scipy`
   - `matplotlib`
   - `tqdm`
   - `unidecode`
   - `inflect`
2. **Required for Lightweight Local Preprocessing / Text-Grid & Audio Inspection**:
   - All of (1) + `librosa`, `soundfile`, `tgt`, `scikit-learn`, `pyworld`
3. **Required for Full Training (Google Colab GPU Recommended)**:
   - All of (1) & (2) + `tensorboard`, `g2p-en`, unzipped `hifigan/generator_universal.pth.tar`
4. **Required for Inference (`synthesize.py`)**:
   - All of (1) + `g2p-en` (for English OOV words), `scipy` (`wavfile.write`), `matplotlib` (`plot_mel`), HiFi-GAN checkpoint (`hifigan/generator_universal.pth.tar`)
5. **Optional / Language-Specific**:
   - `pypinyin` (only used by `preprocess_mandarin` in `synthesize.py`; not needed for English or Telugu if import is guarded)
   - `MelGAN` via `torch.hub` (optional since `HiFi-GAN` is bundled locally in `hifigan/`)

---

## 4. Apple Silicon (M2, 8 GB Unified Memory) Specific Risks & Mitigations

1. **Why `pip install -r requirements.txt` Must NOT Be Run Directly**:
   - Running `pip install -r requirements.txt` on macOS `arm64` Python 3.10+ immediately fails on `PyYAML==5.4.1` (Cython 3 build error), `numpy==1.19.0` (no `arm64` wheel, C compilation error), `numba==0.48` (`llvmlite` lacks Apple Silicon target), and `torch==1.7.0` (no `arm64` wheel).
2. **NumPy 2.x ABI Breakage (Verified on Local Machine)**:
   - Inspection of the machine's `/opt/anaconda3` (`Python 3.12.7`) revealed `numpy 2.2.6` installed alongside `scipy 1.13.1` and `matplotlib`, which crashes on import with:
     `ImportError: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6 as it may crash.`
   - **Mitigation**: Explicitly constrain `numpy>=1.24.0,<2.0.0` inside `/Users/swaruppal/FastSpeech2_Project/.venv`.
3. **Hardcoded CUDA Calls in `original_fastspeech2`**:
   - `audio/stft.py` lines 68–72 (`input_data.cuda()`, `self.forward_basis.cuda()`) and line 111 (`window_sum.cuda()`) will raise `AssertionError: Torch not compiled with CUDA enabled` on Apple Silicon (`mps` or `cpu`).
   - `model/modules.py`, `utils/tools.py`, `train.py`, `evaluate.py`, and `synthesize.py` define `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
   - **PyTorch MPS Caveat**: On MPS, `torch.bucketize()` (used in `VarianceAdaptor.get_pitch_embedding` and `get_energy_embedding` in `model/modules.py` lines 83, 87, 94, 98) and certain `Conv1d` padding/striding operations should be verified on `mps` vs `cpu`. For local inspection and lightweight inference on 8 GB RAM, `cpu` or verified `mps` with `PYTORCH_ENABLE_MPS_FALLBACK=1` should be used.
4. **Packages Reserved for Google Colab vs. Local MacBook**:
   - **Install Locally in `.venv`**: Lightweight core packages (`torch`, `numpy<2.0`, `scipy`, `PyYAML`, `matplotlib`, `tqdm`, `tgt`, `unidecode`, `inflect`, `soundfile`, `librosa`, `scikit-learn`, `g2p-en`, `pyworld`).
   - **Run on Google Colab (Do NOT run heavy jobs locally)**:
     - Montreal Forced Aligner (`mfa align` / `mfa train`) on full Svarah and IndicTTS Telugu corpora.
     - Full dataset Mel/Pitch/Energy extraction (`preprocess.py`) if corpus size exceeds a few thousand utterances.
     - Multi-hundred-thousand-step FastSpeech 2 + HiFi-GAN training (`train.py`) with `tensorboard`.

