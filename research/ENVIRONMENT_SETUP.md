# Local Environment Setup & Verification Report (`ENVIRONMENT_SETUP.md`)

## 1. System Specifications (Phase A Inspection)

| Property | Recorded Value | Command Used |
|---|---|---|
| **Device** | MacBook with Apple Silicon M2 | Hardware spec / `uname -m` |
| **Operating System** | macOS `27.0.1` (Build `26A434`) | `sw_vers` |
| **CPU Architecture** | `arm64` (Native 64-bit Apple Silicon) | `uname -m` |
| **Unified Memory (RAM)** | `8,589,934,592` bytes (**8.00 GB**) | `sysctl -n hw.memsize` |
| **Disk Space (`/`)** | **228 GiB** total, **13 GiB** used, **51 GiB** available (`20%` capacity) | `df -h /` |
| **Git Version** | `git version 2.53.0` | `git --version` |
| **Homebrew** | Installed at `/opt/homebrew/bin/brew` | `which brew` |
| **System Default `python3`** | `Python 3.12.7` (`/opt/anaconda3/bin/python3`, `arm64`) | `python3 --version`, `which python3` |
| **Available Local Python Interpreters** | - `/opt/homebrew/bin/python3.10` (`Python 3.10.18`, `arm64`)<br>- `/opt/homebrew/bin/python3.11` (`Python 3.11.15`, `arm64`)<br>- `/opt/anaconda3/bin/python3.12` (`Python 3.12.7`, `arm64`)<br>- `/opt/anaconda3/envs/torch_m2/bin/python` (`Python 3.10.19`, `arm64`) | `which -a python3 python3.10 python3.11 python3.12` |

---

## 2. Virtual Environment Configuration (Phase C & Phase F)

- **Selected Interpreter**: `/opt/homebrew/bin/python3.10` (`Python 3.10.18`, native `arm64`)
- **Rationale for Choosing Python 3.10**:
  - `original_fastspeech2/requirements.txt` was written in 2020 for Python 3.7/3.8.
  - Python 3.12+ removed standard library `imp` and `distutils`, breaking legacy speech/text packaging scripts, and the system `/opt/anaconda3` (`Python 3.12.7`) exhibited a broken NumPy 2.2.6 vs. SciPy 1.13.1/Matplotlib ABI mismatch.
  - Python 3.10 (`3.10.18` `arm64`) provides maximum compatibility across modern native Apple Silicon wheels (`torch 2.10.0`, `torchaudio 2.10.0`, `PyYAML 6.0.3`, `tgt 1.5`, `inflect 4.1.0`, `unidecode 1.3.8`, `tqdm 4.66.5`).
- **Virtual Environment Location**: `/Users/swaruppal/FastSpeech2_Project/.venv`
- **Prompt Name**: `fastspeech2-env`
- **Active Interpreter Inside `.venv`**: `/Users/swaruppal/FastSpeech2_Project/.venv/bin/python` (`Python 3.10.18`)
- **Active `pip` Version**: `pip 26.2.2`

### Commands Used to Create, Activate, and Configure the Environment

```bash
cd /Users/swaruppal/FastSpeech2_Project

# 1. Create isolated Python 3.10 virtual environment
/opt/homebrew/bin/python3.10 -m venv --prompt fastspeech2-env .venv

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Upgrade core packaging tools
python -m pip install --upgrade pip setuptools wheel

# 4. Install lightweight inspection dependencies & reuse local arm64 PyTorch 2.10.0
pip install PyYAML tgt "inflect==4.1.0"
```

---

## 3. Installed Package Versions in `.venv`

| Package | Installed Version | Purpose in FastSpeech 2 |
|---|---|---|
| `torch` | `2.10.0` (`arm64` + MPS) | Core deep learning framework (`FastSpeech2`, `HiFi-GAN`, `STFT`) |
| `torchaudio` | `2.10.0` (`arm64`) | Audio I/O and resampling utilities |
| `numpy` | `2.2.6` | Numerical array operations and `.npy` feature loading |
| `PyYAML` | `6.0.3` | Parsing `preprocess.yaml`, `model.yaml`, `train.yaml` |
| `tgt` | `1.5` | Praat `.TextGrid` MFA alignment parser (`preprocessor.py`) |
| `inflect` | `4.1.0` | English number-to-word normalization (`text/numbers.py`) |
| `Unidecode` | `1.3.8` | English ASCII transliteration (`text/cleaners.py`) |
| `tqdm` | `4.66.5` | Progress bars for preprocessing and training |
| `pillow` | `12.1.0` | Image rendering support |
| `sympy` | `1.14.0` | Symbolic math dependency for `torch` |
| `networkx` | `3.4.2` | Graph dependency for `torch` |
| `Jinja2` | `3.1.6` | Template dependency for `torch` |
| `filelock` | `3.20.3` | File locking utility for `torch` |
| `fsspec` | `2026.1.0` | Filesystem specification for `torch` |
| `pip` | `26.2.2` | Package installer |
| `setuptools` | `84.2.1` | Build system |
| `wheel` | `0.48.0` | Binary wheel support |

---

## 4. PyTorch & Apple Silicon MPS Verification

Executed inside `/Users/swaruppal/FastSpeech2_Project/.venv/bin/python`:

```python
import torch

print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
print("MPS matmul test:\n", torch.matmul(x, x))
```

### Actual Verification Output

```text
PyTorch version: 2.10.0
MPS available: True
MPS built: True
MPS matmul test:
 tensor([[ 7., 10.],
        [15., 22.]], device='mps:0')
```

### FastSpeech 2 Text Frontend + MPS Embedding Smoke Test Output

```text
Total symbols in vocabulary: 360
Sample ARPAbet sequence IDs: [91, 112, 131, 123, 143, 112, 73, 133, 357, 129, 97, 118, 109, 132, 73, 119]
Round-trip sequence_to_text: {DH IY0 S OW1 V IY0 AH0 T sp P ER0 M IH1 SH AH0 N}
MPS Embedding output shape & device: torch.Size([1, 16, 256]) mps:0
```

---

## 5. Encountered Issues & Resolutions

1. **System Anaconda Base Environment NumPy 2.x Conflict**:
   - **Observation**: The default system `python3` (`/opt/anaconda3/bin/python3`, Python 3.12.7) had `numpy 2.2.6` installed alongside `scipy 1.13.1` and `matplotlib 3.9.2` compiled against NumPy 1.x, causing `ImportError: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6` when importing `scipy` or `matplotlib`.
   - **Resolution**: Created a clean, isolated Python 3.10 virtual environment at `/Users/swaruppal/FastSpeech2_Project/.venv` using `/opt/homebrew/bin/python3.10`. When installing `scipy`, `matplotlib`, or `librosa` in `.venv` for local plotting/preprocessing, either use wheels compiled for NumPy 2.x or pin `numpy==1.26.4`.
2. **Original `requirements.txt` Incompatibility on Apple Silicon**:
   - **Observation**: The 2020 pins (`torch==1.7.0`, `numpy==1.19.0`, `numba==0.48`, `PyYAML==5.4.1`) have no `macosx_arm64` wheels and fail to build on Python 3.10+.
   - **Resolution**: Avoided running `pip install -r requirements.txt` blindly; audited all 17 dependencies in `DEPENDENCY_AUDIT.md` and installed modern `arm64`-compatible equivalents required for inspection.

