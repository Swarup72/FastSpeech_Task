"""High-fidelity band-limited sinc audio resampling for IndicTTS Telugu.

Standardizes all audio from heterogeneous source sampling rates:
- Female speaker (16,000 Hz) -> 22,050 Hz
- Male speaker (48,000 Hz) -> 22,050 Hz

Uses torchaudio Kaiser-windowed sinc interpolation with 64-sample lowpass
filter width and 0.99 rolloff to eliminate aliasing and phase distortion while
preserving exact duration and dynamic range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


def resample_audio(
    audio: Union[np.ndarray, torch.Tensor],
    orig_sr: int,
    target_sr: int = 22050,
) -> tuple[np.ndarray, int]:
    """Resample an audio waveform from orig_sr to target_sr.

    Args:
        audio: 1D or 2D audio array/tensor.
        orig_sr: Original sampling rate in Hz.
        target_sr: Target sampling rate in Hz (default: 22050).

    Returns:
        Tuple of (resampled_audio as np.float32, target_sr).
    """
    if orig_sr == target_sr:
        if isinstance(audio, torch.Tensor):
            return audio.cpu().numpy().astype(np.float32), target_sr
        return audio.astype(np.float32), target_sr

    # Convert to 2D tensor [channels, time]
    if isinstance(audio, np.ndarray):
        tensor = torch.from_numpy(audio).float()
    else:
        tensor = audio.float()

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2 and tensor.shape[0] > tensor.shape[1]:
        tensor = tensor.transpose(0, 1)

    # Band-limited Kaiser sinc resampling
    resampled_tensor = AF.resample(
        tensor,
        orig_freq=orig_sr,
        new_freq=target_sr,
        lowpass_filter_width=64,
        rolloff=0.99,
    )

    resampled_np = resampled_tensor.squeeze(0).cpu().numpy().astype(np.float32)

    # Prevent potential edge clipping overflow due to sinc ripples
    peak = np.max(np.abs(resampled_np))
    if peak > 1.0:
        resampled_np = resampled_np / peak * 0.999

    return resampled_np, target_sr


def resample_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_sr: int = 22050,
) -> dict:
    """Read a WAV audio file, resample to target_sr, and save to output_path.

    Args:
        input_path: Path to input audio file.
        output_path: Path to save resampled audio.
        target_sr: Target sampling rate in Hz.

    Returns:
        Dictionary containing original and resampled metrics.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data, orig_sr = sf.read(str(input_path))
    orig_duration = len(data) / orig_sr
    orig_peak = float(np.max(np.abs(data)))

    resampled_data, new_sr = resample_audio(data, orig_sr, target_sr)
    new_duration = len(resampled_data) / new_sr
    new_peak = float(np.max(np.abs(resampled_data)))

    # Save as 16-bit PCM WAV at target sampling rate
    sf.write(str(output_path), resampled_data, new_sr, subtype="PCM_16")

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "orig_sr": orig_sr,
        "target_sr": new_sr,
        "orig_duration": orig_duration,
        "target_duration": new_duration,
        "duration_diff_ms": abs(new_duration - orig_duration) * 1000.0,
        "orig_peak": orig_peak,
        "new_peak": new_peak,
    }

