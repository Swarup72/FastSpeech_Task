"""Duration extraction and mel-reconciliation utility for MFA TextGrids.

This module provides tools to parse Montreal Forced Aligner (MFA) TextGrids,
validate temporal interval structure, handle silence tokens consistently with
FastSpeech 2, discretize timestamps into mel-frame durations using telescoping
rounding, and reconcile frame rounding discrepancies against mel-spectrogram
frame counts without distorting phoneme temporal alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Sequence, Any
import numpy as np

try:
    import tgt
except ImportError:
    tgt = None

# Standard silence tokens across MFA and FastSpeech 2
SILENCE_PHONES: frozenset[str] = frozenset({"", "sil", "sp", "spn"})


@dataclass
class PhoneInterval:
    """Represents a single phoneme or silence interval in time."""

    text: str
    start_time: float
    end_time: float

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def is_silence(self) -> bool:
        return self.text in SILENCE_PHONES


@dataclass
class DurationExtractionResult:
    """Result of duration extraction and mel reconciliation."""

    phones: list[str]  # Raw phone labels, e.g. ['HH', 'W', 'AH1', 'sp', ...]
    symbols: list[str]  # FastSpeech 2 symbols with '@' prefix, e.g. ['@HH', '@W', ...]
    durations: list[int]  # Frame counts per phone
    start_time: float  # Audio start timestamp in seconds (trimmed)
    end_time: float  # Audio end timestamp in seconds (trimmed)
    total_frames: int  # Total duration frames (sum of durations)
    mel_length: int | None = None  # Expected or observed mel frames
    reconciliation_applied: bool = False
    discrepancy: int = 0  # mel_length - sum(durations) before reconciliation
    fastspeech2_text: str = ""  # Formatted as "{HH W AH1 ...}" for FastSpeech 2 text frontend

    def __post_init__(self) -> None:
        if not self.fastspeech2_text and self.phones:
            self.fastspeech2_text = "{" + " ".join(self.phones) + "}"


def validate_intervals(
    intervals: Sequence[PhoneInterval],
    tolerance: float = 1e-4,
) -> None:
    """Validate interval list for consistency, monotonicity, and non-negativity.

    Args:
        intervals: Sequence of PhoneInterval objects.
        tolerance: Maximum acceptable float overlap / epsilon.

    Raises:
        ValueError: If intervals are empty, contain negative durations, or have
            unacceptable overlaps.
    """
    if not intervals:
        raise ValueError("Interval sequence cannot be empty.")

    prev_end = 0.0
    for idx, interval in enumerate(intervals):
        if interval.start_time < -tolerance:
            raise ValueError(
                f"Interval {idx} ({interval.text}) has negative start time: {interval.start_time}"
            )
        if interval.end_time < interval.start_time - tolerance:
            raise ValueError(
                f"Interval {idx} ({interval.text}) has negative duration: "
                f"[{interval.start_time}, {interval.end_time}]"
            )
        if idx > 0 and interval.start_time < prev_end - tolerance:
            raise ValueError(
                f"Overlapping intervals detected at index {idx} ({interval.text}): "
                f"starts at {interval.start_time:.4f}s before previous end {prev_end:.4f}s "
                f"(overlap = {prev_end - interval.start_time:.4f}s > {tolerance}s)"
            )
        prev_end = max(prev_end, interval.end_time)


def fill_unannotated_gaps(
    intervals: Sequence[PhoneInterval],
    tolerance: float = 1e-4,
    gap_label: str = "sp",
) -> list[PhoneInterval]:
    """Fill unannotated gaps between intervals with silence tokens.

    In some TextGrids, unaligned pause regions between spoken words are represented
    as gaps between consecutive interval boundaries rather than explicit intervals.
    This function detects any gap larger than tolerance and inserts a silence interval.

    Args:
        intervals: Sequence of PhoneInterval objects.
        tolerance: Minimum time threshold to consider as a gap.
        gap_label: Label to assign to inserted gap intervals (default 'sp').

    Returns:
        List of PhoneInterval objects with gaps filled.
    """
    if not intervals:
        return []

    filled: list[PhoneInterval] = []
    prev_end = intervals[0].start_time

    # Leading gap if any
    if prev_end > tolerance:
        filled.append(PhoneInterval(text=gap_label, start_time=0.0, end_time=prev_end))

    for interval in intervals:
        if interval.start_time > prev_end + tolerance:
            # Insert gap interval
            filled.append(
                PhoneInterval(
                    text=gap_label,
                    start_time=prev_end,
                    end_time=interval.start_time,
                )
            )
        filled.append(interval)
        prev_end = interval.end_time

    return filled


def trim_silences(
    intervals: Sequence[PhoneInterval],
    silence_phones: frozenset[str] = SILENCE_PHONES,
) -> tuple[list[PhoneInterval], float, float]:
    """Trim leading and trailing silence intervals, preserving internal pauses.

    Matches FastSpeech 2 preprocessing behavior: audio is cropped to the boundary
    of the first and last non-silence phoneme, while internal silences (inter-word
    pauses) are preserved as 'sp' tokens.

    Args:
        intervals: Sequence of PhoneInterval objects.
        silence_phones: Set of tokens considered silence.

    Returns:
        tuple of (trimmed_intervals, start_time, end_time).
        If all intervals are silence, returns ([], 0.0, 0.0).
    """
    first_non_sil = -1
    last_non_sil = -1

    for idx, interval in enumerate(intervals):
        if interval.text not in silence_phones and interval.duration_seconds > 0:
            if first_non_sil == -1:
                first_non_sil = idx
            last_non_sil = idx

    if first_non_sil == -1:
        return [], 0.0, 0.0

    start_time = intervals[first_non_sil].start_time
    end_time = intervals[last_non_sil].end_time

    # Slice between first and last non-silence intervals
    trimmed: list[PhoneInterval] = []
    for inter in intervals[first_non_sil : last_non_sil + 1]:
        # Discard true zero-duration intervals (e.g. boundary artifacts)
        if inter.duration_seconds <= 0:
            continue
        # Normalize internal silence label
        text = "sp" if inter.text in silence_phones else inter.text
        trimmed.append(
            PhoneInterval(
                text=text,
                start_time=inter.start_time,
                end_time=inter.end_time,
            )
        )

    return trimmed, start_time, end_time


def compute_frame_durations_telescoping(
    intervals: Sequence[PhoneInterval],
    start_offset: float,
    end_offset: float,
    sampling_rate: int = 22050,
    hop_length: int = 256,
    min_phone_duration: int = 1,
) -> list[int]:
    """Compute frame durations using absolute boundary discretization.

    By quantizing each interval boundary relative to the trimmed start time:
        b_i = round((t_i - t_start) * sr / hop)
        d_i = b_{i+1} - b_i
    The sum of durations satisfies the telescoping property:
        sum(d_i) = b_N - b_0 = round((t_end - t_start) * sr / hop)
    This strictly avoids any cumulative drift across intervals.

    For non-silence phonemes, if discretization results in d_i == 0 (e.g., for
    an extremely brief transient phone under ~5.8 ms), duration is set to
    min_phone_duration (default 1) to ensure the phoneme embedding is retained
    in the LengthRegulator.

    Args:
        intervals: Sequence of PhoneInterval objects.
        start_offset: Trimmed start time in seconds.
        end_offset: Trimmed end time in seconds.
        sampling_rate: Audio sampling rate in Hz.
        hop_length: STFT hop length in samples.
        min_phone_duration: Minimum duration in frames for non-silence phonemes.

    Returns:
        List of integer durations, one per interval.
    """
    if not intervals:
        return []

    durations: list[int] = []
    for interval in intervals:
        # Quantize start and end relative to trimmed start_offset
        b_start = int(
            np.round((interval.start_time - start_offset) * sampling_rate / hop_length)
        )
        b_end = int(
            np.round((interval.end_time - start_offset) * sampling_rate / hop_length)
        )
        d = max(0, b_end - b_start)

        # Enforce minimum duration for non-silence phonemes
        if not interval.is_silence and d < min_phone_duration:
            d = min_phone_duration

        durations.append(d)

    return durations


def reconcile_durations_with_mel(
    durations: list[int],
    target_length: int,
    intervals: Sequence[PhoneInterval] | None = None,
    start_offset: float = 0.0,
    sampling_rate: int = 22050,
    hop_length: int = 256,
) -> tuple[list[int], int]:
    """Reconcile frame durations to match the exact target mel spectrogram length.

    When computing mel spectrograms from audio slices (e.g. using STFT with center
    padding), the total number of mel frames may differ by +/- 1 or 2 frames from
    the quantized interval sum due to windowing boundary effects.

    This reconciliation method adjusts durations deterministically:
    1. If target_length > sum(durations) (mel has extra frames):
       Computes the fractional rounding residual for each interval:
           residual = ((end_time - start_time) * sr / hop) - duration
       Distributes extra frames to intervals with the largest positive residual
       (i.e. those that lost the most duration due to downward rounding).
       If intervals are not provided or residuals tie, prefers internal pause
       intervals ('sp') or longest vowels.
    2. If target_length < sum(durations) (mel has fewer frames):
       Finds intervals with duration > 1, sorted by largest negative residual
       (those that gained the most from upward rounding), and deducts 1 frame
       while strictly preserving duration >= 1 for all spoken phonemes.

    Args:
        durations: List of initial integer durations.
        target_length: Desired total frame count (e.g. mel_spectrogram.shape[0]).
        intervals: Optional sequence of corresponding PhoneInterval objects.
        start_offset: Offset used in duration computation.
        sampling_rate: Audio sampling rate in Hz.
        hop_length: STFT hop length in samples.

    Returns:
        tuple of (reconciled_durations, discrepancy).
    """
    reconciled = list(durations)
    current_sum = sum(reconciled)
    discrepancy = target_length - current_sum

    if discrepancy == 0 or not reconciled:
        return reconciled, 0

    n_intervals = len(reconciled)

    # Compute fractional residuals if intervals provided
    residuals: list[float] = []
    if intervals and len(intervals) == n_intervals:
        for inter, dur in zip(intervals, reconciled):
            ideal_frames = inter.duration_seconds * sampling_rate / hop_length
            residuals.append(ideal_frames - dur)
    else:
        # Fallback residuals based on duration magnitude
        residuals = [float(d) for d in reconciled]

    if discrepancy > 0:
        # Need to add frames
        needed = discrepancy
        # Sort indices by largest fractional deficit first
        # Prioritize pause intervals if available
        def add_priority(idx: int) -> tuple[int, float]:
            is_pause = (
                1
                if (intervals and idx < len(intervals) and intervals[idx].is_silence)
                else 0
            )
            return (is_pause, residuals[idx])

        sorted_indices = sorted(range(n_intervals), key=add_priority, reverse=True)
        idx_cycle = 0
        while needed > 0:
            target_idx = sorted_indices[idx_cycle % n_intervals]
            reconciled[target_idx] += 1
            needed -= 1
            idx_cycle += 1

    elif discrepancy < 0:
        # Need to remove frames
        needed = abs(discrepancy)

        # Sort indices by smallest fractional deficit (most rounded up),
        # but only among intervals where duration > 1
        def remove_priority(idx: int) -> float:
            return residuals[idx]

        sorted_indices = sorted(range(n_intervals), key=remove_priority)
        attempts = 0
        max_attempts = n_intervals * 10
        while needed > 0 and attempts < max_attempts:
            progress = False
            for target_idx in sorted_indices:
                if reconciled[target_idx] > 1:
                    reconciled[target_idx] -= 1
                    needed -= 1
                    progress = True
                    if needed == 0:
                        break
            attempts += 1
            if not progress:
                # Cannot reduce further without reducing some phone to 0
                break

    return reconciled, discrepancy


def phone_to_fastspeech2_symbol(phone: str) -> str:
    """Map a raw phone or silence label to a FastSpeech 2 vocabulary symbol.

    In FastSpeech 2:
    - ARPAbet phonemes are prepended with '@', e.g. 'AA1' -> '@AA1'.
    - Silence tokens are mapped: '' or 'sp' -> '@sp', 'sil' -> '@sil', 'spn' -> '@spn'.
    """
    if phone in ("", "sp"):
        return "@sp"
    elif phone == "sil":
        return "@sil"
    elif phone == "spn":
        return "@spn"
    elif phone.startswith("@"):
        return phone
    else:
        return f"@{phone}"


def validate_symbols_against_vocabulary(
    symbols: Sequence[str],
    vocabulary: Sequence[str] | set[str] | None = None,
) -> tuple[bool, list[str]]:
    """Validate that all symbols exist in the FastSpeech 2 vocabulary.

    Args:
        symbols: Sequence of symbol strings (with '@' prefix).
        vocabulary: Optional set or sequence of valid symbols. If None,
            loads original_fastspeech2.text.symbols.

    Returns:
        tuple of (is_valid, list_of_unknown_symbols).
    """
    if vocabulary is None:
        try:
            try:
                from English.text.symbols import symbols as fs2_symbols
            except ImportError:
                import sys

                root = Path(__file__).resolve().parent.parent.parent
                fs2_path = str(root / "original_fastspeech2")
                if fs2_path not in sys.path:
                    sys.path.insert(0, fs2_path)
                from text.symbols import symbols as fs2_symbols

            vocab_set = set(fs2_symbols)
        except Exception:
            # Fallback to standard CMUDict ARPAbet + silences if import unavailable
            from common.utils import ARPABET_FALLBACK_SYMBOLS

            vocab_set = ARPABET_FALLBACK_SYMBOLS  # type: ignore
    else:
        vocab_set = set(vocabulary)

    unknown = [s for s in symbols if s not in vocab_set]
    return len(unknown) == 0, unknown


def extract_durations_from_textgrid(
    textgrid_path_or_tier: str | Path | Any,
    target_mel_length: int | None = None,
    sampling_rate: int = 22050,
    hop_length: int = 256,
    trim_silence: bool = True,
    fill_gaps: bool = True,
    min_phone_duration: int = 1,
) -> DurationExtractionResult:
    """Extract phonemes and frame durations from an MFA TextGrid.

    End-to-end extraction pipeline:
    1. Loads the 'phones' IntervalTier from the TextGrid.
    2. Validates intervals (non-negativity, monotonicity, overlap detection).
    3. Optionally fills unannotated inter-word gaps with silence tokens ('sp').
    4. Trims leading and trailing silences if trim_silence=True, recording start/end.
    5. Computes durations via telescoping boundary discretization.
    6. Reconciles with target_mel_length if provided.
    7. Maps phonemes to FastSpeech 2 model symbols.

    Args:
        textgrid_path_or_tier: Path to .TextGrid file or a tgt.IntervalTier object.
        target_mel_length: Optional expected mel-spectrogram frame count for reconciliation.
        sampling_rate: Audio sampling rate in Hz (default 22050).
        hop_length: STFT hop length in samples (default 256).
        trim_silence: Whether to trim leading/trailing silence (default True).
        fill_gaps: Whether to fill unannotated gaps with silence (default True).
        min_phone_duration: Minimum duration in frames for spoken phones (default 1).

    Returns:
        DurationExtractionResult containing phones, symbols, durations,
        start_time, end_time, and reconciliation metadata.
    """
    # 1. Load Tier
    if isinstance(textgrid_path_or_tier, (str, Path)):
        if tgt is None:
            raise ImportError("tgt package is required to read TextGrid files.")
        tg = tgt.io.read_textgrid(
            str(textgrid_path_or_tier), include_empty_intervals=True
        )
        if "phones" not in tg.get_tier_names():
            raise ValueError(
                f"TextGrid {textgrid_path_or_tier} does not contain a 'phones' tier. "
                f"Available tiers: {tg.get_tier_names()}"
            )
        tier = tg.get_tier_by_name("phones")
    else:
        tier = textgrid_path_or_tier

    # 2. Extract PhoneIntervals
    raw_intervals: list[PhoneInterval] = []
    for interval in tier.intervals:
        raw_intervals.append(
            PhoneInterval(
                text=interval.text.strip(),
                start_time=float(interval.start_time),
                end_time=float(interval.end_time),
            )
        )

    # 3. Validate
    validate_intervals(raw_intervals)

    # 4. Fill gaps if requested
    if fill_gaps:
        intervals = fill_unannotated_gaps(raw_intervals)
    else:
        intervals = raw_intervals

    # 5. Trim silence or keep all
    if trim_silence:
        trimmed_intervals, start_time, end_time = trim_silences(intervals)
    else:
        trimmed_intervals = list(intervals)
        start_time = intervals[0].start_time if intervals else 0.0
        end_time = intervals[-1].end_time if intervals else 0.0

    if not trimmed_intervals:
        return DurationExtractionResult(
            phones=[],
            symbols=[],
            durations=[],
            start_time=start_time,
            end_time=end_time,
            total_frames=0,
            mel_length=target_mel_length,
            reconciliation_applied=False,
            discrepancy=0,
        )

    # 6. Discretize durations
    durations = compute_frame_durations_telescoping(
        trimmed_intervals,
        start_offset=start_time,
        end_offset=end_time,
        sampling_rate=sampling_rate,
        hop_length=hop_length,
        min_phone_duration=min_phone_duration,
    )

    # 7. Reconcile with target mel length if requested
    reconciliation_applied = False
    discrepancy = 0
    if target_mel_length is not None:
        durations, discrepancy = reconcile_durations_with_mel(
            durations=durations,
            target_length=target_mel_length,
            intervals=trimmed_intervals,
            start_offset=start_time,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
        )
        reconciliation_applied = discrepancy != 0

    phones = [inter.text for inter in trimmed_intervals]
    symbols = [phone_to_fastspeech2_symbol(p) for p in phones]

    return DurationExtractionResult(
        phones=phones,
        symbols=symbols,
        durations=durations,
        start_time=start_time,
        end_time=end_time,
        total_frames=sum(durations),
        mel_length=target_mel_length,
        reconciliation_applied=reconciliation_applied,
        discrepancy=discrepancy,
    )

