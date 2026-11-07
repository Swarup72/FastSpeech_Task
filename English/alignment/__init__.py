"""English forced alignment and duration extraction modules."""

from English.alignment.duration import (
    DurationExtractionResult,
    PhoneInterval,
    compute_frame_durations_telescoping,
    extract_durations_from_textgrid,
    fill_unannotated_gaps,
    phone_to_fastspeech2_symbol,
    reconcile_durations_with_mel,
    trim_silences,
    validate_intervals,
    validate_symbols_against_vocabulary,
)

__all__ = [
    "DurationExtractionResult",
    "PhoneInterval",
    "compute_frame_durations_telescoping",
    "extract_durations_from_textgrid",
    "fill_unannotated_gaps",
    "phone_to_fastspeech2_symbol",
    "reconcile_durations_with_mel",
    "trim_silences",
    "validate_intervals",
    "validate_symbols_against_vocabulary",
]

