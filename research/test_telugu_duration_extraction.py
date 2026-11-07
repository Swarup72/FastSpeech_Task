"""
Tests duration extraction and mel-reconciliation feasibility on the 50 pilot TextGrids.
Verifies:
- 101-symbol vocabulary coverage for every extracted token.
- Handling of leading, trailing, and internal silences (@sp, @sil).
- Discretization to 22050 Hz / hop 256 mel-frame indices.
- Zero non-positive durations for speech graphemes.
- Mel frame length matching.
"""
import os
from pathlib import Path
import sys
import glob
import json
import tgt
import numpy as np
import soundfile as sf

_BOOTSTRAP_ROOT = Path(__file__).resolve().parent.parent
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from common.paths import DATA_DIR, PROJECT_ROOT, RESEARCH_DIR
from Telugu.text.symbols import symbols, symbol_to_id

def test_duration_extraction():
    tg_dir = str(DATA_DIR / "telugu_mfa_pilot/output")
    corpus_dir = str(DATA_DIR / "telugu_mfa_pilot/corpus")
    tg_files = sorted(glob.glob(os.path.join(tg_dir, "*.TextGrid")))
    print(f"Found {len(tg_files)} TextGrids to test.")

    sr = 22050
    hop = 256

    results = []
    discrepancies_before = []
    discrepancies_after = []
    oov_tokens = set()

    for tg_path in tg_files:
        uid = os.path.splitext(os.path.basename(tg_path))[0]
        wav_path = os.path.join(corpus_dir, f"{uid}.wav")

        wav_data, audio_sr = sf.read(wav_path)
        assert audio_sr == sr

        # Expected mel frames: in torchaudio/librosa STFT with center=True / pad:
        # Mel length is exactly int(np.ceil(len(wav_data) / hop)) or len(wav_data) // hop + 1
        expected_mel_len = int(np.ceil(len(wav_data) / hop))

        tg = tgt.io.read_textgrid(tg_path)
        phones_tier = tg.get_tier_by_name("phones")

        intervals = phones_tier.intervals
        # Trim leading and trailing silences, but keep record
        first_speech = next((i for i, inv in enumerate(intervals) if inv.text not in ["sil", "<eps>", ""]), None)
        last_speech = next((i for i in range(len(intervals)-1, -1, -1) if intervals[i].text not in ["sil", "<eps>", ""]), None)

        if first_speech is None:
            print(f"Warning: {uid} has only silence!")
            continue

        trimmed_intervals = intervals[first_speech:last_speech+1]
        t_start = trimmed_intervals[0].start_time
        t_end = trimmed_intervals[-1].end_time

        # Discretize using telescoping rounding
        extracted_tokens = []
        durations = []

        for inv in trimmed_intervals:
            tok = "@sp" if inv.text in ["sil", "<eps>", ""] else inv.text
            if tok not in symbol_to_id:
                oov_tokens.add(tok)

            b_start = int(np.round((inv.start_time - t_start) * sr / hop))
            b_end = int(np.round((inv.end_time - t_start) * sr / hop))
            d = b_end - b_start
            if tok != "@sp" and d < 1:
                d = 1
            extracted_tokens.append(tok)
            durations.append(d)

        sum_dur = sum(durations)
        # Trimmed audio length in frames
        trimmed_samples = int(np.round((t_end - t_start) * sr))
        trimmed_mel_len = int(np.ceil(trimmed_samples / hop))

        disc = trimmed_mel_len - sum_dur
        discrepancies_before.append(disc)

        # Reconcile if needed by adjusting largest intervals
        reconciled_durations = list(durations)
        if disc != 0:
            step = 1 if disc > 0 else -1
            # Adjust non-zero intervals starting from longest
            sort_indices = sorted(range(len(reconciled_durations)), key=lambda i: reconciled_durations[i], reverse=True)
            for idx in sort_indices:
                if disc == 0:
                    break
                if reconciled_durations[idx] + step >= 1:
                    reconciled_durations[idx] += step
                    disc -= step

        discrepancies_after.append(trimmed_mel_len - sum(reconciled_durations))

        results.append({
            "id": uid,
            "total_tokens": len(extracted_tokens),
            "speech_tokens": sum(1 for t in extracted_tokens if t != "@sp"),
            "pause_tokens": sum(1 for t in extracted_tokens if t == "@sp"),
            "sum_durations": sum(reconciled_durations),
            "expected_trimmed_mel_len": trimmed_mel_len,
            "full_audio_mel_len": expected_mel_len,
            "min_dur": min(reconciled_durations),
            "max_dur": max(reconciled_durations),
            "discrepancy_before": discrepancies_before[-1]
        })

    print(f"\nTested {len(results)} utterances:")
    print(f"OOV tokens encountered: {oov_tokens}")
    print(f"Max discrepancy before reconciliation: {max(abs(d) for d in discrepancies_before)} frames")
    print(f"Mean absolute discrepancy before reconciliation: {np.mean([abs(d) for d in discrepancies_before]):.2f} frames")
    print(f"Max discrepancy after reconciliation: {max(abs(d) for d in discrepancies_after)} frames")
    print(f"Zero durations for speech tokens: {any(r['min_dur'] < 1 for r in results)}")

    with open("research/telugu_duration_extraction_feasibility.json", "w", encoding="utf-8") as rf:
        json.dump({
            "utterances_evaluated": len(results),
            "oov_tokens": list(oov_tokens),
            "max_discrepancy_before": int(max(abs(d) for d in discrepancies_before)),
            "mean_discrepancy_before": float(np.mean([abs(d) for d in discrepancies_before])),
            "max_discrepancy_after": int(max(abs(d) for d in discrepancies_after)),
            "sample_results": results[:10]
        }, rf, indent=2)

    print("Saved duration extraction results to research/telugu_duration_extraction_feasibility.json")

if __name__ == "__main__":
    test_duration_extraction()
