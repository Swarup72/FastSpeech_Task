"""Full IndicMFA Forced Alignment Pipeline for Telugu Option 1B (Train + Val Splits Only).

Aligns:
- `Telugu/metadata/splits/option_1b/train.txt`: 2,302 utterances (~5.3289 hours)
- `Telugu/metadata/splits/option_1b/val.txt`  :   422 utterances (~1.0000 hour)
Total: 2,724 utterances (~6.3289 hours)

Strict guarantees:
1. NEVER reads audio for or aligns `test.txt` (held-out test split remains untouched).
2. Preserves existing pilot outputs (`data/telugu_mfa_pilot/` and `research/telugu_mfa_pilot_*`).
3. Preserves original transcripts and approved context-sensitive Latin-o -> Anusvara normalization.
4. Processes in manageable batches with crash-safe resume (skips already verified TextGrids).
5. Automatically retries any batch alignment failure with expanded beam (`--beam 100 --retry_beam 400`).
6. Verifies telescoping mel-frame duration extraction (`sr=22050, hop=256`) and records all
   pre- and post-reconciliation frame discrepancies without extracting `.npy` features yet.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import tgt

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from common.paths import DATA_DIR, PROJECT_ROOT, RESEARCH_DIR, TELUGU_DIR
from Telugu.alignment.resample import resample_audio
from Telugu.text.normalizer import normalize_text
from Telugu.text.symbols import symbol_to_id

OPTION_1B_SPLIT_DIR = TELUGU_DIR / "metadata" / "splits" / "option_1b"
ALIGNMENT_META_DIR = OPTION_1B_SPLIT_DIR / "alignment"
TEXTGRID_ROOT_DIR = DATA_DIR / "telugu_mfa_option_1b" / "TextGrid"
STAGING_ROOT_DIR = DATA_DIR / "telugu_mfa_option_1b" / "staging"

MFA_BIN = Path("/opt/anaconda3/envs/mfa_env/bin/mfa")
DICT_PATH = RESEARCH_DIR / "Telugu_Dictionary_g2g.txt"
MODEL_PATH = DATA_DIR / "telugu_mfa_model" / "Telugu_Acoustic_Model.zip"
PARQUET_GLOB = str(DATA_DIR / "telugu_parquet" / "data" / "train-*-of-00004.parquet")

TARGET_SR = 22050
HOP_LENGTH = 256
PUNCT_CHARS = set(" .,!?;:'\"-")


def _get_ram_usage_mb() -> float:
    """Return current process / system RAM usage in MB."""
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss / (1024 * 1024))
    except Exception:
        rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = (1024 * 1024) if sys.platform == "darwin" else 1024
        return float(rusage / divisor)


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via a temporary file rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _load_split_entries(split_txt_path: Path, split_name: str) -> list[dict[str, Any]]:
    """Parse Option 1B split file (`uid|speaker|raw|norm|dur|gender`)."""
    entries: list[dict[str, Any]] = []
    with open(split_txt_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line_s = line.strip()
            if not line_s:
                continue
            parts = line_s.split("|")
            if len(parts) != 6:
                raise ValueError(f"Malformed line {idx} in {split_txt_path}: {line_s}")
            uid, spk_id, raw_text, norm_text, dur_str, gender_str = parts
            clean_text = normalize_text(raw_text)
            chars = [c for c in clean_text if c not in PUNCT_CHARS]
            spaced_mfa_text = " ".join(chars)
            entries.append({
                "id": uid,
                "split": split_name,
                "speaker_id": int(spk_id),
                "gender": gender_str,
                "raw_text": raw_text,
                "normalized_text": clean_text,
                "manifest_norm_text": norm_text,
                "spaced_mfa_text": spaced_mfa_text,
                "expected_grapheme_count": len(chars),
                "manifest_duration_sec": float(dur_str),
            })
    return entries


def _load_parquet_audio_for_ids(target_ids: set[str]) -> dict[str, bytes]:
    """Load raw WAV bytes from the 4 Parquet shards using O(1) row-group indexing."""
    with open(PROJECT_ROOT / "Telugu" / "metadata" / "text_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    uid_to_row_idx = {os.path.splitext(r["path"])[0]: int(r["row_idx"]) for r in manifest}
    shard_paths = sorted(glob.glob(PARQUET_GLOB))
    shards = [pq.ParquetFile(p) for p in shard_paths]

    # Group requested target_ids by (shard_idx, row_group_idx) so each needed row group is read at most once
    rg_groups: dict[tuple[int, int], list[tuple[str, int]]] = {}
    for uid in target_ids:
        if uid not in uid_to_row_idx:
            raise KeyError(f"Utterance ID {uid} not found in text_manifest.json")
        r_idx = uid_to_row_idx[uid]
        s_idx = r_idx // 2144
        l_idx = r_idx % 2144
        rg_idx = l_idx // 100
        rg_off = l_idx % 100
        rg_groups.setdefault((s_idx, rg_idx), []).append((uid, rg_off))

    audio_map: dict[str, bytes] = {}
    for (s_idx, rg_idx), items in sorted(rg_groups.items()):
        t = shards[s_idx].read_row_group(rg_idx, columns=["audio"])
        audio_col = t["audio"]
        for uid, rg_off in items:
            audio_map[uid] = audio_col[rg_off].as_py()["bytes"]

    return audio_map


def _parse_and_verify_textgrid(
    tg_path: Path,
    entry: dict[str, Any],
    resampled_num_samples: int,
    original_sr: int,
    dict_words: set[str],
) -> dict[str, Any]:
    """Parse TextGrid, verify interval integrity, and verify telescoping mel-frame durations."""
    uid = entry["id"]
    actual_dur_sec = resampled_num_samples / float(TARGET_SR)
    full_audio_mel_frames = int(np.ceil(resampled_num_samples / float(HOP_LENGTH)))

    clean_chars = [c for c in entry["normalized_text"] if c not in PUNCT_CHARS]
    dict_oov = sorted({c for c in clean_chars if c not in dict_words})

    result: dict[str, Any] = {
        "id": uid,
        "split": entry["split"],
        "speaker_id": entry["speaker_id"],
        "gender": entry["gender"],
        "raw_text": entry["raw_text"],
        "normalized_text": entry["normalized_text"],
        "spaced_mfa_text": entry["spaced_mfa_text"],
        "original_sr": original_sr,
        "target_sr": TARGET_SR,
        "resampled_num_samples": resampled_num_samples,
        "actual_wav_duration_sec": round(actual_dur_sec, 6),
        "full_audio_mel_frames": full_audio_mel_frames,
        "textgrid_path": str(tg_path.relative_to(PROJECT_ROOT)),
        "textgrid_exists": tg_path.exists(),
        "status": "failed",
        "failure_reason": None,
        "words_tier_intervals": 0,
        "phones_tier_intervals": 0,
        "speech_intervals_count": 0,
        "silence_intervals_count": 0,
        "initial_silence_sec": 0.0,
        "final_silence_sec": 0.0,
        "total_speech_sec": 0.0,
        "total_silence_sec": 0.0,
        "trimmed_start_sec": 0.0,
        "trimmed_end_sec": 0.0,
        "expected_trimmed_mel_frames": 0,
        "raw_duration_sum": 0,
        "discrepancy_before_reconciliation": 0,
        "reconciled_duration_sum": 0,
        "discrepancy_after_reconciliation": 0,
        "speech_token_count": 0,
        "pause_token_count": 0,
        "min_speech_frame_dur": 0,
        "max_speech_frame_dur": 0,
        "dict_oov_tokens": dict_oov,
        "vocab_oov_tokens": [],
        "unknown_phones": [],
        "malformed_intervals": [],
        "extracted_tokens": [],
        "reconciled_durations": [],
    }

    if not tg_path.exists():
        result["failure_reason"] = "missing_textgrid"
        return result

    try:
        tg = tgt.io.read_textgrid(str(tg_path))
        words_tier = tg.get_tier_by_name("words")
        phones_tier = tg.get_tier_by_name("phones")
    except Exception as e:
        result["failure_reason"] = f"textgrid_parse_error: {e}"
        return result

    intervals = phones_tier.intervals
    result["words_tier_intervals"] = len(words_tier.intervals)
    result["phones_tier_intervals"] = len(intervals)

    if not intervals:
        result["failure_reason"] = "empty_phones_tier"
        return result

    malformed: list[str] = []
    unknown_phones: list[dict[str, Any]] = []
    sil_count = 0
    speech_count = 0
    sil_dur = 0.0
    speech_dur = 0.0

    for i, inv in enumerate(intervals):
        dur = inv.end_time - inv.start_time
        if dur <= 0:
            malformed.append(f"interval_{i}_non_positive_dur_{inv.start_time}_{inv.end_time}")
        if i > 0 and inv.start_time < intervals[i - 1].end_time - 1e-4:
            malformed.append(f"interval_{i}_overlap_{inv.start_time}_lt_{intervals[i - 1].end_time}")

        if inv.text in ("sil", "<eps>", ""):
            sil_count += 1
            sil_dur += dur
        else:
            speech_count += 1
            speech_dur += dur

        if inv.text in ("spn", "<unk>"):
            unknown_phones.append({
                "interval_idx": i,
                "phone": inv.text,
                "start_time": round(float(inv.start_time), 4),
                "end_time": round(float(inv.end_time), 4),
            })

    if intervals[0].text in ("sil", "<eps>", ""):
        result["initial_silence_sec"] = round(float(intervals[0].end_time - intervals[0].start_time), 6)
    if intervals[-1].text in ("sil", "<eps>", ""):
        result["final_silence_sec"] = round(float(intervals[-1].end_time - intervals[-1].start_time), 6)

    result["silence_intervals_count"] = sil_count
    result["speech_intervals_count"] = speech_count
    result["total_silence_sec"] = round(float(sil_dur), 6)
    result["total_speech_sec"] = round(float(speech_dur), 6)
    result["unknown_phones"] = unknown_phones
    result["malformed_intervals"] = malformed

    first_speech = next(
        (i for i, inv in enumerate(intervals) if inv.text not in ("sil", "<eps>", "")),
        None,
    )
    last_speech = next(
        (i for i in range(len(intervals) - 1, -1, -1) if intervals[i].text not in ("sil", "<eps>", "")),
        None,
    )
    if first_speech is None or last_speech is None:
        result["failure_reason"] = "no_speech_intervals_in_textgrid"
        return result

    trimmed_intervals = intervals[first_speech : last_speech + 1]
    t_start = float(trimmed_intervals[0].start_time)
    t_end = float(trimmed_intervals[-1].end_time)
    result["trimmed_start_sec"] = round(t_start, 6)
    result["trimmed_end_sec"] = round(t_end, 6)

    extracted_tokens: list[str] = []
    raw_durations: list[int] = []
    vocab_oov: set[str] = set()

    for inv in trimmed_intervals:
        tok = "@sp" if inv.text in ("sil", "<eps>", "") else inv.text
        if tok not in symbol_to_id:
            vocab_oov.add(tok)
        b_start = int(np.round((inv.start_time - t_start) * TARGET_SR / float(HOP_LENGTH)))
        b_end = int(np.round((inv.end_time - t_start) * TARGET_SR / float(HOP_LENGTH)))
        d = b_end - b_start
        if tok != "@sp" and d < 1:
            d = 1
        extracted_tokens.append(tok)
        raw_durations.append(int(d))

    raw_dur_sum = int(sum(raw_durations))
    trimmed_samples = int(np.round((t_end - t_start) * TARGET_SR))
    expected_trimmed_mel_len = int(np.ceil(trimmed_samples / float(HOP_LENGTH)))
    disc_before = expected_trimmed_mel_len - raw_dur_sum

    reconciled_durations = list(raw_durations)
    rem_disc = disc_before
    if rem_disc != 0:
        step = 1 if rem_disc > 0 else -1
        sort_indices = sorted(
            range(len(reconciled_durations)),
            key=lambda idx: reconciled_durations[idx],
            reverse=True,
        )
        for idx in sort_indices:
            if rem_disc == 0:
                break
            min_allowed = 0 if extracted_tokens[idx] == "@sp" else 1
            if reconciled_durations[idx] + step >= min_allowed:
                reconciled_durations[idx] += step
                rem_disc -= step

    reconciled_sum = int(sum(reconciled_durations))
    disc_after = expected_trimmed_mel_len - reconciled_sum

    speech_durs = [
        d for tok, d in zip(extracted_tokens, reconciled_durations) if tok != "@sp"
    ]
    result["expected_trimmed_mel_frames"] = expected_trimmed_mel_len
    result["raw_duration_sum"] = raw_dur_sum
    result["discrepancy_before_reconciliation"] = int(disc_before)
    result["reconciled_duration_sum"] = reconciled_sum
    result["discrepancy_after_reconciliation"] = int(disc_after)
    result["speech_token_count"] = len(speech_durs)
    result["pause_token_count"] = sum(1 for tok in extracted_tokens if tok == "@sp")
    result["min_speech_frame_dur"] = int(min(speech_durs)) if speech_durs else 0
    result["max_speech_frame_dur"] = int(max(speech_durs)) if speech_durs else 0
    result["vocab_oov_tokens"] = sorted(vocab_oov)
    result["extracted_tokens"] = extracted_tokens
    result["reconciled_durations"] = reconciled_durations

    if malformed:
        result["status"] = "failed"
        result["failure_reason"] = f"malformed_intervals: {malformed}"
    elif vocab_oov:
        result["status"] = "failed"
        result["failure_reason"] = f"vocab_oov_tokens: {sorted(vocab_oov)}"
    else:
        result["status"] = "success"

    return result


def _run_mfa_batch(
    corpus_dir: Path,
    out_dir: Path,
    num_jobs: int = 4,
    retry_mode: bool = False,
) -> tuple[int, float, str, str]:
    """Invoke `mfa align` on a staged batch directory."""
    cmd = [
        str(MFA_BIN),
        "align",
        "--clean",
        "--use_mp",
        "-j",
        str(num_jobs),
        "--single_speaker",
        "--no_textgrid_cleanup",
    ]
    if retry_mode:
        cfg_path = corpus_dir.parent / "wide_beam_config.yaml"
        with open(cfg_path, "w", encoding="utf-8") as cf:
            cf.write("beam: 200\nretry_beam: 800\n")
        cmd.extend(["--config_path", str(cfg_path)])
    cmd.extend([
        str(corpus_dir),
        str(DICT_PATH),
        str(MODEL_PATH),
        str(out_dir),
    ])

    env = os.environ.copy()
    env["PATH"] = f"{MFA_BIN.parent}:" + env.get("PATH", "")

    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - t0
    return res.returncode, elapsed, res.stdout, res.stderr


def run_option_1b_alignment(
    batch_size: int = 461,
    num_jobs: int = 4,
) -> dict[str, Any]:
    """Execute full IndicMFA alignment across Option 1B train (2,302) and val (422) splits."""
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    if not MFA_BIN.exists():
        raise FileNotFoundError(f"MFA binary not found at {MFA_BIN}")
    if not DICT_PATH.exists():
        raise FileNotFoundError(f"IndicMFA dictionary not found at {DICT_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"IndicMFA acoustic model not found at {MODEL_PATH}")

    ALIGNMENT_META_DIR.mkdir(parents=True, exist_ok=True)
    (TEXTGRID_ROOT_DIR / "train").mkdir(parents=True, exist_ok=True)
    (TEXTGRID_ROOT_DIR / "val").mkdir(parents=True, exist_ok=True)

    with open(DICT_PATH, encoding="utf-8") as f:
        dict_words = {line.split()[0] for line in f if line.strip()}

    train_entries = _load_split_entries(OPTION_1B_SPLIT_DIR / "train.txt", "train")
    val_entries = _load_split_entries(OPTION_1B_SPLIT_DIR / "val.txt", "val")

    assert len(train_entries) == 2302, f"Expected 2,302 train entries, got {len(train_entries)}"
    assert len(val_entries) == 422, f"Expected 422 val entries, got {len(val_entries)}"

    # Load existing metadata if resuming
    split_records: dict[str, dict[str, dict[str, Any]]] = {"train": {}, "val": {}}
    for sname in ("train", "val"):
        meta_path = ALIGNMENT_META_DIR / f"duration_metadata_{sname}.json"
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    loaded = json.load(f)
                for item in loaded:
                    tg_file = TEXTGRID_ROOT_DIR / sname / f"{item['id']}.TextGrid"
                    if item.get("status") == "success" and tg_file.exists():
                        split_records[sname][item["id"]] = item
            except Exception:
                pass

    progress_path = ALIGNMENT_META_DIR / "alignment_progress.json"
    failures_path = ALIGNMENT_META_DIR / "alignment_failures.json"
    summary_path = ALIGNMENT_META_DIR / "alignment_summary.json"

    batch_logs: list[dict[str, Any]] = []
    if progress_path.exists():
        try:
            with open(progress_path, encoding="utf-8") as f:
                prev_prog = json.load(f)
            batch_logs = prev_prog.get("batch_logs", [])
        except Exception:
            batch_logs = []

    t_pipeline_start = time.perf_counter()

    for sname, entries in (("train", train_entries), ("val", val_entries)):
        total_split = len(entries)
        num_batches = (total_split + batch_size - 1) // batch_size

        for b_idx in range(num_batches):
            b_start = b_idx * batch_size
            b_end = min(total_split, (b_idx + 1) * batch_size)
            batch_entries = entries[b_start:b_end]
            batch_label = f"{sname}_batch_{b_idx + 1:02d}_of_{num_batches:02d}"

            # Filter to entries not yet successfully aligned
            pending_entries = [
                e for e in batch_entries if e["id"] not in split_records[sname]
            ]
            if not pending_entries:
                print(
                    f"[SKIP/RESUME] {batch_label}: all {len(batch_entries)} utterances already aligned & verified."
                )
                continue

            print(
                f"\n[START] {batch_label}: aligning {len(pending_entries)}/{len(batch_entries)} utterances "
                f"(indices {b_start}..{b_end - 1})..."
            )
            t_batch_start = time.perf_counter()

            # Stage WAV and .lab files for pending entries
            batch_corpus_dir = STAGING_ROOT_DIR / batch_label / "corpus"
            batch_out_dir = STAGING_ROOT_DIR / batch_label / "output"
            if (STAGING_ROOT_DIR / batch_label).exists():
                shutil.rmtree(STAGING_ROOT_DIR / batch_label, ignore_errors=True)
            batch_corpus_dir.mkdir(parents=True, exist_ok=True)
            batch_out_dir.mkdir(parents=True, exist_ok=True)

            target_ids = {e["id"] for e in pending_entries}
            audio_bytes_map = _load_parquet_audio_for_ids(target_ids)

            sample_Info_map: dict[str, tuple[int, int]] = {}
            for e in pending_entries:
                uid = e["id"]
                wav_data, orig_sr = sf.read(io.BytesIO(audio_bytes_map[uid]))
                resampled_wav, _ = resample_audio(wav_data, orig_sr, TARGET_SR)
                sample_Info_map[uid] = (len(resampled_wav), int(orig_sr))

                wav_out = batch_corpus_dir / f"{uid}.wav"
                lab_out = batch_corpus_dir / f"{uid}.lab"
                sf.write(str(wav_out), resampled_wav, TARGET_SR)
                with open(lab_out, "w", encoding="utf-8") as lf:
                    lf.write(e["spaced_mfa_text"])

            # Run MFA batch alignment
            retcode, mfa_sec, stdout_s, stderr_s = _run_mfa_batch(
                batch_corpus_dir, batch_out_dir, num_jobs=num_jobs, retry_mode=False
            )

            # Check each utterance and copy TextGrid to permanent directory
            batch_success = 0
            batch_failed_entries: list[dict[str, Any]] = []
            for e in pending_entries:
                uid = e["id"]
                staged_tg = batch_out_dir / f"{uid}.TextGrid"
                perm_tg = TEXTGRID_ROOT_DIR / sname / f"{uid}.TextGrid"
                if staged_tg.exists():
                    shutil.copy2(staged_tg, perm_tg)

                num_samples, orig_sr = sample_Info_map[uid]
                rec = _parse_and_verify_textgrid(
                    perm_tg, e, num_samples, orig_sr, dict_words
                )
                split_records[sname][uid] = rec
                if rec["status"] == "success":
                    batch_success += 1
                else:
                    batch_failed_entries.append(e)

            # Retry any failed utterances in this batch with wider beam
            recovered_in_retry = 0
            if batch_failed_entries:
                print(
                    f"[RETRY] {batch_label}: retrying {len(batch_failed_entries)} failed utterance(s) with wide beam (100/400)..."
                )
                retry_corpus_dir = STAGING_ROOT_DIR / f"{batch_label}_retry" / "corpus"
                retry_out_dir = STAGING_ROOT_DIR / f"{batch_label}_retry" / "output"
                retry_corpus_dir.mkdir(parents=True, exist_ok=True)
                retry_out_dir.mkdir(parents=True, exist_ok=True)
                for e in batch_failed_entries:
                    uid = e["id"]
                    shutil.copy2(batch_corpus_dir / f"{uid}.wav", retry_corpus_dir / f"{uid}.wav")
                    shutil.copy2(batch_corpus_dir / f"{uid}.lab", retry_corpus_dir / f"{uid}.lab")

                _run_mfa_batch(retry_corpus_dir, retry_out_dir, num_jobs=1, retry_mode=True)
                for e in batch_failed_entries:
                    uid = e["id"]
                    staged_tg = retry_out_dir / f"{uid}.TextGrid"
                    perm_tg = TEXTGRID_ROOT_DIR / sname / f"{uid}.TextGrid"
                    if staged_tg.exists():
                        shutil.copy2(staged_tg, perm_tg)
                    num_samples, orig_sr = sample_Info_map[uid]
                    rec = _parse_and_verify_textgrid(
                        perm_tg, e, num_samples, orig_sr, dict_words
                    )
                    rec["recovered_via_wide_beam_retry"] = rec["status"] == "success"
                    split_records[sname][uid] = rec
                    if rec["status"] == "success":
                        batch_success += 1
                        recovered_in_retry += 1
                shutil.rmtree(STAGING_ROOT_DIR / f"{batch_label}_retry", ignore_errors=True)

            # Clean staging directory to keep disk footprint minimal
            shutil.rmtree(STAGING_ROOT_DIR / batch_label, ignore_errors=True)

            batch_total_sec = time.perf_counter() - t_batch_start
            ram_mb = _get_ram_usage_mb()
            batch_log = {
                "batch_label": batch_label,
                "split": sname,
                "batch_index": b_idx + 1,
                "total_batches_in_split": num_batches,
                "attempted_in_batch": len(pending_entries),
                "succeeded_in_batch": batch_success,
                "failed_in_batch": len(pending_entries) - batch_success,
                "recovered_via_retry": recovered_in_retry,
                "mfa_Return_code": retcode,
                "mfa_align_sec": round(mfa_sec, 3),
                "batch_wall_sec": round(batch_total_sec, 3),
                "sec_per_utterance": round(batch_total_sec / max(len(pending_entries), 1), 3),
                "process_ram_mb": round(ram_mb, 2),
            }
            batch_logs.append(batch_log)
            print(
                f"[DONE] {batch_label}: {batch_success}/{len(pending_entries)} succeeded "
                f"({batch_total_sec:.1f}s total, {mfa_sec:.1f}s MFA, {batch_log['sec_per_utterance']:.2f}s/utt, RAM={ram_mb:.1f}MB)"
            )

            # Persist duration metadata and progress after every batch
            ordered_split_list = [
                split_records[sname][e["id"]]
                for e in entries
                if e["id"] in split_records[sname]
            ]
            _atomic_write_json(
                ALIGNMENT_META_DIR / f"duration_metadata_{sname}.json",
                ordered_split_list,
            )

            all_failures = [
                rec
                for sp in ("train", "val")
                for rec in split_records[sp].values()
                if rec.get("status") != "success"
            ]
            _atomic_write_json(failures_path, all_failures)

            _atomic_write_json(
                progress_path,
                {
                    "train_completed": sum(
                        1 for r in split_records["train"].values() if r.get("status") == "success"
                    ),
                    "train_total": len(train_entries),
                    "val_completed": sum(
                        1 for r in split_records["val"].values() if r.get("status") == "success"
                    ),
                    "val_total": len(val_entries),
                    "failure_count": len(all_failures),
                    "batch_logs": batch_logs,
                },
            )

    total_wall_sec = time.perf_counter() - t_pipeline_start

    # Build final summary across train and val
    def _summarize_split(sname: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
        recs = [split_records[sname][e["id"]] for e in entries if e["id"] in split_records[sname]]
        succ = [r for r in recs if r["status"] == "success"]
        fail = [r for r in recs if r["status"] != "success"]
        f_succ = [r for r in succ if r["gender"] == "female"]
        m_succ = [r for r in succ if r["gender"] == "male"]
        disc_before = [abs(r["discrepancy_before_reconciliation"]) for r in succ]
        disc_after = [abs(r["discrepancy_after_reconciliation"]) for r in succ]
        unk_phones = [p for r in succ for p in r["unknown_phones"]]
        dict_oovs = {tok for r in recs for tok in r["dict_oov_tokens"]}
        vocab_oovs = {tok for r in recs for tok in r["vocab_oov_tokens"]}

        return {
            "split": sname,
            "attempted_utterances": len(entries),
            "evaluated_utterances": len(recs),
            "successful_alignments": len(succ),
            "failed_alignments": len(fail),
            "success_rate_pct": round((len(succ) / max(len(entries), 1)) * 100.0, 4),
            "female_attempted": sum(1 for e in entries if e["gender"] == "female"),
            "female_succeeded": len(f_succ),
            "male_attempted": sum(1 for e in entries if e["gender"] == "male"),
            "male_succeeded": len(m_succ),
            "total_audio_hours": round(sum(r["actual_wav_duration_sec"] for r in recs) / 3600.0, 6),
            "total_speech_hours": round(sum(r["total_speech_sec"] for r in succ) / 3600.0, 6),
            "total_silence_hours": round(sum(r["total_silence_sec"] for r in succ) / 3600.0, 6),
            "dict_oov_token_count": len(dict_oovs),
            "dict_oov_tokens": sorted(dict_oovs),
            "vocab_oov_token_count": len(vocab_oovs),
            "vocab_oov_tokens": sorted(vocab_oovs),
            "unknown_phone_interval_count": len(unk_phones),
            "max_abs_frame_discrepancy_before_reconciliation": int(max(disc_before)) if disc_before else 0,
            "mean_abs_frame_discrepancy_before_reconciliation": round(float(np.mean(disc_before)), 4) if disc_before else 0.0,
            "max_abs_frame_discrepancy_after_reconciliation": int(max(disc_after)) if disc_after else 0,
            "zero_frame_discrepancy_after_reconciliation_count": sum(1 for d in disc_after if d == 0),
            "min_speech_token_frame_duration": int(min(r["min_speech_frame_dur"] for r in succ)) if succ else 0,
            "max_speech_token_frame_duration": int(max(r["max_speech_frame_dur"] for r in succ)) if succ else 0,
        }

    train_sum = _summarize_split("train", train_entries)
    val_sum = _summarize_split("val", val_entries)
    all_failures = [
        rec
        for sp in ("train", "val")
        for rec in split_records[sp].values()
        if rec.get("status") != "success"
    ]

    summary = {
        "dataset": "SPRINGLab/IndicTTS_Telugu (Option 1B)",
        "acoustic_model": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "g2g_dictionary": str(DICT_PATH.relative_to(PROJECT_ROOT)),
        "mfa_binary": str(MFA_BIN),
        "target_sample_rate_hz": TARGET_SR,
        "hop_length_samples": HOP_LENGTH,
        "test_split_aligned": False,
        "test_split_isolation_verified": not (TEXTGRID_ROOT_DIR / "test").exists(),
        "total_attempted_utterances": train_sum["attempted_utterances"] + val_sum["attempted_utterances"],
        "total_successful_alignments": train_sum["successful_alignments"] + val_sum["successful_alignments"],
        "total_failed_alignments": train_sum["failed_alignments"] + val_sum["failed_alignments"],
        "overall_success_rate_pct": round(
            (
                (train_sum["successful_alignments"] + val_sum["successful_alignments"])
                / float(train_sum["attempted_utterances"] + val_sum["attempted_utterances"])
            )
            * 100.0,
            4,
        ),
        "total_wall_clock_sec": round(total_wall_sec, 3),
        "cumulative_mfa_align_sec": round(sum(b["mfa_align_sec"] for b in batch_logs), 3),
        "peak_process_ram_mb": round(max((b["process_ram_mb"] for b in batch_logs), default=_get_ram_usage_mb()), 2),
        "splits": {
            "train": train_sum,
            "val": val_sum,
        },
        "failures": all_failures,
        "batch_logs": batch_logs,
    }

    _atomic_write_json(summary_path, summary)
    print("\n=== OPTION 1B FULL MFA ALIGNMENT SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "batch_logs"}, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full IndicMFA alignment for Telugu Option 1B (train + val).")
    parser.add_argument("--batch_size", type=int, default=461, help="Utterances per MFA batch.")
    parser.add_argument("--num_jobs", type=int, default=4, help="Number of parallel MFA jobs (-j).")
    args = parser.parse_args()
    run_option_1b_alignment(batch_size=args.batch_size, num_jobs=args.num_jobs)
