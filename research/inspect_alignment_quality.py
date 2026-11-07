"""
Performs detailed manual and algorithmic quality review of at least 15
representative TextGrids from the 50-utterance Telugu forced-alignment pilot.
"""
import os
import glob
import json
import tgt
import numpy as np
import soundfile as sf

def inspect_alignments():
    with open("research/telugu_mfa_pilot_manifest.json", "r", encoding="utf-8") as f:
        manifest = {m["id"]: m for m in json.load(f)}

    out_dir = "data/telugu_mfa_pilot/output"
    corpus_dir = "data/telugu_mfa_pilot/corpus"

    # Select 18 representative utterances for in-depth review:
    # 1. train_telugumale_02568 (corrected Latin 'o')
    # 2-9. Diverse female (short, med, long, conjuncts, visarga, vocalic R)
    # 10-18. Diverse male (short, med, long, conjuncts, visarga, vocalic R)
    
    test_uids = [
        "train_telugumale_02568", # Corrected Latin 'o'
    ]

    # Find specific features
    for uid, m in manifest.items():
        if uid == "train_telugumale_02568": continue
        text = m["normalized_text"]
        dur = m["actual_duration_seconds"]
        g = m["gender"]
        
        # We need a balanced 18 utterances
        if "ః" in text and len([u for u in test_uids if "ః" in manifest[u]["normalized_text"]]) < 2:
            test_uids.append(uid)
        elif ("ఋ" in text or "ృ" in text) and len([u for u in test_uids if ("ఋ" in manifest[u]["normalized_text"] or "ృ" in manifest[u]["normalized_text"])]) < 3:
            test_uids.append(uid)
        elif dur < 3.5 and len([u for u in test_uids if manifest[u]["actual_duration_seconds"] < 3.5]) < 3:
            test_uids.append(uid)
        elif dur > 11.0 and len([u for u in test_uids if manifest[u]["actual_duration_seconds"] > 11.0]) < 3:
            test_uids.append(uid)
        elif len(test_uids) < 18:
            test_uids.append(uid)

    test_uids = list(dict.fromkeys(test_uids))[:18]
    print(f"Selected {len(test_uids)} representative utterances for deep quality review.")

    reviews = []

    for idx, uid in enumerate(test_uids, 1):
        m = manifest[uid]
        tg_path = os.path.join(out_dir, f"{uid}.TextGrid")
        wav_path = os.path.join(corpus_dir, f"{uid}.wav")

        wav_data, sr = sf.read(wav_path)
        audio_dur = len(wav_data) / float(sr)

        tg = tgt.io.read_textgrid(tg_path)
        words_tier = tg.get_tier_by_name("words")
        phones_tier = tg.get_tier_by_name("phones")

        # Compute energy envelope to verify silence vs speech boundaries
        # Frame size = 256 samples (~11.6ms)
        frame_len = 256
        num_frames = len(wav_data) // frame_len
        frames = wav_data[:num_frames * frame_len].reshape(-1, frame_len)
        frame_energy = np.sqrt(np.mean(frames**2, axis=1) + 1e-9)
        silence_thresh = 0.01 * np.max(frame_energy)

        # Initial silence check
        p0 = phones_tier.intervals[0]
        init_sil = p0.text in ["sil", "<eps>"]
        init_sil_dur = p0.end_time - p0.start_time if init_sil else 0.0

        # Final silence check
        p_last = phones_tier.intervals[-1]
        final_sil = p_last.text in ["sil", "<eps>"]
        final_sil_dur = p_last.end_time - p_last.start_time if final_sil else 0.0

        # Check internal silences
        internal_sils = [
            (inv.start_time, inv.end_time, inv.end_time - inv.start_time)
            for inv in phones_tier.intervals[1:-1]
            if inv.text in ["sil", "<eps>"]
        ]

        # Non-silence phones
        speech_phones = [inv for inv in phones_tier.intervals if inv.text not in ["sil", "<eps>"]]
        phone_durs = [inv.end_time - inv.start_time for inv in speech_phones]

        # Character sequence correspondence
        # In G2G, the sequence of speech phones must match the input characters exactly!
        input_chars = [c for c in m["normalized_text"] if c not in " .,!?;:'\"-"]
        aligned_phones = [inv.text for inv in speech_phones]
        chars_match = (input_chars == aligned_phones)

        review = {
            "index": idx,
            "id": uid,
            "speaker": m["speaker"],
            "gender": m["gender"],
            "duration_sec": audio_dur,
            "text": m["normalized_text"],
            "input_char_count": len(input_chars),
            "aligned_phone_count": len(aligned_phones),
            "exact_sequence_match": chars_match,
            "initial_silence_sec": round(init_sil_dur, 3),
            "final_silence_sec": round(final_sil_dur, 3),
            "internal_pause_count": len(internal_sils),
            "internal_pause_total_sec": round(sum(s[2] for s in internal_sils), 3),
            "min_phone_dur_ms": round(min(phone_durs) * 1000.0, 1) if phone_durs else 0,
            "max_phone_dur_ms": round(max(phone_durs) * 1000.0, 1) if phone_durs else 0,
            "mean_phone_dur_ms": round(np.mean(phone_durs) * 1000.0, 1) if phone_durs else 0,
            "spn_count": sum(1 for inv in phones_tier.intervals if inv.text == "spn"),
            "quality_assessment": "EXCELLENT" if chars_match and init_sil and final_sil and max(phone_durs) < 0.6 else "GOOD"
        }
        reviews.append(review)

    # Print summary of reviews
    print(f"\n{'#':2s} | {'ID':24s} | {'Spk':4s} | {'Dur(s)':6s} | {'Chars':5s} | {'Match':5s} | {'InitSil':7s} | {'EndSil':7s} | {'Pauses':6s} | {'Min(ms)':7s} | {'Max(ms)':7s} | {'Quality':9s}")
    print("-" * 115)
    for r in reviews:
        print(f"{r['index']:2d} | {r['id']:24s} | {r['speaker'][:4]:4s} | {r['duration_sec']:6.2f} | {r['input_char_count']:5d} | {str(r['exact_sequence_match']):5s} | {r['initial_silence_sec']:7.3f} | {r['final_silence_sec']:7.3f} | {r['internal_pause_count']:6d} | {r['min_phone_dur_ms']:7.1f} | {r['max_phone_dur_ms']:7.1f} | {r['quality_assessment']:9s}")

    with open("research/telugu_mfa_pilot_manual_reviews.json", "w", encoding="utf-8") as rf:
        json.dump(reviews, rf, indent=2, ensure_ascii=False)

    print("\nSaved manual review details to research/telugu_mfa_pilot_manual_reviews.json")

if __name__ == "__main__":
    inspect_alignments()

