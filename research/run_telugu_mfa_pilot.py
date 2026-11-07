"""
Executes IndicMFA alignment on the 50 representative pilot utterances
and audits the generated TextGrid outputs.
"""
import os
import sys
import time
import glob
import json
import shutil
import subprocess
import tgt

def run_pilot():
    corpus_dir = "data/telugu_mfa_pilot/corpus"
    out_dir = "data/telugu_mfa_pilot/output"
    os.makedirs(out_dir, exist_ok=True)

    # Clean previous output
    for f in glob.glob(os.path.join(out_dir, "*")):
        if os.path.isfile(f):
            os.remove(f)

    # Load pilot manifest
    with open("research/telugu_mfa_pilot_manifest.json", "r", encoding="utf-8") as f:
        pilot_manifest = {m["id"]: m for m in json.load(f)}

    print(f"Loaded manifest for {len(pilot_manifest)} utterances.")

    mfa_bin = "/opt/anaconda3/envs/mfa_env/bin/mfa"
    dict_path = "research/Telugu_Dictionary_g2g.txt"
    model_path = "data/telugu_mfa_model/Telugu_Acoustic_Model.zip"

    cmd = [
        mfa_bin,
        "align",
        "--clean",
        "--use_mp",
        "-j", "4",
        "--single_speaker",
        "--no_textgrid_cleanup",
        corpus_dir,
        dict_path,
        model_path,
        out_dir
    ]

    print("\nRunning MFA command:")
    print(" ".join(cmd))

    env = os.environ.copy()
    env["PATH"] = "/opt/anaconda3/envs/mfa_env/bin:" + env.get("PATH", "")

    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0

    print(f"\nMFA exited with code {res.returncode} in {elapsed:.2f} seconds ({elapsed/len(pilot_manifest):.2f} s/utt)")
    print("STDOUT tail:\n", res.stdout[-1500:])
    if res.stderr:
        print("STDERR tail:\n", res.stderr[-1500:])

    # Audit outputs
    results = {
        "execution_time_seconds": elapsed,
        "utterance_count": len(pilot_manifest),
        "success_count": 0,
        "failure_count": 0,
        "utterance_results": []
    }

    for uid, m in pilot_manifest.items():
        tg_path = os.path.join(out_dir, f"{uid}.TextGrid")
        utt_res = {
            "id": uid,
            "speaker": m["speaker"],
            "gender": m["gender"],
            "actual_audio_duration": m["actual_duration_seconds"],
            "textgrid_exists": os.path.exists(tg_path),
            "status": "failed",
            "phones_tier_intervals": 0,
            "words_tier_intervals": 0,
            "silence_intervals_count": 0,
            "speech_intervals_count": 0,
            "total_silence_duration": 0.0,
            "total_speech_duration": 0.0,
            "max_interval_duration": 0.0,
            "min_interval_duration": 0.0,
            "malformed_intervals": [],
            "unknown_phones": [],
            "initial_silence_duration": 0.0,
            "final_silence_duration": 0.0
        }

        if not os.path.exists(tg_path):
            results["failure_count"] += 1
            results["utterance_results"].append(utt_res)
            continue

        try:
            tg = tgt.io.read_textgrid(tg_path)
            words_tier = tg.get_tier_by_name("words")
            phones_tier = tg.get_tier_by_name("phones")

            utt_res["words_tier_intervals"] = len(words_tier.intervals)
            utt_res["phones_tier_intervals"] = len(phones_tier.intervals)

            # Check interval monotonicity and bounds
            min_dur = 999999.0
            max_dur = 0.0
            speech_dur = 0.0
            sil_dur = 0.0
            sil_count = 0
            speech_count = 0
            malformed = []
            unknown_phones = []

            for i, inv in enumerate(phones_tier.intervals):
                dur = inv.end_time - inv.start_time
                if dur <= 0:
                    malformed.append(f"Interval {i} negative/zero dur: [{inv.start_time}, {inv.end_time}]")
                if i > 0 and inv.start_time < phones_tier.intervals[i-1].end_time - 1e-4:
                    malformed.append(f"Interval {i} overlap with previous: {inv.start_time} < {phones_tier.intervals[i-1].end_time}")

                if inv.text in ["sil", "<eps>", ""]:
                    sil_count += 1
                    sil_dur += dur
                else:
                    speech_count += 1
                    speech_dur += dur
                    if dur < min_dur: min_dur = dur
                    if dur > max_dur: max_dur = dur

                if inv.text == "spn":
                    unknown_phones.append((i, inv.start_time, inv.end_time))

            # Initial and final silence
            if phones_tier.intervals and phones_tier.intervals[0].text in ["sil", "<eps>"]:
                utt_res["initial_silence_duration"] = phones_tier.intervals[0].end_time - phones_tier.intervals[0].start_time
            if phones_tier.intervals and phones_tier.intervals[-1].text in ["sil", "<eps>"]:
                utt_res["final_silence_duration"] = phones_tier.intervals[-1].end_time - phones_tier.intervals[-1].start_time

            utt_res["silence_intervals_count"] = sil_count
            utt_res["speech_intervals_count"] = speech_count
            utt_res["total_silence_duration"] = sil_dur
            utt_res["total_speech_duration"] = speech_dur
            utt_res["min_interval_duration"] = min_dur if speech_count > 0 else 0.0
            utt_res["max_interval_duration"] = max_dur
            utt_res["malformed_intervals"] = malformed
            utt_res["unknown_phones"] = unknown_phones

            if not malformed:
                utt_res["status"] = "success"
                results["success_count"] += 1
            else:
                utt_res["status"] = "malformed"
                results["failure_count"] += 1

        except Exception as e:
            utt_res["status"] = f"error: {str(e)}"
            results["failure_count"] += 1

        results["utterance_results"].append(utt_res)

    print(f"\nAudit complete: {results['success_count']} successes, {results['failure_count']} failures out of {len(pilot_manifest)} utterances.")

    # Save results
    with open("research/telugu_mfa_pilot_results.json", "w", encoding="utf-8") as rf:
        json.dump(results, rf, indent=2, ensure_ascii=False)

    print("Detailed results saved to research/telugu_mfa_pilot_results.json")

if __name__ == "__main__":
    run_pilot()

