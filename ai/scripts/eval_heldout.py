"""
Evaluation of the 20 held-out test utterances (heldout.json) on the unedited regex classifier.
Reports raw hit rate on scams, false positives on benign cases, and per-utterance results as-is.
"""

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from conversation_manager import Turn
from regex_classifier import classify_regex

HELDOUT_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "heldout.json")

def evaluate_heldout():
    with open(HELDOUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    scam_total = 0
    scam_hits = 0
    tactic_expected_total = 0
    tactic_matches = 0

    benign_total = 0
    benign_clean = 0
    benign_false_positives = 0

    print("=" * 110)
    print("EVALUATION ON 20 HELD-OUT UNSEEN UTTERANCES (ZERO RETUNING)")
    print("=" * 110)
    print(f"{'ID':<11} | {'Type':<6} | {'Expected Tactics':<32} | {'Regex Detected':<32} | {'Result':<12}")
    print("-" * 110)

    for item in data:
        turn = Turn(id=item["id"], text=item["text"], lang=item["lang"], t0=0.0, t1=3.0)
        events = classify_regex(turn)
        detected = sorted(list({e.tactic for e in events}))
        expected = sorted(list(set(item["expected_tactics"])))

        is_scam = item["is_scam"]
        if is_scam:
            scam_total += 1
            tactic_expected_total += len(expected)
            # Check tactic overlap
            matched = set(detected).intersection(set(expected))
            tactic_matches += len(matched)
            if len(matched) > 0:
                scam_hits += 1
                res_str = f"HIT ({len(matched)}/{len(expected)})"
            else:
                res_str = "MISSED"
        else:
            benign_total += 1
            if len(detected) == 0:
                benign_clean += 1
                res_str = "CLEAN (0 FP)"
            else:
                benign_false_positives += 1
                res_str = f"FP ({len(detected)})"

        exp_str = ", ".join(expected) if expected else "(none)"
        det_str = ", ".join(detected) if detected else "(none)"
        print(f"{item['id']:<11} | {'SCAM' if is_scam else 'BENIGN':<6} | {exp_str[:32]:<32} | {det_str[:32]:<32} | {res_str:<12}")

    print("=" * 110)
    print(f"Scam Utterance Hit Rate:       {scam_hits}/{scam_total} ({scam_hits/scam_total*100:.1f}%)")
    print(f"Per-Tactic Recall on Scams:    {tactic_matches}/{tactic_expected_total} ({tactic_matches/tactic_expected_total*100:.1f}%)")
    print(f"Benign False Positive Rate:    {benign_false_positives}/{benign_total} ({benign_false_positives/benign_total*100:.1f}%)")
    print(f"Benign Clean Rate (0 FP):      {benign_clean}/{benign_total} ({benign_clean/benign_total*100:.1f}%)")
    print("=" * 110)

if __name__ == "__main__":
    evaluate_heldout()

