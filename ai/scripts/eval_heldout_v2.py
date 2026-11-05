import json
import sys
from pathlib import Path

# Add ai directory to sys.path
ai_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ai_dir))

from conversation_manager import Turn
from regex_classifier import classify_regex
from risk_engine import RiskEngine

sys.stdout.reconfigure(encoding="utf-8")

def main():
    heldout_path = ai_dir / "tests" / "fixtures" / "heldout_v2.json"
    out_path = ai_dir / "scripts" / "out" / "heldout_v2_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(heldout_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    print("=" * 115)
    print("HELDOUT V2 (AFTER FIXES): ONE-TURN CALL RISK ENGINE MEASUREMENT (25 FRESH UTTERANCES)")
    print("=" * 115)
    print(f"{'ID':<12} | {'Type':<6} | {'Lang':<8} | {'Risk':<5} | {'Level':<8} | {'Regex Detected':<35} | {'Expected'}")
    print("-" * 115)

    benign_total = 0
    benign_high_crit = 0
    benign_low_med = 0
    benign_zero_tactic = 0

    scam_total = 0
    scam_high_crit = 0
    scam_low_med = 0
    scam_hits = 0
    total_expected_tactics = 0
    matched_expected_tactics = 0

    for item in data:
        turn = Turn(
            id=item["id"],
            speaker="caller",
            text=item["text"],
            lang=item.get("lang", "en"),
            t0=0.0,
            t1=2.0
        )
        events = classify_regex(turn, context_str="", has_prior_tactics=False)
        detected_tactics = [e.tactic for e in events]
        detected_set = set(detected_tactics)
        expected_set = set(item["expected_tactics"])

        engine = RiskEngine()
        eval_result = engine.calculate_risk(f"call_{item['id']}", events, current_time=turn.t1)

        is_scam = item["is_scam"]
        item_type = "SCAM" if is_scam else "BENIGN"

        record = {
            "id": item["id"],
            "text": item["text"],
            "lang": item["lang"],
            "is_scam": is_scam,
            "expected_tactics": item["expected_tactics"],
            "detected_tactics": detected_tactics,
            "risk_score": eval_result.score,
            "risk_level": eval_result.level,
            "risk_patterns": eval_result.patterns
        }
        results.append(record)

        if is_scam:
            scam_total += 1
            overlap = detected_set.intersection(expected_set)
            if len(overlap) > 0:
                scam_hits += 1
            total_expected_tactics += len(expected_set)
            matched_expected_tactics += len(overlap)

            if eval_result.level in ("High", "Critical"):
                scam_high_crit += 1
            else:
                scam_low_med += 1
        else:
            benign_total += 1
            if len(detected_set) == 0:
                benign_zero_tactic += 1
            if eval_result.level in ("High", "Critical"):
                benign_high_crit += 1
            else:
                benign_low_med += 1

        print(f"{item['id']:<12} | {item_type:<6} | {item['lang']:<8} | {eval_result.score:<5.1f} | {eval_result.level:<8} | {str(detected_tactics):<35} | {str(item['expected_tactics'])}")

    print("=" * 115)
    print(f"BENIGN CASES (Total: {benign_total}):")
    print(f"  - Zero Tactics Detected (Clean): {benign_zero_tactic}/{benign_total} ({benign_zero_tactic/benign_total*100:.1f}%)")
    print(f"  - Low/Medium Level:             {benign_low_med}/{benign_total} ({benign_low_med/benign_total*100:.1f}%)")
    print(f"  - High/Critical (False Alarm):  {benign_high_crit}/{benign_total} ({benign_high_crit/benign_total*100:.1f}%)")
    print("-" * 115)
    print(f"SCAM CASES (Total: {scam_total}):")
    print(f"  - Scam Utterance Hit Rate:      {scam_hits}/{scam_total} ({scam_hits/scam_total*100:.1f}%)")
    print(f"  - Per-Tactic Recall on Scams:   {matched_expected_tactics}/{total_expected_tactics} ({matched_expected_tactics/total_expected_tactics*100:.1f}%)")
    print(f"  - High/Critical Risk Level:     {scam_high_crit}/{scam_total} ({scam_high_crit/scam_total*100:.1f}%)")
    print(f"  - Low/Medium Level:             {scam_low_med}/{scam_total} ({scam_low_med/scam_total*100:.1f}%)")
    print("=" * 115)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved raw V2 results to: {out_path}")

if __name__ == "__main__":
    main()

