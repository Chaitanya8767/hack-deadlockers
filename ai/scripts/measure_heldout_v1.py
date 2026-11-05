import json
import sys
from pathlib import Path

# Add ai directory to sys.path
ai_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ai_dir))

from conversation_manager import Turn, TacticEvent
from regex_classifier import classify_regex
from risk_engine import RiskEngine

sys.stdout.reconfigure(encoding="utf-8")

def main():
    heldout_path = ai_dir / "tests" / "fixtures" / "heldout.json"
    out_path = ai_dir / "scripts" / "out" / "heldout_v1_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(heldout_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    
    print("=" * 110)
    print("HELDOUT V1 (BEFORE SNAPSHOT): ONE-TURN CALL RISK ENGINE MEASUREMENT")
    print("=" * 110)
    print(f"{'ID':<11} | {'Type':<6} | {'Lang':<8} | {'Risk':<5} | {'Level':<8} | {'Regex Tactics':<35} | {'Expected Tactics'}")
    print("-" * 110)

    benign_total = 0
    benign_high_crit = 0
    benign_low_med = 0

    scam_total = 0
    scam_high_crit = 0
    scam_low_med = 0

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
        
        # Fresh risk engine for isolated single turn
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
            if eval_result.level in ("High", "Critical"):
                scam_high_crit += 1
            else:
                scam_low_med += 1
        else:
            benign_total += 1
            if eval_result.level in ("High", "Critical"):
                benign_high_crit += 1
            else:
                benign_low_med += 1

        print(f"{item['id']:<11} | {item_type:<6} | {item['lang']:<8} | {eval_result.score:<5.1f} | {eval_result.level:<8} | {str(detected_tactics):<35} | {str(item['expected_tactics'])}")

    print("=" * 110)
    print(f"BENIGN CASES (Total: {benign_total}):")
    print(f"  - Low/Medium (Non-alarm):   {benign_low_med}/{benign_total} ({benign_low_med/benign_total*100:.1f}%)")
    print(f"  - High/Critical (False alarm): {benign_high_crit}/{benign_total} ({benign_high_crit/benign_total*100:.1f}%)")
    print("-" * 110)
    print(f"SCAM CASES (Total: {scam_total}):")
    print(f"  - High/Critical (Alarmed):   {scam_high_crit}/{scam_total} ({scam_high_crit/scam_total*100:.1f}%)")
    print(f"  - Low/Medium:               {scam_low_med}/{scam_total} ({scam_low_med/scam_total*100:.1f}%)")
    print("=" * 110)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved raw V1 snapshot to: {out_path}")

if __name__ == "__main__":
    main()
