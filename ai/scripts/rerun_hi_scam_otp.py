import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Reconfigure stdout for utf-8
sys.stdout.reconfigure(encoding="utf-8")

ai_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ai_dir))

load_dotenv(ai_dir / ".env")

import httpx
from conversation_manager import Turn, TacticEvent
from regex_classifier import classify_regex
from risk_engine import RiskEngine
from tactic_classifier import ClassifierOutput, SYSTEM_PROMPT, VALID_TACTICS

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "qwen/qwen3.8-27b"
ORIGINAL_RESULTS_PATH = ai_dir / "scripts" / "out" / "groq_results.json"
RERUN_RESULTS_PATH = ai_dir / "scripts" / "out" / "groq_results_rerun.json"
CONV_FILE = ai_dir / "tests" / "conversations" / "hi.json"

def call_groq_api(user_prompt: str, max_retries: int = 5):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    raw_text = resp.json()["choices"][0]["message"]["content"]
                    parsed = json.loads(raw_text)
                    return {"status": "SUCCESS", "raw_response": raw_text, "parsed": parsed}
                elif resp.status_code == 429:
                    wait_s = 5.0 * (attempt + 1)
                    print(f"  [Rate-limit 429] Waiting {wait_s}s before retry {attempt + 1}...", flush=True)
                    time.sleep(wait_s)
                    continue
                else:
                    return {"status": "ERROR", "error": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            wait_s = 4.0 * (attempt + 1)
            print(f"  [Exception {e}] Waiting {wait_s}s before retry {attempt + 1}...", flush=True)
            time.sleep(wait_s)

    return {"status": "ERROR", "error": "Max retries exceeded"}

def main():
    if not ORIGINAL_RESULTS_PATH.exists():
        print(f"Original results not found at: {ORIGINAL_RESULTS_PATH}")
        sys.exit(1)

    with open(ORIGINAL_RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(CONV_FILE, "r", encoding="utf-8") as f:
        hi_convs = json.load(f)

    scenario_name = "scam_bank_kyc_otp"
    turns = hi_convs[scenario_name]
    scenario_key = f"hi:{scenario_name}"

    print(f"Rerunning live Groq for scenario [{scenario_key}] ({len(turns)} turns)...")

    regex_engine = RiskEngine()
    groq_engine = RiskEngine()
    regex_events_accum = []
    groq_events_accum = []
    history_lines = []

    regex_first_high = "Never"
    groq_first_high = "Never"
    turn_agreements = 0

    rerun_turns = []

    for turn_idx, t in enumerate(turns):
        turn_text = t["text"]
        context_str = "\n".join(history_lines) if history_lines else "No previous utterances."
        user_prompt = f"""CONVERSATION CONTEXT:\n{context_str}\n\nLATEST CALLER UTTERANCE:\n[HI] "{turn_text}"\n\nAnalyze the latest utterance for the 10 scam tactics. Respond strictly with JSON:"""

        # 1. Regex
        turn_obj = Turn(
            id=f"{scenario_name}-{turn_idx}",
            speaker="caller",
            text=turn_text,
            lang="hi",
            t0=float(turn_idx * 5.0),
            t1=float((turn_idx + 1) * 5.0)
        )
        reg_events = classify_regex(turn_obj, context_str=context_str, has_prior_tactics=bool(regex_events_accum))
        regex_events_accum.extend(reg_events)
        reg_risk = regex_engine.calculate_risk(scenario_name, regex_events_accum)
        reg_tactics = set(e.tactic for e in reg_events)

        if reg_risk.score > 60.0 and regex_first_high == "Never":
            regex_first_high = f"Turn {turn_idx}"

        # 2. Live Groq call
        groq_call = call_groq_api(user_prompt)
        groq_events = []
        groq_tactics = set()

        if groq_call["status"] == "SUCCESS":
            parsed = ClassifierOutput.model_validate(groq_call["parsed"])
            for item in parsed.tactics:
                if item.tactic in VALID_TACTICS and item.confidence >= 0.40:
                    groq_events.append(TacticEvent(
                        tactic=item.tactic,
                        confidence=round(item.confidence, 2),
                        evidence=item.evidence or turn_text,
                        explanation=item.explanation or f"Detected {item.tactic}",
                        t=float((turn_idx + 1) * 5.0),
                        turn_id=f"{scenario_name}-{turn_idx}"
                    ))
                    groq_tactics.add(item.tactic)

            groq_events_accum.extend(groq_events)
            groq_risk = groq_engine.calculate_risk(scenario_name, groq_events_accum)
            groq_score = groq_risk.score
            groq_level = groq_risk.level

            if groq_score > 60.0 and groq_first_high == "Never":
                groq_first_high = f"Turn {turn_idx}"
        else:
            groq_score = "ERROR"
            groq_level = "ERROR"

        is_turn_agree = (reg_tactics == groq_tactics)
        if is_turn_agree:
            turn_agreements += 1

        print(f"  Turn {turn_idx}: Regex={sorted(list(reg_tactics))} | Groq={sorted(list(groq_tactics))} | Match={is_turn_agree}")

        rerun_turns.append({
            "turn_index": turn_idx,
            "text": turn_text,
            "regex_tactics": sorted(list(reg_tactics)),
            "regex_turn_score": reg_risk.score,
            "groq_status": groq_call["status"],
            "groq_raw": groq_call.get("raw_response", groq_call.get("error")),
            "groq_tactics": sorted(list(groq_tactics)),
            "groq_turn_score": groq_score,
            "turn_agreement": is_turn_agree
        })

        history_lines.append(f"Caller: {turn_text}")
        time.sleep(1.0)

    # Clone original data and replace scenario entry
    rerun_data = dict(data)
    rerun_data[scenario_key] = {
        "language": "hi",
        "scenario": scenario_name,
        "turns": rerun_turns
    }

    with open(RERUN_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(rerun_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved updated rerun audit dataset to: {RERUN_RESULTS_PATH}")
    print("\nUpdated row:")
    agreement_str = f"{(turn_agreements / len(turns)) * 100:.1f}% ({turn_agreements}/{len(turns)})"
    print(f"hi | scam_bank_kyc_otp | Reg: {reg_risk.score:.1f} ({reg_risk.level}) | Groq: {groq_score:.1f} ({groq_level}) | Reg 1st High: {regex_first_high} | Groq 1st High: {groq_first_high} | Tactic Agree: {agreement_str}")

if __name__ == "__main__":
    main()

