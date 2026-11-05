"""
Full live Groq evaluation across all 12 conversations in ai/tests/conversations/.
Executes real Groq API calls, runs both Regex and Groq through ScamChain's RiskEngine,
records raw responses into ai/scripts/out/groq_results.json, and prints side-by-side tables.
"""

import json
import os
import sys
import time
import httpx

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from conversation_manager import Turn, TacticEvent
from risk_engine import RiskEngine
from regex_classifier import classify_regex
from tactic_classifier import SYSTEM_PROMPT, ClassifierOutput, VALID_TACTICS

CONV_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "conversations"))
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "out"))
os.makedirs(OUT_DIR, exist_ok=True)
RAW_OUTPUT_PATH = os.path.join(OUT_DIR, "groq_results.json")

def call_groq_api(user_content: str, retries: int = 3) -> dict:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.llm.groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen/qwen3.8-27b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 1024
    }

    for attempt in range(retries):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=20.0)
            if resp.status_code == 200:
                raw_text = resp.json()["choices"][0]["message"]["content"]
                return {"status": "SUCCESS", "raw_response": raw_text, "parsed": json.loads(raw_text)}
            elif resp.status_code == 429:
                print(f"[Rate-limit 429] Waiting 4s before retry {attempt+1}...", flush=True)
                time.sleep(4.0)
                continue
            else:
                return {"status": "ERROR", "error": f"HTTP {resp.status_code}: {resp.text[:300]}"}
        except Exception as e:
            if attempt == retries - 1:
                return {"status": "ERROR", "error": str(e)}
            time.sleep(2.0)

    return {"status": "ERROR", "error": "Rate limit / retry limit exceeded"}

def run_all_evaluations():
    languages = ["en", "hi", "hinglish", "mr"]
    all_raw_results = {}
    summary_rows = []

    print("=" * 80, flush=True)
    print("STARTING LIVE EVALUATION ACROSS ALL 12 CONVERSATIONS", flush=True)
    print(f"API Target: https://api.groq.com/openai/v1/chat/completions (Model: qwen/qwen3.8-27b)", flush=True)
    print(f"Audit log output destination: {RAW_OUTPUT_PATH}", flush=True)
    print("=" * 80, flush=True)

    for lang in languages:
        conv_file = os.path.join(CONV_DIR, f"{lang}.json")
        with open(conv_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for scenario_name, turns in data.items():
            if scenario_name.startswith("_"):
                continue

            print(f"\nEvaluating [{lang.upper()}] {scenario_name} ({len(turns)} turns)...", flush=True)

            scenario_key = f"{lang}:{scenario_name}"
            all_raw_results[scenario_key] = {
                "language": lang,
                "scenario": scenario_name,
                "turns": []
            }

            regex_engine = RiskEngine()
            groq_engine = RiskEngine()

            regex_events_accum = []
            groq_events_accum = []
            history_lines = []

            regex_first_high = "Never"
            groq_first_high = "Never"

            turn_agreements = 0

            for turn_idx, t in enumerate(turns):
                turn_text = t["text"]
                context_str = "\n".join(history_lines) if history_lines else "No previous utterances."
                user_prompt = f"""CONVERSATION CONTEXT:\n{context_str}\n\nLATEST CALLER UTTERANCE:\n[{lang.upper()}] "{turn_text}"\n\nAnalyze the latest utterance for the 10 scam tactics. Respond strictly with JSON:"""

                # 1. Regex Classification
                turn_obj = Turn(
                    id=f"{scenario_name}-{turn_idx}",
                    text=turn_text,
                    lang=lang,
                    t0=float(turn_idx * 5.0),
                    t1=float((turn_idx + 1) * 5.0)
                )
                reg_events = classify_regex(turn_obj, context_str=context_str, has_prior_tactics=bool(regex_events_accum))
                regex_events_accum.extend(reg_events)
                reg_risk = regex_engine.calculate_risk(scenario_name, regex_events_accum)
                reg_tactics = set(e.tactic for e in reg_events)

                if reg_risk.score > 60.0 and regex_first_high == "Never":
                    regex_first_high = f"Turn {turn_idx}"

                # 2. Live Groq Classification
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

                all_raw_results[scenario_key]["turns"].append({
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

                history_lines.append(f"[{lang.upper()}] Caller: \"{turn_text}\"")
                print(f"  Turn {turn_idx}: Regex={sorted(list(reg_tactics))} | Groq={sorted(list(groq_tactics))} | Match={is_turn_agree}", flush=True)

                time.sleep(1.0)

            agreement_pct = (turn_agreements / len(turns)) * 100.0
            summary_rows.append({
                "lang": lang,
                "scenario": scenario_name,
                "turns_count": len(turns),
                "regex_final_score": reg_risk.score,
                "regex_final_level": reg_risk.level,
                "regex_first_high": regex_first_high,
                "groq_final_score": groq_score,
                "groq_final_level": groq_level,
                "groq_first_high": groq_first_high,
                "turn_agreement": f"{agreement_pct:.1f}% ({turn_agreements}/{len(turns)})"
            })

            # Save incrementally after each scenario
            with open(RAW_OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(all_raw_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 115, flush=True)
    print("HONEST HEAD-TO-HEAD COMPARISON TABLE: REGEX vs LIVE GROQ (Audit Log in groq_results.json)", flush=True)
    print("=" * 115, flush=True)
    headers = f"{'Language':<9} | {'Scenario':<26} | {'Regex Score':<11} | {'Groq Score':<10} | {'Regex Level':<11} | {'Groq Level':<10} | {'Reg 1st High':<12} | {'Groq 1st High':<13} | {'Tactic Agree':<16}"
    print(headers, flush=True)
    print("-" * 115, flush=True)
    for r in summary_rows:
        line = f"{r['lang']:<9} | {r['scenario']:<26} | {r['regex_final_score']:<11} | {str(r['groq_final_score']):<10} | {r['regex_final_level']:<11} | {str(r['groq_final_level']):<10} | {r['regex_first_high']:<12} | {r['groq_first_high']:<13} | {r['turn_agreement']:<16}"
        print(line, flush=True)
    print("=" * 115, flush=True)

if __name__ == "__main__":
    run_all_evaluations()
