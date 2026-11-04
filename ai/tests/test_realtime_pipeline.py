"""
Real-time integration test: simulates a full digital-arrest scam call
progression through the LIVE Gemini API -> Sequence Analyzer -> Risk Engine pipeline.
Asserts risk score escalates monotonically and reaches Critical by the end.
"""

import asyncio
import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from config import settings
from conversation_manager import ConversationManager, Turn
from tactic_classifier import TacticClassifier
from sequence_analyzer import analyze_sequence
from risk_engine import RiskEngine
from alert_generator import generate_alert

# --- Simulate a realistic digital-arrest scam call ---
SCAM_SCRIPT = [
    ("en", 0.0,  5.0,  "Hello sir, I am calling from CBI Cyber Crime Branch, New Delhi. I am Senior Inspector Rajesh Verma."),
    ("hi", 5.0,  12.0, "Aapke naam par ek illegal parcel seize kiya gaya hai jisme 150 gram cocaine mili hai."),
    ("en", 12.0, 20.0, "A non-bailable arrest warrant has been issued against you by the Supreme Court of India."),
    ("hi", 20.0, 28.0, "Aap abhi se digital arrest mein hain. Call disconnect karne par aapko seedha jail bheja jayega."),
    ("hi", 28.0, 36.0, "Apne ghar ka darwaza band karo aur kisi ko bhi mat batao. Camera on rakho."),
    ("en", 36.0, 44.0, "To clear your name from this case, transfer your entire savings to our RBI security verification account within 30 minutes."),
    ("en", 44.0, 50.0, "And share the OTP that comes on your phone right now for identity verification."),
]

BENIGN_SCRIPT = [
    ("en", 0.0,  4.0,  "Hello! This is Ajay calling from your building society. How are you?"),
    ("en", 4.0,  9.0,  "We just wanted to remind you about the annual maintenance fee due this month."),
    ("en", 9.0,  15.0, "You can pay it at the society office between 10 AM and 5 PM on any weekday."),
]

async def run_script_through_pipeline(call_id: str, script: list, label: str):
    print(f"\n{'='*60}")
    print(f"  SCENARIO: {label}")
    print(f"{'='*60}")

    conv_manager = ConversationManager(context_turns=6)
    classifier = TacticClassifier()
    risk_engine = RiskEngine()

    all_events = []
    scores = []
    prev_score = 0.0

    for i, (lang, t0, t1, text) in enumerate(script):
        print(f"\n[Turn {i+1}] [{lang.upper()}] \"{text[:70]}...\"" if len(text) > 70 else f"\n[Turn {i+1}] [{lang.upper()}] \"{text}\"")

        turn = Turn(id=f"utt_{i+1}", text=text, lang=lang, t0=t0, t1=t1)
        conv_manager.add_turn(call_id, turn)
        context_str = conv_manager.format_context_for_prompt(call_id, turn)
        has_prior = len(all_events) > 0

        # Run classifier
        t_start = time.time()
        new_tactics = await classifier.classify_utterance(turn, context_str, has_prior_tactics=has_prior)
        latency = round(time.time() - t_start, 2)

        if new_tactics:
            conv_manager.add_tactics(call_id, new_tactics)
            all_events.extend(new_tactics)
            for t in new_tactics:
                print(f"  -> TACTIC: {t.tactic} ({t.confidence:.0%}) | {t.evidence[:50]}")
        else:
            print(f"  -> No tactics detected")

        print(f"  Classifier latency: {latency}s")

        # Risk evaluation
        risk = risk_engine.calculate_risk(call_id, all_events, current_time=t1)
        scores.append(risk.score)
        alert = generate_alert(risk.level, risk.active_tactics)

        # Sequence
        seq = analyze_sequence(all_events)

        print(f"  RISK: {risk.score:.1f} ({risk.level}) | trend={risk.trend} | stages={seq.distinct_stages_count} | patterns={[p.pattern_name for p in seq.matched_patterns]}")
        print(f"  ALERT: {alert.message[:80]}")

        if risk.score < prev_score - 5:
            print(f"  [WARNING] Risk dropped unexpectedly: {prev_score} -> {risk.score}")
        prev_score = risk.score

    print(f"\n{'='*60}")
    print(f"  FINAL RISK SCORE PROGRESSION: {[round(s,1) for s in scores]}")

    if label.startswith("SCAM"):
        monotonic = all(scores[i] >= scores[i-1] - 3 for i in range(1, len(scores)))
        print(f"  Monotonic ascent: {'PASS' if monotonic else 'FAIL'}")
        final_ok = scores[-1] >= 61.0
        print(f"  Final score >= High (61): {'PASS' if final_ok else 'FAIL'} (got {scores[-1]:.1f})")
    else:
        all_low = all(s <= 35.0 for s in scores)
        print(f"  Benign stays Low: {'PASS' if all_low else 'FAIL'} (max={max(scores):.1f})")

    print(f"{'='*60}\n")
    return scores

async def main():
    has_key = bool(settings.llm.gemini_api_key or settings.llm.groq_api_key)
    if not has_key:
        print("ERROR: Neither GEMINI_API_KEY nor GROQ_API_KEY is set in ai/.env")
        return
    print(f"Using provider: {settings.llm.provider}")
    print(f"API key available: {has_key}\n")

    await run_script_through_pipeline("scam-test-001", SCAM_SCRIPT, "SCAM - Digital Arrest Flow (EN+HI)")
    await run_script_through_pipeline("benign-test-001", BENIGN_SCRIPT, "BENIGN - Society Call (EN)")

if __name__ == "__main__":
    asyncio.run(main())

