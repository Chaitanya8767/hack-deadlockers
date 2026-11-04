import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from conversation_manager import Turn, TacticEvent
from sequence_analyzer import analyze_sequence
from risk_engine import RiskEngine
from alert_generator import generate_alert

CONV_DIR = os.path.join(os.path.dirname(__file__), "conversations")

def load_conversation(lang_code: str, scenario_key: str):
    file_path = os.path.join(CONV_DIR, f"{lang_code}.json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[scenario_key]

def test_benign_calls_stay_low():
    engine = RiskEngine()
    for lang, scenario in [
        ("en", "benign_flight_support"),
        ("hi", "benign_food_delivery"),
        ("mr", "benign_family_call"),
        ("hinglish", "benign_dinner_plan"),
    ]:
        call_id = f"test-benign-{lang}"
        turns = load_conversation(lang, scenario)
        events = []
        for i, t in enumerate(turns):
            # Benign turns produce no scam tactics
            assert len(t["expected_tactics"]) == 0
            risk = engine.calculate_risk(call_id, events)
            assert risk.score <= 30.0, f"Benign turn {i} in {lang} got high score: {risk.score}"
            assert risk.level == "Low", f"Benign call in {lang} should be Low, got {risk.level}"

def test_scam_flow_monotonic_risk_increase():
    """
    Asserts that as scam tactics progress through stages (Hook -> Fear -> Urgency -> Money/OTP),
    the calculated risk score increases monotonically and culminates in High/Critical.
    """
    engine = RiskEngine()
    call_id = "test-scam-digital-arrest-en"
    turns = load_conversation("en", "scam_digital_arrest")

    events = []
    scores = []
    t_clock = 1000.0

    for i, t in enumerate(turns):
        t_clock += 10.0
        for tactic in t["expected_tactics"]:
            events.append(TacticEvent(
                tactic=tactic,
                confidence=0.90,
                evidence=t["text"][:30],
                explanation=f"Testing {tactic}",
                t=t_clock
            ))

        risk = engine.calculate_risk(call_id, events, current_time=t_clock)
        scores.append(risk.score)

    print(f"Digital arrest risk progression: {scores}")
    # Single isolated sentence rule: first turn must not be High/Critical
    assert scores[0] <= 55.0, f"Turn 0 score exceeded single-sentence cap: {scores[0]}"

    # Risk must be strictly non-decreasing (monotonic up)
    for i in range(1, len(scores)):
        assert scores[i] >= scores[i-1], f"Risk dropped from {scores[i-1]} to {scores[i]} at step {i}"

    # Final score for full digital arrest must be High or Critical (> 60)
    assert scores[-1] >= 61.0, f"Final digital arrest score too low: {scores[-1]}"

def test_sensitive_ask_floor_rule():
    """
    Asserts that when a sensitive ask (OTP, funds transfer, remote access)
    follows a pressure tactic, the score is floored at >= 65.
    """
    engine = RiskEngine()
    call_id = "test-floor"
    t_clock = 100.0

    # Step 1: Pressure tactic (authority / fear)
    events = [
        TacticEvent(tactic="authority_impersonation", confidence=0.85, evidence="CBI officer", explanation="Authority", t=t_clock)
    ]
    r1 = engine.calculate_risk(call_id, events, current_time=t_clock)

    # Step 2: Sensitive ask (otp_request) following pressure
    t_clock += 5.0
    events.append(
        TacticEvent(tactic="otp_request", confidence=0.95, evidence="Share OTP", explanation="OTP request", t=t_clock)
    )
    r2 = engine.calculate_risk(call_id, events, current_time=t_clock)
    assert r2.score >= 65.0, f"Expected sensitive ask floor >= 65, got {r2.score}"
    assert r2.level in ["High", "Critical"]

def test_sequence_analyzer_detection():
    events = [
        TacticEvent(tactic="authority_impersonation", confidence=0.9, evidence="", explanation="", t=1.0),
        TacticEvent(tactic="digital_arrest", confidence=0.9, evidence="", explanation="", t=2.0),
        TacticEvent(tactic="urgency_creation", confidence=0.9, evidence="", explanation="", t=3.0),
        TacticEvent(tactic="financial_manipulation", confidence=0.9, evidence="", explanation="", t=4.0),
    ]
    res = analyze_sequence(events)
    assert res.distinct_stages_count >= 4
    assert res.escalation_detected is True
    assert res.sequence_multiplier > 1.20
    matched_names = [p.pattern_name for p in res.matched_patterns]
    assert "digital_arrest_flow" in matched_names

def test_alert_generator():
    a_crit = generate_alert("Critical", ["otp_request", "authority_impersonation"])
    assert a_crit.level == "Critical"
    assert "DO NOT SHARE OTP" in a_crit.action

    a_low = generate_alert("Low", [])
    assert a_low.level == "Low"
    assert "normal" in a_low.action

if __name__ == "__main__":
    pytest.main(["-v", __file__])

