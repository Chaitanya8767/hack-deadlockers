"""
Risk progression demo — runs purely through risk engine + sequence analyzer
(no LLM needed) to show how risk escalates turn-by-turn for a digital arrest scam.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv; load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from conversation_manager import TacticEvent
from sequence_analyzer import analyze_sequence
from risk_engine import RiskEngine
from alert_generator import generate_alert

SCAM_CALL_PROGRESSION = [
    # (t_seconds, tactic, confidence, text_sample)
    (5.0,  "authority_impersonation", 0.93, "I am Senior Inspector from CBI Cyber Crime Branch"),
    (14.0, "fear_manipulation",       0.91, "Illegal narcotics found in parcel registered in your name"),
    (22.0, "digital_arrest",          0.96, "You are now under digital arrest. Do not disconnect."),
    (31.0, "urgency_creation",        0.89, "Camera on, stay in your room, do not inform anyone"),
    (40.0, "financial_manipulation",  0.94, "Transfer savings to RBI verification account in 30 minutes"),
    (48.0, "otp_request",             0.98, "Share the OTP received on your phone for identity verification"),
]

BENIGN_CALL_PROGRESSION = [
    (4.0,  None, 0,    "Hello, calling from society office about annual maintenance"),
    (9.0,  None, 0,    "Fee is due this month, can be paid at office"),
    (15.0, None, 0,    "Office open 10 AM to 5 PM on weekdays"),
]

def run_scam(title, progression, expect_high=True):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")
    print(f"  {'Turn':<5} {'Risk':>6} {'Level':<10} {'Stage':>5}  Alert")
    print(f"  {'-'*55}")

    engine = RiskEngine()
    events = []
    scores = []

    for i, row in enumerate(progression):
        t, tactic, conf, text = row
        if tactic:
            events.append(TacticEvent(
                tactic=tactic, confidence=conf,
                evidence=text[:45], explanation=f"Detected {tactic}", t=t
            ))
        risk = engine.calculate_risk("demo", events, current_time=t)
        seq  = analyze_sequence(events)
        alert = generate_alert(risk.level, risk.active_tactics)
        scores.append(risk.score)

        bar = "#" * int(risk.score / 5)
        level_icon = {"Low":"[LOW]","Medium":"[MED]","High":"[HIGH]","Critical":"[CRIT]"}.get(risk.level,"[?]")
        print(f"  T={t:4.0f}s  {risk.score:5.1f}  {level_icon:<6} {risk.level:<10} max_stage={seq.max_stage}  {alert.message[:40]}...")

    print(f"\n  Progression: {[round(s,1) for s in scores]}")
    if expect_high:
        ok = scores[-1] >= 61.0 and all(scores[i] >= scores[i-1]-3 for i in range(1,len(scores)))
        print(f"  checkmark PASS -- risk escalates correctly" if ok else f"  FAIL")
    else:
        ok = max(scores) <= 35.0
        print(f"  checkmark PASS -- benign call stays Low" if ok else f"  FAIL -- score too high: {max(scores)}")

run_scam("SCAM CALL: Digital Arrest (CBI impersonation + OTP request)", SCAM_CALL_PROGRESSION, expect_high=True)
run_scam("BENIGN CALL: Society Office Reminder", [(t,None,0,txt) for t,_,__,txt in BENIGN_CALL_PROGRESSION], expect_high=False)
