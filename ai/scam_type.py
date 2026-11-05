"""Explainable, rule-only scam-type classification."""
from collections import defaultdict


TYPE_RULES = {
    "Digital Arrest": ({"digital_arrest": 5, "authority_impersonation": 2, "fear_manipulation": 2, "financial_manipulation": 2}, {"digital_arrest_flow": 4}),
    "Fake KYC": ({"kyc_fraud": 5, "identity_manipulation": 2, "otp_request": 2, "fear_manipulation": 1}, {"kyc_freeze_flow": 4}),
    "OTP Phishing": ({"otp_request": 5, "identity_manipulation": 2, "urgency_creation": 2}, {"otp_phish_flow": 4}),
    "Remote Access": ({"remote_access": 6, "authority_impersonation": 1, "urgency_creation": 1}, {"remote_access_flow": 4}),
    "Reward/Job Scam": ({"reward_scam": 6, "financial_manipulation": 2, "urgency_creation": 2}, {"lottery_reward_flow": 4}),
    "Utility/Bill Threat": ({"kyc_fraud": 2, "fear_manipulation": 3, "urgency_creation": 2, "financial_manipulation": 1}, {}),
    "Impersonation (generic)": ({"authority_impersonation": 5, "identity_manipulation": 2, "fear_manipulation": 1}, {}),
}


def _value(item, key, default=None):
    if isinstance(item, str) and key in {"pattern_name", "name"}:
        return item
    return getattr(item, key, item.get(key, default) if isinstance(item, dict) else default)


def classify_scam_type(events, flows, risk_history):
    """Return a stable classification and exact caller-sentence evidence.

    Events are TacticEvent objects/dicts; flows are SequenceAnalysisResult patterns
    or dicts. Low final risk is deliberately reported as Unknown/Benign.
    """
    last = risk_history[-1] if risk_history else {}
    level = _value(last, "level", "Low")
    score = float(_value(last, "score", 0) or 0)
    if level == "Low" or score <= 30:
        return {"type": "Unknown/Benign", "confidence": 0.9, "runners_up": [], "evidence": []}

    flow_names = {_value(f, "pattern_name", _value(f, "name", "")) for f in (flows or [])}
    scores, supporting = defaultdict(float), defaultdict(list)
    for event in events or []:
        tactic, confidence = _value(event, "tactic", ""), float(_value(event, "confidence", 0) or 0)
        for name, (tactics, _flow_weights) in TYPE_RULES.items():
            if tactic in tactics:
                scores[name] += tactics[tactic] * confidence
                supporting[name].append(event)
    for name, (_tactics, flow_weights) in TYPE_RULES.items():
        for flow, weight in flow_weights.items():
            if flow in flow_names:
                scores[name] += weight
    if not scores:
        return {"type": "Impersonation (generic)", "confidence": round(min(.75, score / 100), 2), "runners_up": [], "evidence": []}

    ranked = sorted(scores, key=lambda name: scores[name], reverse=True)
    winner = ranked[0]
    evidence = sorted(supporting[winner], key=lambda e: _value(e, "confidence", 0), reverse=True)[:5]
    return {
        "type": winner,
        "confidence": round(min(.99, .45 + scores[winner] / 16), 2),
        "runners_up": [{"type": name, "confidence": round(min(.95, .35 + scores[name] / 16), 2)} for name in ranked[1:3]],
        "evidence": [{"turn_id": _value(e, "turn_id", ""), "t": _value(e, "t", 0), "text": _value(e, "sentence", _value(e, "text", _value(e, "evidence", ""))), "tactic": _value(e, "tactic", "")} for e in evidence],
    }
