from typing import List, Dict, Set
from pydantic import BaseModel

from conversation_manager import TacticEvent

TACTIC_STAGES = {
    # Stage 0: Hook / Identity / Authority / Reward
    "authority_impersonation": 0,
    "identity_manipulation": 3,
    "reward_scam": 0,

    # Stage 1: Threat / Coercion
    "fear_manipulation": 1,
    "digital_arrest": 1,

    # Stage 2: Pressure / Time limits
    "urgency_creation": 2,

    # Stage 3: Vulnerability exploit (Sensitive ask)
    "remote_access": 3,
    "kyc_fraud": 3,
    "otp_request": 3,

    # Stage 4: Extraction
    "financial_manipulation": 4,
}

KNOWN_FLOWS = {
    "digital_arrest_flow": [0, 1, 2, 4],     # Authority -> Threat/Arrest -> Urgency -> Money
    "otp_phish_flow": [0, 2, 3],              # Identity/Authority -> Urgency -> OTP
    "remote_access_flow": [0, 3],             # Authority -> Remote App
    "kyc_freeze_flow": [0, 1, 3],             # Authority -> Threat of Block -> KYC/OTP
    "lottery_reward_flow": [0, 2, 4]          # Reward -> Urgency -> Fee Transfer
}

class MatchedPattern(BaseModel):
    pattern_name: str
    completion_ratio: float
    matched_stages: List[int]

class SequenceAnalysisResult(BaseModel):
    matched_patterns: List[MatchedPattern] = []
    distinct_stages_count: int = 0
    max_stage: int = 0
    escalation_detected: bool = False
    sequence_multiplier: float = 1.0

def assign_stages(events: List[TacticEvent]) -> List[TacticEvent]:
    for e in events:
        e.stage = TACTIC_STAGES.get(e.tactic, 0)
    return events

def analyze_sequence(tactic_events: List[TacticEvent]) -> SequenceAnalysisResult:
    if not tactic_events:
        return SequenceAnalysisResult()

    events = assign_stages(tactic_events)
    # Chronological sequence of unique stage transitions
    stage_sequence = [e.stage for e in events]
    distinct_stages: Set[int] = set(stage_sequence)
    max_stage = max(stage_sequence) if stage_sequence else 0

    matched_patterns: List[MatchedPattern] = []
    max_completion = 0.0

    for flow_name, target_flow in KNOWN_FLOWS.items():
        # Check subsequence match
        target_idx = 0
        stages_found = []
        for s in stage_sequence:
            if target_idx < len(target_flow) and s == target_flow[target_idx]:
                stages_found.append(s)
                target_idx += 1

        completion = len(stages_found) / float(len(target_flow))
        if completion >= 0.50:  # At least half of the sequence observed
            matched_patterns.append(MatchedPattern(
                pattern_name=flow_name,
                completion_ratio=round(completion, 2),
                matched_stages=stages_found
            ))
            max_completion = max(max_completion, completion)

    # Escalation condition: progression across >= 3 stages reaching stage >= 2
    escalation_detected = (len(distinct_stages) >= 3 and max_stage >= 2)

    # Multiplier: Base 1.0, scales with pattern completion and distinct stages
    multiplier = 1.0
    if len(distinct_stages) >= 2:
        multiplier += 0.10 * (len(distinct_stages) - 1)
    if escalation_detected:
        multiplier += 0.20
    if max_completion >= 0.75:
        multiplier += 0.15

    multiplier = min(1.60, round(multiplier, 2))

    return SequenceAnalysisResult(
        matched_patterns=matched_patterns,
        distinct_stages_count=len(distinct_stages),
        max_stage=max_stage,
        escalation_detected=escalation_detected,
        sequence_multiplier=multiplier
    )

