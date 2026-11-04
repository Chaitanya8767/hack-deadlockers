import math
import time
from typing import List, Dict, Optional
from pydantic import BaseModel

from config import settings
from conversation_manager import TacticEvent
from sequence_analyzer import SequenceAnalysisResult, analyze_sequence

PRESSURE_TACTICS = {"authority_impersonation", "fear_manipulation", "urgency_creation", "digital_arrest"}
SENSITIVE_ASKS = {"otp_request", "financial_manipulation", "remote_access", "kyc_fraud", "identity_manipulation"}
# A direct, high-confidence ask for these is dangerous even as the very first sentence
DIRECT_ASK_TACTICS = {"otp_request", "identity_manipulation"}

class RiskEvaluation(BaseModel):
    score: float
    level: str  # Low | Medium | High | Critical
    trend: str  # rising | stable | falling
    active_tactics: List[str]
    max_confidences: Dict[str, float]
    patterns: List[str]
    sequence_result: SequenceAnalysisResult

class RiskEngine:
    def __init__(self):
        self.smoothed_scores: Dict[str, float] = {}  # call_id -> smoothed score
        self.last_update_times: Dict[str, float] = {}

    def reset_call(self, call_id: str):
        self.smoothed_scores.pop(call_id, None)
        self.last_update_times.pop(call_id, None)

    def calculate_risk(self, call_id: str, events: List[TacticEvent], current_time: Optional[float] = None) -> RiskEvaluation:
        now = current_time or time.time()
        prev_score = self.smoothed_scores.get(call_id, 0.0)

        if not events:
            # Smoothly decay towards 0
            smoothed = max(0.0, prev_score * (1.0 - settings.risk.fall_alpha))
            self.smoothed_scores[call_id] = round(smoothed, 1)
            return RiskEvaluation(
                score=round(smoothed, 1),
                level=self._get_level(smoothed),
                trend="falling" if smoothed < prev_score else "stable",
                active_tactics=[],
                max_confidences={},
                patterns=[],
                sequence_result=SequenceAnalysisResult()
            )

        # 1. Deduplication: Keep MAX confidence per tactic with time decay
        max_conf: Dict[str, float] = {}
        half_life = settings.risk.decay_half_life_s
        seen_pressure = False
        seen_sensitive_after_pressure = False

        latest_event_t = max((ev.t for ev in events if ev.t > 0), default=0.0)
        is_relative_t = (latest_event_t < 100000)

        for e in events:
            if is_relative_t:
                ref_time = current_time if (current_time is not None and current_time < 100000) else latest_event_t
                age = max(0.0, ref_time - e.t)
            else:
                age = max(0.0, now - e.t) if e.t > 0 else 0.0

            decay_factor = math.pow(0.5, age / half_life)
            effective_conf = e.confidence * decay_factor

            if effective_conf > max_conf.get(e.tactic, 0.0):
                max_conf[e.tactic] = effective_conf

            if e.tactic in PRESSURE_TACTICS:
                seen_pressure = True
            elif e.tactic in SENSITIVE_ASKS and seen_pressure:
                seen_sensitive_after_pressure = True

        # 2. Noisy-OR combination of weights
        # P_combined = 1 - Prod(1 - w_i * c_i)
        prob_none = 1.0
        weights = settings.risk.tactic_weights

        for tactic, conf in max_conf.items():
            w = weights.get(tactic, 0.50)
            prob_none *= (1.0 - (w * conf))

        base_prob = 1.0 - prob_none  # in [0.0, 1.0]
        raw_score = base_prob * 100.0

        # 3. Sequence Analysis & Escalation Multiplier
        seq_res = analyze_sequence(events)
        raw_score *= seq_res.sequence_multiplier

        # 4. Sensitive ASK Floor Rule (~65)
        if seen_sensitive_after_pressure:
            raw_score = max(raw_score, settings.risk.sensitive_ask_floor)

        # 5. Isolated single sentence rule:
        # A single isolated sentence or single tactic must NEVER produce High/Critical (> 60)
        direct_ask = any(e.tactic in DIRECT_ASK_TACTICS and e.confidence >= 0.9 for e in events)
        if len(events) <= 1 or seq_res.distinct_stages_count <= 1:
            raw_score = min(raw_score, 75.0 if direct_ask else 55.0)

        raw_score = min(100.0, max(0.0, raw_score))

        # 6. Asymmetric smoothing (rises fast, falls slow)
        if raw_score >= prev_score:
            smoothed = (settings.risk.rise_alpha * raw_score) + ((1.0 - settings.risk.rise_alpha) * prev_score)
            trend = "rising" if smoothed > prev_score + 1.0 else "stable"
        else:
            smoothed = (settings.risk.fall_alpha * raw_score) + ((1.0 - settings.risk.fall_alpha) * prev_score)
            trend = "falling" if smoothed < prev_score - 1.0 else "stable"

        smoothed = round(min(100.0, max(0.0, smoothed)), 1)
        self.smoothed_scores[call_id] = smoothed
        self.last_update_times[call_id] = now

        level = self._get_level(smoothed)
        patterns_list = [p.pattern_name for p in seq_res.matched_patterns]

        return RiskEvaluation(
            score=smoothed,
            level=level,
            trend=trend,
            active_tactics=list(max_conf.keys()),
            max_confidences={k: round(v, 2) for k, v in max_conf.items()},
            patterns=patterns_list,
            sequence_result=seq_res
        )

    def _get_level(self, score: float) -> str:
        if score > 80.0:
            return "Critical"
        elif score > 60.0:
            return "High"
        elif score > 30.0:
            return "Medium"
        return "Low"

