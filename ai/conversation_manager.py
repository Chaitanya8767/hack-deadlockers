import time
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class Turn(BaseModel):
    id: str
    speaker: str = "caller"  # Audio captured is strictly the caller
    text: str
    lang: str
    t0: float
    t1: float
    timestamp: float = Field(default_factory=time.time)

class TacticEvent(BaseModel):
    tactic: str
    confidence: float
    evidence: str
    explanation: str
    t: float
    stage: int = 0

class ConversationManager:
    """
    Maintains per-call conversation history, windowed context (N=6),
    and accumulated detected tactics.
    """
    def __init__(self, context_turns: int = 6):
        self.context_turns = context_turns
        self.turns: Dict[str, List[Turn]] = {}
        self.tactic_events: Dict[str, List[TacticEvent]] = {}

    def reset_call(self, call_id: str):
        self.turns.pop(call_id, None)
        self.tactic_events.pop(call_id, None)

    def add_turn(self, call_id: str, turn: Turn):
        call_turns = self.turns.setdefault(call_id, [])
        call_turns.append(turn)

    def get_recent_context(self, call_id: str, n: Optional[int] = None) -> List[Turn]:
        limit = n or self.context_turns
        call_turns = self.turns.get(call_id, [])
        return call_turns[-limit:] if call_turns else []

    def format_context_for_prompt(self, call_id: str, current_turn: Turn) -> str:
        history = self.get_recent_context(call_id, self.context_turns)
        lines = []
        for t in history:
            if t.id != current_turn.id:
                lines.append(f"[{t.lang.upper()}] Caller: \"{t.text}\"")
        if not lines:
            return "No previous utterances."
        return "\n".join(lines)

    def add_tactics(self, call_id: str, events: List[TacticEvent]):
        call_events = self.tactic_events.setdefault(call_id, [])
        call_events.extend(events)

    def get_tactics(self, call_id: str) -> List[TacticEvent]:
        return self.tactic_events.get(call_id, [])

