import json
import os
import sys
import time
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from regex_classifier import (
    classify_regex,
    classify_regex_async,
    normalize_text,
    VALID_TACTICS,
)
from conversation_manager import Turn, TacticEvent
from risk_engine import RiskEngine

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "transcripts.json")
CONVERSATIONS_DIR = os.path.join(os.path.dirname(__file__), "conversations")

def load_transcripts():
    with open(FIXTURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def test_output_schema_conformance():
    """Verify that regex classifier returns TacticEvent objects matching the exact Groq schema."""
    sample_text = "I am CBI officer, you are under digital arrest right now, immediately tell me the OTP."
    turn = Turn(id="schema-test-1", text=sample_text, lang="en", t0=0.0, t1=4.5)
    events = classify_regex(turn)

    assert isinstance(events, list)
    assert len(events) > 0

    for ev in events:
        assert isinstance(ev, TacticEvent)
        assert ev.tactic in VALID_TACTICS
        assert 0.40 <= ev.confidence <= 1.0
        assert round(ev.confidence, 2) == ev.confidence
        assert isinstance(ev.evidence, str) and len(ev.evidence) > 0
        assert isinstance(ev.explanation, str) and len(ev.explanation) > 0
        assert ev.t == turn.t1
        assert ev.stage == 0
        assert ev.turn_id == turn.id

def test_async_wrapper_matches_sync():
    """Verify that async wrapper is a coroutine and produces identical results to sync classifier."""
    import inspect
    turn = Turn(id="async-test", text="Immediately transfer your funds to RBI safe account.", lang="en", t0=1.0, t1=5.0)
    sync_events = classify_regex(turn)
    coro = classify_regex_async(turn)
    assert inspect.iscoroutine(coro)

    try:
        coro.send(None)
        async_events = []
    except StopIteration as exc:
        async_events = exc.value

    assert len(sync_events) == len(async_events)
    for s_ev, a_ev in zip(sync_events, async_events):
        assert s_ev.tactic == a_ev.tactic
        assert s_ev.confidence == a_ev.confidence
        assert s_ev.evidence == a_ev.evidence

def test_execution_latency_under_5ms():
    """Verify that classification runs in under 5ms per utterance."""
    turn = Turn(
        id="latency-test",
        text="A federal non-bailable arrest warrant has been issued against you under the NDPS Act.",
        lang="en",
        t0=0.0,
        t1=3.0,
    )
    # Warmup
    for _ in range(5):
        classify_regex(turn)

    iterations = 200
    start = time.perf_counter()
    for _ in range(iterations):
        classify_regex(turn)
    total_elapsed = time.perf_counter() - start
    avg_ms = (total_elapsed / iterations) * 1000.0

    print(f"\nAverage regex classification latency: {avg_ms:.3f} ms")
    assert avg_ms < 5.0, f"Classifier exceeded 5ms budget: {avg_ms:.3f} ms"

def test_normalization_and_whisper_variants():
    """Test phonetic normalization for common Whisper and speech artifacts."""
    variants = [
        ("Please send the oh tee pee now", "otp_request"),
        ("Tell me the o t p code", "otp_request"),
        ("Install any desk quick support", "remote_access"),
        ("Open team viewer application", "remote_access"),
        ("Case transferred to c b i headquarters", "authority_impersonation"),
    ]
    for text, expected_tactic in variants:
        turn = Turn(id="var-test", text=text, lang="en", t0=0.0, t1=2.0)
        events = classify_regex(turn)
        detected = [e.tactic for e in events]
        assert expected_tactic in detected, f"Failed to detect {expected_tactic} in '{text}'. Got: {detected}"

def test_negation_and_benign_advisories():
    """Advisory warnings must suppress credential harvesting and sensitive asks."""
    advisories = [
        "Dear customer, bank will never ask for your OTP or password.",
        "Never share your OTP with anyone.",
        "सतर्क रहें! बैंक कभी भी आपसे फोन पर ओटीपी या पासवर्ड नहीं मांगता।",
        "बँक कधीही ग्राहकांना ओटीपी किंवा पासवर्ड विचारत नाही, कोणालाही सांगू नका.",
        "Kisi ko bhi apna OTP share mat karna, beware of fraudsters.",
    ]
    for adv in advisories:
        turn = Turn(id="adv-test", text=adv, lang="en", t0=0.0, t1=3.0)
        events = classify_regex(turn)
        # Sensitive credentials must not be detected
        detected = [e.tactic for e in events]
        assert "otp_request" not in detected, f"False positive OTP request on advisory: '{adv}'"
        assert "kyc_fraud" not in detected
        assert "financial_manipulation" not in detected

def test_fixtures_labeled_utterances():
    """Test the full set of 40+ labeled utterances from transcripts.json."""
    data = load_transcripts()
    assert len(data) >= 40, f"Expected at least 40 transcripts, got {len(data)}"

    for item in data:
        turn = Turn(id=item["id"], text=item["text"], lang=item["lang"], t0=0.0, t1=3.0)
        events = classify_regex(turn)
        detected = [e.tactic for e in events]

        if item["is_scam"]:
            # Assert all expected tactics are found
            for exp in item["expected_tactics"]:
                assert exp in detected, (
                    f"[{item['id']}] Expected tactic '{exp}' not detected in '{item['text']}'. "
                    f"Detected: {detected}"
                )
        else:
            # Benign utterances must produce no tactics
            assert len(detected) == 0, (
                f"[{item['id']}] False positive tactics {detected} detected in benign utterance: '{item['text']}'"
            )

def test_conversations_risk_progression():
    """
    Test all 4 benchmark conversations (en, hi, hinglish, mr):
    scam calls must reach High/Critical (> 60), benign calls must stay Low (<= 30).
    """
    for lang in ["en", "hi", "hinglish", "mr"]:
        file_path = os.path.join(CONVERSATIONS_DIR, f"{lang}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            conv_data = json.load(f)

        for scenario, turns in conv_data.items():
            if scenario.startswith("_"):
                continue

            engine = RiskEngine()
            events = []
            history = []
            has_prior = False

            for i, t in enumerate(turns):
                turn = Turn(id=f"{scenario}-{i}", text=t["text"], lang=lang, t0=i * 5.0, t1=(i + 1) * 5.0)
                context_str = "\n".join(history)
                turn_events = classify_regex(turn, context_str=context_str, has_prior_tactics=has_prior)
                if turn_events:
                    has_prior = True
                events.extend(turn_events)
                history.append(f"[{lang.upper()}] Caller: \"{t['text']}\"")
                risk = engine.calculate_risk(scenario, events)

            if scenario.startswith("scam_"):
                assert risk.level in ["High", "Critical"], (
                    f"Scam scenario '{scenario}' ({lang}) only reached {risk.score} ({risk.level})"
                )
                assert risk.score >= 70.0, f"Scam scenario '{scenario}' score too low: {risk.score}"
            elif scenario.startswith("benign_"):
                assert risk.level == "Low", (
                    f"Benign scenario '{scenario}' ({lang}) reached non-low level: {risk.level} ({risk.score})"
                )
                assert risk.score <= 30.0, f"Benign scenario '{scenario}' score too high: {risk.score}"
