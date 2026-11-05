"""
Script to compare Groq LLM classification and Regex classification
on labeled transcripts from ScamChain AI benchmarks.
Computes per-tactic precision, recall, F1, and inter-classifier agreement.
"""

import inspect
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Set

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from conversation_manager import Turn
from regex_classifier import classify_regex, VALID_TACTICS
from tactic_classifier import TacticClassifier

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "transcripts.json")

def load_transcripts():
    with open(FIXTURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def run_coroutine(coro):
    if inspect.iscoroutine(coro):
        try:
            coro.send(None)
        except StopIteration as exc:
            return exc.value
    return coro

def evaluate_classifiers():
    transcripts = load_transcripts()
    # If run in sandbox without network, default to regex benchmark
    force_regex_only = "--regex-only" in sys.argv
    has_groq_key = bool(settings.llm.groq_api_key) and not force_regex_only

    classifier = TacticClassifier()
    settings.llm.classifier_mode = "llm"

    print(f"Loaded {len(transcripts)} evaluation transcripts.")
    print(f"Groq API Key status: {'ACTIVE' if has_groq_key else 'UNSET / OFFLINE (Regex Mode)'}\n")

    regex_tp = defaultdict(int)
    regex_fp = defaultdict(int)
    regex_fn = defaultdict(int)

    llm_tp = defaultdict(int)
    llm_fp = defaultdict(int)
    llm_fn = defaultdict(int)

    agreements = defaultdict(int)
    total_evals = 0

    print(f"{'ID':<8} | {'Lang':<5} | {'Expected Tactics':<35} | {'Regex Detected':<35} | {'LLM Detected':<25}")
    print("-" * 115)

    for item in transcripts:
        turn = Turn(id=item["id"], text=item["text"], lang=item["lang"], t0=0.0, t1=3.0)
        expected = set(item.get("expected_tactics", []))

        # 1. Regex Classifier
        regex_events = classify_regex(turn)
        regex_tactics = set(e.tactic for e in regex_events)

        # 2. LLM Classifier (if active)
        llm_tactics = set()
        if has_groq_key:
            try:
                events = run_coroutine(classifier.classify_utterance(turn, context_str=""))
                llm_tactics = set(e.tactic for e in events)
            except Exception:
                llm_tactics = set()

        total_evals += 1

        for t in VALID_TACTICS:
            in_exp = t in expected
            in_reg = t in regex_tactics
            in_llm = t in llm_tactics

            if in_exp and in_reg:
                regex_tp[t] += 1
            elif not in_exp and in_reg:
                regex_fp[t] += 1
            elif in_exp and not in_reg:
                regex_fn[t] += 1

            if has_groq_key:
                if in_exp and in_llm:
                    llm_tp[t] += 1
                elif not in_exp and in_llm:
                    llm_fp[t] += 1
                elif in_exp and not in_llm:
                    llm_fn[t] += 1

                if in_reg == in_llm:
                    agreements[t] += 1

        reg_str = ", ".join(regex_tactics) if regex_tactics else "(none)"
        exp_str = ", ".join(expected) if expected else "(none)"
        llm_str = (", ".join(llm_tactics) if llm_tactics else "(none)") if has_groq_key else "N/A"

        print(f"{item['id']:<8} | {item['lang']:<5} | {exp_str[:35]:<35} | {reg_str[:35]:<35} | {llm_str[:25]:<25}")

    print("\n" + "=" * 80)
    print("PER-TACTIC BENCHMARK RESULTS (Regex vs Ground Truth)")
    print("=" * 80)
    print(f"{'Tactic':<25} | {'Prec':<8} | {'Recall':<8} | {'F1':<8} | {'Agreement (Regex/LLM)':<20}")
    print("-" * 80)

    for t in sorted(VALID_TACTICS):
        tp = regex_tp[t]
        fp = regex_fp[t]
        fn = regex_fn[t]

        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        agree_str = f"{(agreements[t] / total_evals) * 100:.1f}%" if has_groq_key else "N/A"
        print(f"{t:<25} | {prec:<8.2f} | {rec:<8.2f} | {f1:<8.2f} | {agree_str:<20}")

    print("=" * 80)

if __name__ == "__main__":
    evaluate_classifiers()
