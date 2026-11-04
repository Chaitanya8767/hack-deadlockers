"""
Learning layer for the word-based detector.

Solves "the scammer used a word that is not in our list":

  1. skeleton()      - folds spelling variants together (ASR writes the same
                       Marathi word 3 different ways; English gets typos).
  2. fuzzy_hits()    - catches near-misses of known scam words.
  3. CustomRules     - extra phrases stored in custom_rules.json. Edited LIVE
                       (no restart) through /rules/add or by hand.
  4. log_unknown()   - sentences from a call that is ALREADY suspicious but
                       matched no rule are saved to unknown_phrases.jsonl so
                       the team can review them and add them as rules.
"""
import difflib
import json
import os
import re
import threading
import time
import unicodedata
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
CUSTOM_RULES_PATH = os.getenv("CUSTOM_RULES_PATH", os.path.join(_HERE, "custom_rules.json"))
UNKNOWN_LOG_PATH = os.getenv("UNKNOWN_LOG_PATH", os.path.join(_HERE, "unknown_phrases.jsonl"))

VALID_TACTICS = {
    "authority_impersonation", "urgency_creation", "fear_manipulation",
    "otp_request", "financial_manipulation", "kyc_fraud", "digital_arrest",
    "remote_access", "reward_scam", "identity_manipulation",
}

# opt-in: in hybrid mode, phrases the LLM catches are added as rules automatically
AUTO_LEARN = os.getenv("AUTO_LEARN", "false").lower() == "true"

_lock = threading.Lock()

# ---------------------------------------------------------------------------
# 1. skeleton: one canonical spelling for variants
# ---------------------------------------------------------------------------
_DEVA_FOLD = str.maketrans({
    "\u093c": "",          # nukta
    "\u200d": "", "\u200c": "",
    "\u0901": "\u0902",    # chandrabindu -> anusvara
    "\u0949": "\u094b",    # ॉ -> ो   (ऑ-sound)
    "\u0945": "\u0947",    # ॅ -> े
    "\u0911": "\u0913",    # ऑ -> ओ
    "\u090d": "\u090f",    # ऍ -> ए
    "\u0940": "\u093f",    # ी -> ि   (long/short i)
    "\u0942": "\u0941",    # ू -> ु
    "\u0908": "\u0907",    # ई -> इ
    "\u090a": "\u0909",    # ऊ -> उ
    "\u0948": "\u0947",    # ै -> े
    "\u0910": "\u090f",    # ऐ -> ए
    "\u0914": "\u0913",    # औ -> ओ
    "\u094c": "\u094b",    # ौ -> ो
})


def skeleton(text: str) -> str:
    t = unicodedata.normalize("NFC", text.lower()).translate(_DEVA_FOLD)
    t = re.sub(r"([a-z])\1+", r"\1", t)          # otpp -> otp, passsword -> pasword
    t = re.sub(r"[.\-_,;:!?\"'()]", " ", t)        # o.t.p -> o t p
    return re.sub(r"\s+", " ", t).strip()


_TOKEN = re.compile(r"[A-Za-z0-9\u0900-\u0963\u0966-\u097F]+")


def tokens(text: str) -> List[str]:
    return _TOKEN.findall(text)


# ---------------------------------------------------------------------------
# 2. fuzzy: near-miss of distinctive scam words
# ---------------------------------------------------------------------------
# word -> (tactic, is_sensitive_item, label)
FUZZY_VOCAB: Dict[str, Tuple[str, bool, str]] = {
    # sensitive items
    "otp": ("otp_request", True, "OTP"),
    "password": ("otp_request", True, "password"),
    "passcode": ("otp_request", True, "passcode"),
    "cvv": ("otp_request", True, "CVV"),
    "aadhaar": ("identity_manipulation", True, "Aadhaar"),
    "aadhar": ("identity_manipulation", True, "Aadhaar"),
    "ओटीपी": ("otp_request", True, "OTP"),
    "पासवर्ड": ("otp_request", True, "password"),
    "सीव्हीव्ही": ("otp_request", True, "CVV"),
    "आधार": ("identity_manipulation", True, "Aadhaar"),
    # tactics
    "anydesk": ("remote_access", False, "AnyDesk"),
    "teamviewer": ("remote_access", False, "TeamViewer"),
    "quicksupport": ("remote_access", False, "QuickSupport"),
    "rustdesk": ("remote_access", False, "RustDesk"),
    "apk": ("remote_access", False, "APK file"),
    "एनीडेस्क": ("remote_access", False, "AnyDesk"),
    "टीमव्ह्यूअर": ("remote_access", False, "TeamViewer"),
    "एपीके": ("remote_access", False, "APK file"),
    "kyc": ("kyc_fraud", False, "KYC"),
    "केवायसी": ("kyc_fraud", False, "KYC"),
    "lottery": ("reward_scam", False, "lottery"),
    "jackpot": ("reward_scam", False, "jackpot"),
    "cashback": ("reward_scam", False, "cashback"),
    "लॉटरी": ("reward_scam", False, "lottery"),
    "narcotics": ("fear_manipulation", False, "narcotics"),
    "digital": None,  # placeholder removed below
}
FUZZY_VOCAB.pop("digital")

# words that look close to a scam word but are normal speech
FUZZY_IGNORE = {"warranty", "pincode", "lotteries", "apex", "kyoto", "aadhaarcard"}

_SKEL_VOCAB: Dict[str, Tuple[str, bool, str]] = {skeleton(w): v for w, v in FUZZY_VOCAB.items()}


def fuzzy_hits(norm_text: str) -> List[Tuple[str, str, bool, str, float]]:
    """Returns (matched_word, tactic, is_sensitive, label, confidence).
    exact skeleton match (spelling variant) -> 0.70 ; near miss -> 0.55"""
    out = []
    vocab_keys = list(_SKEL_VOCAB.keys())
    for tok in tokens(norm_text):
        if tok in FUZZY_IGNORE:
            continue
        sk = skeleton(tok)
        if not sk or len(sk) < 3:
            continue
        if sk in _SKEL_VOCAB:
            tactic, sens, label = _SKEL_VOCAB[sk]
            out.append((tok, tactic, sens, label, 0.70))
            continue
        if len(sk) < 6:
            continue
        near = difflib.get_close_matches(sk, vocab_keys, n=1, cutoff=0.84)
        if near and abs(len(near[0]) - len(sk)) <= 2:
            tactic, sens, label = _SKEL_VOCAB[near[0]]
            out.append((tok, tactic, sens, label, 0.55))
    return out


# ---------------------------------------------------------------------------
# 3. custom rules (hot reload)
# ---------------------------------------------------------------------------
class CustomRules:
    def __init__(self, path: str = CUSTOM_RULES_PATH):
        self.path = path
        self._mtime = -1.0
        self._compiled: List[Tuple[str, re.Pattern, float, str, str]] = []
        self._raw: List[dict] = []

    def _read(self) -> List[dict]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return list(data.get("rules", []))
        except FileNotFoundError:
            return []
        except Exception:
            return self._raw  # keep last good copy if file is half-written

    def _reload_if_changed(self):
        try:
            m = os.path.getmtime(self.path)
        except OSError:
            m = 0.0
        if m == self._mtime:
            return
        with _lock:
            self._mtime = m
            self._raw = self._read()
            compiled = []
            for r in self._raw:
                phrase = skeleton(str(r.get("phrase", "")))
                tactic = r.get("tactic")
                if not phrase or tactic not in VALID_TACTICS:
                    continue
                body = r"\s+".join(re.escape(p) for p in phrase.split())
                if re.match(r"[a-z0-9]", phrase[0]):
                    body = r"(?<![a-z0-9])" + body
                if re.match(r"[a-z0-9]", phrase[-1]):
                    body = body + r"(?![a-z0-9])"
                try:
                    compiled.append((phrase, re.compile(body), float(r.get("confidence", 0.75)),
                                     tactic, str(r.get("note", ""))))
                except re.error:
                    continue
            self._compiled = compiled

    def match(self, norm_text: str) -> List[Tuple[str, str, float, str]]:
        """-> list of (phrase, tactic, confidence, note)"""
        self._reload_if_changed()
        sk = skeleton(norm_text)
        return [(p, tac, conf, note) for p, rx, conf, tac, note in self._compiled if rx.search(sk)]

    def list(self) -> List[dict]:
        self._reload_if_changed()
        return list(self._raw)

    def add(self, phrase: str, tactic: str, confidence: float = 0.75, note: str = "") -> dict:
        phrase = re.sub(r"\s+", " ", phrase.strip())
        if len(skeleton(phrase)) < 3:
            raise ValueError("phrase too short")
        if tactic not in VALID_TACTICS:
            raise ValueError(f"tactic must be one of {sorted(VALID_TACTICS)}")
        confidence = max(0.3, min(0.99, float(confidence)))
        with _lock:
            rules = self._read()
            rules = [r for r in rules if skeleton(r.get("phrase", "")) != skeleton(phrase)]
            entry = {"phrase": phrase, "tactic": tactic, "confidence": confidence,
                     "note": note, "added": time.strftime("%Y-%m-%d %H:%M:%S")}
            rules.append(entry)
            self._write(rules)
        self._mtime = -1.0
        return entry

    def remove(self, phrase: str) -> int:
        with _lock:
            rules = self._read()
            keep = [r for r in rules if skeleton(r.get("phrase", "")) != skeleton(phrase)]
            self._write(keep)
        self._mtime = -1.0
        return len(rules) - len(keep)

    def _write(self, rules: List[dict]):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"rules": rules}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)


custom_rules = CustomRules()


# ---------------------------------------------------------------------------
# 4. unknown-phrase log (review queue)
# ---------------------------------------------------------------------------
def log_unknown(text: str, reason: str, suggested_tactic: Optional[str] = None,
                evidence: Optional[str] = None) -> None:
    """Append a sentence the rules did not understand. Only called for calls
    that are already suspicious, so ordinary chit-chat is not stored."""
    text = (text or "").strip()
    if len(text) < 4:
        return
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "text": text[:300], "reason": reason}
    if suggested_tactic:
        rec["suggested_tactic"] = suggested_tactic
    if evidence:
        rec["evidence"] = evidence[:120]
    try:
        with _lock, open(UNKNOWN_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


def read_unknown(limit: int = 100) -> List[dict]:
    """Newest first, de-duplicated, with a count."""
    try:
        with open(UNKNOWN_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()[-2000:]
    except FileNotFoundError:
        return []
    seen: Dict[str, dict] = {}
    for line in lines:
        try:
            r = json.loads(line)
        except Exception:
            continue
        key = skeleton(r["text"])
        if key in seen:
            seen[key]["count"] += 1
            seen[key]["time"] = r["time"]
            seen[key].update({k: v for k, v in r.items() if k in ("suggested_tactic", "evidence")})
        else:
            seen[key] = {**r, "count": 1}
    return list(reversed(list(seen.values())))[:limit]
