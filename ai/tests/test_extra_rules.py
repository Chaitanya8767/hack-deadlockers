import os, sys, tempfile, importlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import lexicon
from rule_engine import analyze_text


def tactics(s):
    return {e.tactic for e in analyze_text(s)}


# ---- new Marathi / Hindi / English phrases ----
def test_marathi_new_phrases():
    assert "fear_manipulation" in tactics("तुमचे खाते गोठवले आहे")
    assert "reward_scam" in tactics("तुम्हाला बक्षीस लागले आहे")
    assert "reward_scam" in tactics("पार्ट टाइम जॉब आहे घरबसल्या कमवा")
    assert "remote_access" in tactics("आम्ही लिंक पाठवली आहे, लिंकवर क्लिक करा")
    assert "kyc_fraud" in tactics("केवायसी अपडेट करा")
    assert "financial_manipulation" in tactics("कृपया फोनपे वरून पैसे पाठवा")
    assert "digital_arrest" in tactics("कॉल कट करू नका आणि घरच्यांना सांगू नका")
    assert "otp_request" in tactics("मोबाईलवर आलेला सहा अंकी कोड सांगा")


def test_family_emergency_scam():
    got = tactics("तुमच्या मुलाचा अपघात झाला आहे, जामीनासाठी पैसे पाठवा")
    assert "fear_manipulation" in got and "financial_manipulation" in got


# ---- words we never listed ----
def test_spelling_variants():
    assert "otp_request" in tactics("तुमचा ओटिपी सांगा")        # short i
    assert "otp_request" in tactics("तुमचा ऑटीपी सांगा")        # ऑ spelling
    assert "otp_request" in tactics("tell me your otpp")
    assert "otp_request" in tactics("your passwrod please")
    assert "remote_access" in tactics("Install AnyDeks now")


def test_generic_ask_for_unlisted_item():
    assert "identity_manipulation" in tactics("tell me your ATM card back number")
    assert "identity_manipulation" in tactics("तुमची माहिती द्या")


# ---- safe sentences stay safe ----
def test_safe_sentences():
    for s in ["आज संध्याकाळी जेवायला येणार का", "The warranty on my phone ends soon",
              "Never share your OTP", "तुमचा ओटीपी कोणालाही सांगू नका", "दूध आणायला जा",
              "I paid my electricity bill yesterday"]:
        assert not tactics(s), s


# ---- live teaching (no restart) ----
def test_custom_rule_roundtrip(tmp_path):
    lexicon.custom_rules = lexicon.CustomRules(str(tmp_path / "rules.json"))
    import rule_engine
    rule_engine.lexicon.custom_rules = lexicon.custom_rules
    phrase = "सरकारी योजनेचा लाभ"
    assert not tactics(phrase + " मिळवा")
    lexicon.custom_rules.add(phrase, "reward_scam", 0.8)
    assert "reward_scam" in tactics(phrase + " मिळवा")
    assert lexicon.custom_rules.remove(phrase) == 1
    assert not tactics(phrase + " मिळवा")


def test_bad_rule_rejected(tmp_path):
    rules = lexicon.CustomRules(str(tmp_path / "r.json"))
    for args in [("ab", "otp_request"), ("some phrase", "not_a_tactic")]:
        try:
            rules.add(*args)
            assert False
        except ValueError:
            pass


def test_unknown_log(tmp_path, monkeypatch):
    monkeypatch.setattr(lexicon, "UNKNOWN_LOG_PATH", str(tmp_path / "u.jsonl"))
    lexicon.log_unknown("कोणतेही नवीन शब्द", "test")
    lexicon.log_unknown("कोणतेही नवीन शब्द", "test")
    items = lexicon.read_unknown()
    assert len(items) == 1 and items[0]["count"] == 2
