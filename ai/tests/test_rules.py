import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rule_engine import analyze_text
from risk_engine import RiskEngine

def tactics(text):
    return {e.tactic for e in analyze_text(text)}

# --- must be flagged: sensitive asks ---
def test_otp_english():
    assert "otp_request" in tactics("Sir please tell me the OTP you just received")

def test_otp_spelled_out():
    assert "otp_request" in tactics("share your O T P with me")

def test_password_pin_cvv():
    assert "otp_request" in tactics("Give me your net banking password")
    assert "otp_request" in tactics("Please send your UPI PIN")
    assert "otp_request" in tactics("what is the CVV on the back of your card")

def test_account_number():
    assert "identity_manipulation" in tactics("Please tell me your account number")
    assert "identity_manipulation" in tactics("Aapka account number batao")
    assert "identity_manipulation" in tactics("share your debit card number and expiry date")

def test_aadhaar_pan():
    assert "identity_manipulation" in tactics("Send me your Aadhaar number and PAN card")

def test_hinglish_hindi_marathi():
    assert "otp_request" in tactics("Aapke phone pe OTP aaya hoga, jaldi batao")
    assert "otp_request" in tactics("आपका ओटीपी बताओ")
    assert "otp_request" in tactics("तुमचा ओटीपी सांगा")
    assert "identity_manipulation" in tactics("तुमचा खाते क्रमांक सांगा")

# --- other tactics ---
def test_other_tactics():
    assert "authority_impersonation" in tactics("I am calling from CBI crime branch")
    assert "fear_manipulation" in tactics("There is an arrest warrant against you")
    assert "digital_arrest" in tactics("You are under digital arrest, do not disconnect the call")
    assert "urgency_creation" in tactics("Do it within 10 minutes")
    assert "remote_access" in tactics("Download AnyDesk and share your screen")
    assert "kyc_fraud" in tactics("Your KYC has expired")
    assert "financial_manipulation" in tactics("Transfer the money to the RBI safe account")
    assert "reward_scam" in tactics("Congratulations you won a lottery")
    assert "digital_arrest" in tactics("डिजिटल अटक झाली आहे")

# --- must NOT be flagged ---
def test_warnings_are_safe():
    assert not tactics("Never share your OTP with anyone")
    assert not tactics("Bank will never ask for your password or PIN")
    assert not tactics("Kisi ko apna OTP mat batana")
    assert not tactics("तुमचा ओटीपी कोणालाही सांगू नका")

def test_normal_conversation_is_safe():
    for s in ["Hello, how are you today?", "Dinner at eight tonight, okay?",
              "Your food order is arriving in ten minutes", "Aaj shaam ko milte hain",
              "तुम्ही जेवलात का?"]:
        got = tactics(s)
        assert "otp_request" not in got and "identity_manipulation" not in got, (s, got)

# --- risk bar behaviour ---
def test_single_otp_ask_goes_high():
    eng = RiskEngine()
    ev = analyze_text("Please tell me your OTP", t=1.0)
    r = eng.calculate_risk("c1", ev, current_time=1.0)
    assert r.level in ("High", "Critical"), r

def test_full_scam_flow_critical():
    eng = RiskEngine()
    events, last = [], None
    for i, s in enumerate([
        "This is Inspector Sharma from Mumbai cyber crime police",
        "There is an arrest warrant against you, you are under digital arrest",
        "Do not disconnect, you must act urgently within 10 minutes",
        "Tell me your OTP and account number"]):
        events += analyze_text(s, t=float(i * 5))
        last = eng.calculate_risk("c2", events, current_time=float(i * 5))
    assert last.level == "Critical", last

def test_benign_stays_low():
    eng = RiskEngine()
    r = eng.calculate_risk("c3", analyze_text("Hello, your pizza is on the way", t=1.0), current_time=1.0)
    assert r.level == "Low"
