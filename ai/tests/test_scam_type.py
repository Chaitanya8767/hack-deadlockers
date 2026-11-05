from scam_type import classify_scam_type

def event(tactic, text="Caller sentence", confidence=.95):
    return {"tactic": tactic, "confidence": confidence, "turn_id": "t1", "t": 12, "sentence": text}

def classify(events, flow=None):
    return classify_scam_type(events, [{"pattern_name": flow}] if flow else [], [{"score": 80, "level": "High"}])

def test_digital_arrest(): assert classify([event("digital_arrest", "You are under arrest")], "digital_arrest_flow")["type"] == "Digital Arrest"
def test_fake_kyc(): assert classify([event("kyc_fraud", "Your KYC expired")], "kyc_freeze_flow")["type"] == "Fake KYC"
def test_otp(): assert classify([event("otp_request", "Tell me OTP")], "otp_phish_flow")["type"] == "OTP Phishing"
def test_remote(): assert classify([event("remote_access", "Install AnyDesk")], "remote_access_flow")["type"] == "Remote Access"
def test_reward(): assert classify([event("reward_scam", "You won a job reward")], "lottery_reward_flow")["type"] == "Reward/Job Scam"
def test_utility_threat(): assert classify([event("fear_manipulation"), event("urgency_creation")])["type"] == "Utility/Bill Threat"
def test_impersonation(): assert classify([event("authority_impersonation")])["type"] == "Impersonation (generic)"
def test_benign(): assert classify([event("otp_request")], "otp_phish_flow") if False else classify_scam_type([], [], [{"score": 5, "level": "Low"}])["type"] == "Unknown/Benign"
