from typing import List, Dict
from pydantic import BaseModel

class Alert(BaseModel):
    level: str  # Low | Medium | High | Critical
    message: str
    action: str
    active_tactics: List[str]

TACTIC_DISPLAY_NAMES = {
    "authority_impersonation": "authority impersonation",
    "urgency_creation": "artificial urgency",
    "fear_manipulation": "fear and intimidation",
    "otp_request": "OTP / security code requests",
    "financial_manipulation": "coerced money transfers",
    "kyc_fraud": "fake KYC verification",
    "digital_arrest": "digital arrest threats",
    "remote_access": "remote screen access requests",
    "reward_scam": "fake lottery/rewards",
    "identity_manipulation": "requests for account / card / Aadhaar details",
}

def generate_alert(level: str, active_tactics: List[str]) -> Alert:
    readable_tactics = [TACTIC_DISPLAY_NAMES.get(t, t) for t in active_tactics]

    if level == "Critical":
        if "otp_request" in active_tactics:
            tactics_desc = "requesting OTP / security codes"
        elif "identity_manipulation" in active_tactics:
            tactics_desc = "asking for your account, card or Aadhaar details"
        elif "financial_manipulation" in active_tactics:
            tactics_desc = "demanding immediate fund transfers"
        elif "remote_access" in active_tactics:
            tactics_desc = "asking to install remote screen sharing apps"
        elif readable_tactics:
            tactics_desc = "using " + ", ".join(readable_tactics[:2])
        else:
            tactics_desc = "attempting credential or fund extraction"

        message = f"Critical scam warning! The caller is {tactics_desc}. Do not share OTPs, passwords, or transfer money."
        action = "DO NOT SHARE OTP OR TRANSFER FUNDS. Hang up and verify through official channels."

    elif level == "High":
        if readable_tactics:
            tactics_desc = "using " + " and ".join(readable_tactics[:2])
        else:
            tactics_desc = "pressuring you with false claims"

        message = f"Warning! Possible scam detected. The caller is {tactics_desc}."
        action = "Do not comply with instructions. Verify caller independently."

    elif level == "Medium":
        if readable_tactics:
            tactics_desc = "is creating " + readable_tactics[0]
        else:
            tactics_desc = "is displaying unusual urgency"

        message = f"Suspicious behavior detected. The caller {tactics_desc}."
        action = "Proceed with caution. Do not share personal credentials."

    else:  # Low
        message = "No significant suspicious behavior detected."
        action = "Call appears normal."

    return Alert(
        level=level,
        message=message,
        action=action,
        active_tactics=active_tactics
    )

