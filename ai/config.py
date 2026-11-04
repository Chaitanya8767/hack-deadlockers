import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class STTConfig(BaseModel):
    model_size: str = Field(default=os.getenv("STT_MODEL_SIZE", "small"))
    device: str = Field(default=os.getenv("STT_DEVICE", "auto"))
    compute_type: str = Field(default=os.getenv("STT_COMPUTE_TYPE", "auto"))
    language: str = Field(default=os.getenv("STT_LANGUAGE", "auto"))
    indic_conformer_enabled: bool = Field(default=os.getenv("ENABLE_INDIC_CONFORMER", "false").lower() == "true")

class VADConfig(BaseModel):
    sample_rate: int = 16000
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 600  # Commit utterance after ~600ms silence
    max_speech_duration_s: float = 15.0  # Cap utterances at 15s

class LLMConfig(BaseModel):
    provider: str = Field(default=os.getenv("LLM_PROVIDER", "ollama"))  # ollama | gemini_free | groq_free
    ollama_url: str = Field(default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = Field(default=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"))
    gemini_api_key: str = Field(default=os.getenv("GEMINI_API_KEY", ""))
    groq_api_key: str = Field(default=os.getenv("GROQ_API_KEY", ""))
    context_turns: int = 6

class RiskConfig(BaseModel):
    # Weights for individual tactics (used in Noisy-OR)
    tactic_weights: dict = {
        "authority_impersonation": 0.50,
        "urgency_creation": 0.45,
        "fear_manipulation": 0.55,
        "otp_request": 0.85,
        "financial_manipulation": 0.80,
        "kyc_fraud": 0.70,
        "digital_arrest": 0.85,
        "remote_access": 0.80,
        "reward_scam": 0.60,
        "identity_manipulation": 0.80,
    }
    # Floor score applied when sensitive ask follows pressure
    sensitive_ask_floor: float = 65.0
    # Half-life for time decay in seconds
    decay_half_life_s: float = 120.0
    # Smoothing factors: rise fast, decay slow
    rise_alpha: float = 0.85
    fall_alpha: float = 0.20
    # Escalation multiplier for multi-stage progression
    escalation_multiplier: float = 1.35

class Config(BaseModel):
    # rules  = local word-based detection only (no LLM, no internet)  <- default
    # hybrid = rules first, LLM adds extra tactics if it answers in time
    # llm    = LLM only (old behaviour)
    detection_mode: str = os.getenv("DETECTION_MODE", "rules").lower()
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ]
    stt: STTConfig = STTConfig()
    vad: VADConfig = VADConfig()
    llm: LLMConfig = LLMConfig()
    risk: RiskConfig = RiskConfig()

settings = Config()

