import io
import re
import hashlib
import logging
import numpy as np
from typing import Optional, Tuple, Dict
from pydantic import BaseModel

from config import settings
from language_detector import detect_language

logger = logging.getLogger("scamchain-stt")

class TranscriptResult(BaseModel):
    id: str
    text: str
    lang: str
    t0: float
    t1: float
    confidence: float
    final: bool = True

class SpeechToTextEngine:
    """
    Local STT using faster-whisper (CTranslate2) with deduplication,
    automatic CPU/CUDA detection, and language ID heuristic.
    """
    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.compute_type = "int8"
        self.committed_until: Dict[str, float] = {}  # call_id -> timestamp
        self.recent_hashes: Dict[str, set] = {}      # call_id -> set of recent hashes

        self._initialize_model()

    def _initialize_model(self):
        try:
            import ctranslate2
            cuda_available = ctranslate2.get_cuda_device_count() > 0
            if settings.stt.device == "cuda" or (settings.stt.device == "auto" and cuda_available):
                self.device = "cuda"
                self.compute_type = "float16" if settings.stt.compute_type == "auto" else settings.stt.compute_type
            else:
                self.device = "cpu"
                self.compute_type = "int8" if settings.stt.compute_type == "auto" else settings.stt.compute_type

            from faster_whisper import WhisperModel
            model_size = settings.stt.model_size
            logger.info(f"Loading faster-whisper model '{model_size}' on {self.device} ({self.compute_type})...")
            self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
            logger.info("faster-whisper model loaded successfully!")
        except Exception as e:
            logger.exception(f"Failed to load faster-whisper model: {e}")
            self.model = None

    def reset_call(self, call_id: str):
        self.committed_until.pop(call_id, None)
        self.recent_hashes.pop(call_id, None)

    def _normalize_text_hash(self, text: str) -> str:
        clean = re.sub(r'[\W_]+', '', text.lower().strip())
        return hashlib.md5(clean.encode('utf-8')).hexdigest()

    def transcribe(self, call_id: str, pcm_bytes: bytes, t0: float, t1: float, utt_id: str) -> Optional[TranscriptResult]:
        """
        Transcribes raw 16kHz Int16 mono PCM bytes.
        """
        if not self.model or len(pcm_bytes) == 0:
            return None

        # Check committed timestamp
        call_committed = self.committed_until.get(call_id, 0.0)
        if t1 <= call_committed:
            logger.debug(f"Dropping utterance ending before committed timestamp: {t1} <= {call_committed}")
            return None

        # Convert Int16 PCM to float32 numpy array normalized to [-1.0, 1.0]
        audio_data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            # We do NOT pass an initial_prompt with scam words to avoid biasing results
            segments, info = self.model.transcribe(
                audio_data,
                beam_size=3,
                word_timestamps=False,
                vad_filter=False, # We already performed VAD segmentation
                language=None if settings.stt.language == "auto" else settings.stt.language
            )

            text_parts = []
            for seg in segments:
                text_parts.append(seg.text.strip())

            full_text = " ".join(text_parts).strip()
            if not full_text:
                return None

            # Deduplication: normalized-text hash check
            text_hash = self._normalize_text_hash(full_text)
            call_hashes = self.recent_hashes.setdefault(call_id, set())
            if text_hash in call_hashes:
                logger.debug(f"Dropping duplicate text utterance: {full_text}")
                return None

            call_hashes.add(text_hash)
            if len(call_hashes) > 100:
                call_hashes.clear()

            self.committed_until[call_id] = t1

            # Language identification with script heuristic
            detected_lang, lang_conf = detect_language(
                full_text,
                whisper_lang=info.language,
                whisper_prob=info.language_probability
            )

            # Optional Marathi IndicConformer fallback check
            if detected_lang == "mr" and settings.stt.indic_conformer_enabled:
                # IndicConformer hook (if enabled in config)
                logger.info("IndicConformer enabled for Marathi utterance")

            return TranscriptResult(
                id=utt_id,
                text=full_text,
                lang=detected_lang,
                t0=round(t0, 2),
                t1=round(t1, 2),
                confidence=round(lang_conf, 2),
                final=True
            )

        except Exception as e:
            logger.exception(f"Error during transcription: {e}")
            return None

