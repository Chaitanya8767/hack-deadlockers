import os
import logging
import numpy as np
from typing import Optional, List, Generator
from pydantic import BaseModel

logger = logging.getLogger("scamchain-vad")

class Utterance(BaseModel):
    id: str
    t0: float
    t1: float
    duration: float
    samples_count: int
    audio_pcm: bytes

    class Config:
        arbitrary_types_allowed = True

class SileroVAD:
    """Silero VAD v5 runner using onnxruntime."""
    def __init__(self, model_path: Optional[str] = None):
        self.session = None
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.sr = np.array(16000, dtype=np.int64)

        default_path = os.path.join(os.path.dirname(__file__), "models", "silero_vad.onnx")
        chosen_path = model_path or default_path

        if os.path.exists(chosen_path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 1
                self.session = ort.InferenceSession(chosen_path, sess_options=opts, providers=["CPUExecutionProvider"])
                logger.info(f"Loaded Silero VAD ONNX model from {chosen_path}")
            except Exception as e:
                logger.warning(f"Failed to load Silero VAD ONNX model: {e}. Falling back to energy VAD.")
        else:
            logger.warning(f"Silero VAD model not found at {chosen_path}. Falling back to energy VAD.")

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)

    def is_speech(self, chunk: np.ndarray, threshold: float = 0.30) -> bool:
        """Run VAD on 512 float32 samples at 16kHz."""
        if len(chunk) != 512:
            return False

        rms = float(np.sqrt(np.mean(chunk**2)))
        prob = 0.0

        if self.session is not None:
            try:
                x = np.expand_dims(chunk, axis=0) # [1, 512]
                outputs = self.session.run(None, {
                    "input": x,
                    "state": self.state,
                    "sr": self.sr
                })
                prob = float(outputs[0][0][0])
                self.state = outputs[1]
            except Exception as e:
                logger.debug(f"ONNX VAD error: {e}")

        # Combined neural VAD + acoustic energy:
        # Detected as speech if neural probability >= threshold OR if RMS indicates clear voice activity
        is_speech = (prob >= threshold) or (rms >= 0.02)
        return is_speech

class AudioStreamProcessor:
    """
    Buffers streaming 16kHz Int16 audio, runs VAD,
    and yields committed Utterances upon ~600ms silence or 15s max duration.
    """
    def __init__(self, sample_rate: int = 16000, silence_ms: int = 600, max_duration_s: float = 15.0, min_speech_ms: int = 300):
        self.sample_rate = sample_rate
        self.frame_size = 512  # 32ms at 16kHz
        self.silence_samples_threshold = int((silence_ms / 1000.0) * sample_rate)
        self.max_samples = int(max_duration_s * sample_rate)
        self.min_speech_samples = int((min_speech_ms / 1000.0) * sample_rate)

        self.vad = SileroVAD()
        self.pcm_buffer = bytearray()
        self.total_samples_received = 0
        self.utterance_counter = 0

        # Speech tracking state
        self.in_speech = False
        self.speech_start_sample = 0
        self.trailing_silence_samples = 0
        self.speech_chunks: List[bytes] = []

    def reset(self):
        self.vad.reset()
        self.pcm_buffer.clear()
        self.total_samples_received = 0
        self.utterance_counter = 0
        self.in_speech = False
        self.speech_start_sample = 0
        self.trailing_silence_samples = 0
        self.speech_chunks.clear()

    def feed(self, pcm16_bytes: bytes) -> List[Utterance]:
        """
        Accepts raw Int16 PCM bytes (16kHz mono), returns list of any committed Utterances.
        """
        self.pcm_buffer.extend(pcm16_bytes)
        bytes_per_frame = self.frame_size * 2 # 1024 bytes for 512 Int16 samples
        committed: List[Utterance] = []

        while len(self.pcm_buffer) >= bytes_per_frame:
            frame_bytes = bytes(self.pcm_buffer[:bytes_per_frame])
            del self.pcm_buffer[:bytes_per_frame]

            samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            is_speech_frame = self.vad.is_speech(samples, threshold=0.5)

            if is_speech_frame:
                if not self.in_speech:
                    self.in_speech = True
                    self.speech_start_sample = self.total_samples_received
                    self.trailing_silence_samples = 0
                    self.speech_chunks.clear()

                self.speech_chunks.append(frame_bytes)
                self.trailing_silence_samples = 0
            else:
                if self.in_speech:
                    self.speech_chunks.append(frame_bytes)
                    self.trailing_silence_samples += self.frame_size

            self.total_samples_received += self.frame_size

            # Check commit conditions:
            # 1. Trailing silence >= silence_samples_threshold (~600ms)
            # 2. Maximum speech duration capped at ~15s
            if self.in_speech:
                current_speech_samples = len(self.speech_chunks) * self.frame_size
                is_silence_timeout = self.trailing_silence_samples >= self.silence_samples_threshold
                is_max_duration = current_speech_samples >= self.max_samples

                if is_silence_timeout or is_max_duration:
                    # Strip excessive trailing silence from audio
                    audio_bytes = b"".join(self.speech_chunks)
                    speech_len = len(audio_bytes) // 2

                    if speech_len >= self.min_speech_samples:
                        self.utterance_counter += 1
                        t0 = self.speech_start_sample / float(self.sample_rate)
                        t1 = (self.speech_start_sample + speech_len) / float(self.sample_rate)
                        committed.append(Utterance(
                            id=f"utt_{self.utterance_counter}",
                            t0=round(t0, 2),
                            t1=round(t1, 2),
                            duration=round(t1 - t0, 2),
                            samples_count=speech_len,
                            audio_pcm=audio_bytes
                        ))

                    # Reset tracking
                    self.in_speech = False
                    self.speech_chunks.clear()
                    self.trailing_silence_samples = 0

        return committed

