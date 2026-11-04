# ScamChain AI — Real-Time Multilingual AI Scam-Call Detection

ScamChain AI is a free, local, open-source real-time fraud detection engine designed to protect voice calls against financial fraud, digital arrest scams, KYC phishing, and remote access attacks in English, Hindi, Marathi, and Hinglish.

---

## Architecture Overview

```
Caller Voice (remoteStream only)
       │
       ▼ [16 kHz Int16 Mono PCM Frames ~250ms via AudioWorklet]
  FastAPI WebSocket (/ws/analyze/{call_id})
       │
       ▼ [Silero VAD ONNX]
  Utterance Segmentation (~600ms silence commit, 15s max cap)
       │
       ▼ [faster-whisper (CTranslate2, int8 on CPU / float16 on CUDA)]
  Speech-To-Text & Language Detection (MR / HI / EN / Hinglish)
       │
       ▼ [N=6 Conversation Context + Pre-gating]
  Tactic Classifier (Ollama qwen2.5:7b-instruct / qwen2.5:3b-instruct)
       │
       ▼ [Stage 0 to Stage 4 Subsequence Matching]
  Sequence Analyzer (Digital Arrest / OTP / Remote Access Flows)
       │
       ▼ [Noisy-OR + Asymmetric Smoothing + Sensitive-Ask Floor]
  Risk Engine (0-100 Score, Low / Medium / High / Critical)
       │
       ▼ [Dynamic Threat-Specific Guidance]
  Alert Generator ────► Real-Time Frontend WebSocket Events
```

---

## 1. Quick Setup (Windows)

### Prerequisites
- Node.js 18+ (for WebRTC signaling)
- Python 3.10+ (Python 3.12 recommended)
- [Ollama](https://ollama.com/) (Free, local LLM runner)

### Step 1: Install & Pull Ollama Model
1. Download and install Ollama from [ollama.com](https://ollama.com).
2. Open a terminal and pull the local Qwen 2.5 instruction model:
   ```cmd
   ollama pull qwen2.5:7b-instruct
   ```
   *(For low-spec or CPU-only laptops with <8 GB RAM, use the smaller model:)*
   ```cmd
   ollama pull qwen2.5:3b-instruct
   ```

### Step 2: Set Up Python Virtual Environment
Navigate to `hack-deadlockers/ai` and set up the environment:

```cmd
cd hack-deadlockers\ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

*(Silero VAD ONNX model is already included in `ai/models/silero_vad.onnx`)*

---

## 2. Configuration (`.env`)

Copy `ai/.env.example` to `ai/.env`:

```ini
# STT Configuration: tiny, base, small, medium, large-v3-turbo
STT_MODEL_SIZE=tiny
STT_DEVICE=auto
STT_COMPUTE_TYPE=auto

# LLM Classifier Provider: ollama (default) | gemini_free | groq_free
LLM_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct

# Optional Free-Tier API fallbacks (Zero keys needed if using Ollama)
GEMINI_API_KEY=
GROQ_API_KEY=

# Optional Marathi IndicConformer
ENABLE_INDIC_CONFORMER=false
```

### CPU vs GPU Performance Expectations
- **CPU (Default):**
  - VAD: ~2ms per frame (Silero ONNX on CPU)
  - STT (`tiny`/`base`/`small`, `int8`): ~300ms–800ms per utterance
  - LLM (`qwen2.5:3b-instruct` on CPU): ~800ms–1.5s per turn
- **NVIDIA GPU (CUDA):**
  - Auto-detected (`float16`).
  - STT: ~100ms
  - LLM (`qwen2.5:7b-instruct`): ~250ms

---

## 3. Running the Project

### One-Click Launch (Both Services)
Double-click `hack-deadlockers/start.bat`, or from terminal run:
```cmd
npm run dev:all
```
This starts:
1. **WebRTC Signaling Server** on `http://localhost:3000`
2. **FastAPI AI Scam Engine** on `http://localhost:8000`

### Or Run Individually
In Window 1:
```cmd
cd hack-deadlockers
npm start
```

In Window 2:
```cmd
cd hack-deadlockers\ai
.venv\Scripts\activate
python main.py
```

Check health status:
[http://localhost:8000/health](http://localhost:8000/health)

---

## 4. How to Test & Verify

### Test A: Unit & Pipeline Tests
Run the automated test suite covering sequence escalation, noisy-OR monotonic climb, sensitive-ask floors, and synthetic conversation assertions:
```cmd
cd hack-deadlockers\ai
.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v
```

### Test B: Audio Replay Test
Simulate real-time 16 kHz PCM streaming over WebSocket:
```cmd
cd hack-deadlockers\ai
.venv\Scripts\python.exe tests/audio_replay.py ws://127.0.0.1:8000/ws/analyze/test-replay
```

### Test C: In-Browser Live Call Testing
1. Open two browser windows at `http://localhost:3000`.
2. Register ID `1001` in window 1, and `1002` in window 2.
3. Call `1002` from window 1 and accept.
4. When the call state transitions to **"connected"**, the **🛡️ AI Scam Shield** appears automatically.
5. As speech is detected on the caller's mic:
   - Live audio byte counter increments.
   - Real-time transcript appears with language code (`[EN]`, `[HI]`, `[MR]`, `[HINGLISH]`).
   - Scam tactics light up as chips with confidence percentages.
   - Dynamic risk gauge animates (Green -> Amber -> Orange -> Red).
   - If an urgent or critical ask occurs (e.g. asking for OTP or money transfer), an actionable warning banner triggers with vibration/tone alerts.

---

## 5. Marathi IndicConformer (Optional)

To enable AI4Bharat's IndicConformer for Marathi ASR:
1. Set `ENABLE_INDIC_CONFORMER=true` in `ai/.env`.
2. Install the Hugging Face `transformers` and `torchaudio` libraries:
   ```cmd
   pip install transformers torchaudio soundfile
   ```
3. The model checkpoint `ai4bharat/indicwav2vec-hindi` or IndicConformer will automatically be used when Marathi speech (`lang == "mr"`) is detected, falling back to `faster-whisper` if unavailable.

