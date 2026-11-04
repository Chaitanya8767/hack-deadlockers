import asyncio
import os
import sys
import wave
import json
import numpy as np
import websockets

"""
Audio replay script:
Streams 16kHz Int16 mono PCM frames over the WebSocket to test the live pipeline end-to-end.
Can accept an existing WAV file, or generate a synthesized tone/audio sequence.
"""

def generate_synthetic_audio(duration_s: float = 6.0, sample_rate: int = 16000) -> bytes:
    """Generates synthetic speech-like modulated waveform for testing."""
    total_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, total_samples, endpoint=False)
    # Carrier + formant simulation + pauses
    carrier = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)
    envelope = (np.sin(2 * np.pi * 1.5 * t) > 0).astype(np.float32) # alternating speech bursts and silence
    audio = carrier * envelope * 0.4
    pcm16 = (audio * 32767).astype(np.int16).tobytes()
    return pcm16

def read_wav_file(wav_path: str, target_sr: int = 16000) -> bytes:
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    samples = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    if framerate != target_sr:
        # Simple resample
        num_target = int(len(samples) * target_sr / framerate)
        samples = np.interp(
            np.linspace(0, len(samples), num_target, endpoint=False),
            np.arange(len(samples)),
            samples
        ).astype(np.int16)
    return samples.tobytes()

async def replay_audio(ws_url: str, audio_bytes: bytes, call_id: str = "replay-test"):
    print(f"Connecting to {ws_url} for call_id='{call_id}'...")
    frame_size = 4000 * 2 # 250ms of 16kHz Int16 = 8000 bytes

    async with websockets.connect(ws_url) as ws:
        # Send start
        await ws.send(json.dumps({"type": "start", "call_id": call_id, "sample_rate": 16000}))

        async def listener():
            try:
                while True:
                    msg_text = await ws.recv()
                    data = json.loads(msg_text)
                    mtype = data.get("type")
                    if mtype == "status":
                        print(f"  [STATUS] state={data.get('state')}, bytes={data.get('audio_bytes')}")
                    elif mtype == "transcript":
                        print(f"  [TRANSCRIPT] [{data.get('lang')}] {data.get('text')}")
                    elif mtype == "tactics":
                        for t in data.get("items", []):
                            print(f"  [TACTIC] {t.get('tactic')} ({t.get('confidence')}) -> {t.get('explanation')}")
                    elif mtype == "risk":
                        print(f"  [RISK] score={data.get('score')} level={data.get('level')} trend={data.get('trend')} patterns={data.get('patterns')}")
                    elif mtype == "alert":
                        print(f"  [ALERT] ({data.get('level')}): {data.get('message')}")
            except (asyncio.CancelledError, websockets.ConnectionClosed):
                pass

        listen_task = asyncio.create_task(listener())

        # Stream frames in chunks of 250ms
        offset = 0
        total = len(audio_bytes)
        print(f"Streaming {total} bytes of audio (~{total / 32000:.1f} seconds)...")

        while offset < total:
            chunk = audio_bytes[offset:offset + frame_size]
            await ws.send(chunk)
            offset += len(chunk)
            await asyncio.sleep(0.25)

        print("Audio stream finished. Waiting 2 seconds for final processing...")
        await asyncio.sleep(2.0)

        # Send stop
        await ws.send(json.dumps({"type": "stop"}))
        listen_task.cancel()
        print("Replay completed successfully.")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8000/ws/analyze/replay-session"
    wav_file = sys.argv[2] if len(sys.argv) > 2 else None

    if wav_file and os.path.exists(wav_file):
        print(f"Loading WAV file: {wav_file}")
        audio = read_wav_file(wav_file)
    else:
        print("No WAV file specified, generating synthetic audio...")
        audio = generate_synthetic_audio(duration_s=5.0)

    asyncio.run(replay_audio(url, audio))

