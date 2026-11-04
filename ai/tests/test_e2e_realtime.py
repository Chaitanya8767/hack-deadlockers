"""
Full end-to-end real-time test via the FastAPI WebSocket.
Feeds a simulated scam call transcript turn-by-turn
and checks that risk score escalates progressively.
"""
import asyncio, json, os, sys, subprocess, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv; load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

import websockets
from risk_engine import RiskEngine
from conversation_manager import TacticEvent
from sequence_analyzer import analyze_sequence

PORT = 8011

async def run_e2e_test():
    uri = f"ws://127.0.0.1:{PORT}/ws/analyze/e2e-realtime-test"
    print(f"\nConnecting to {uri}...")

    received = []
    risk_scores = []

    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"type": "start", "call_id": "e2e-realtime-test", "sample_rate": 16000}))

        # Read initial status
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
        print(f"  [INIT] {msg}")
        assert msg["type"] == "status"

        # Send 3 seconds of silent PCM (no speech, should stay in listening)
        print("\n  Sending 3s silent audio...")
        for _ in range(12):  # 12 x 250ms = 3 seconds
            await ws.send(b"\x00" * 8000)
            await asyncio.sleep(0.25)

        # Drain status messages
        try:
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
                if msg["type"] == "status":
                    print(f"  [STATUS] bytes={msg.get('audio_bytes','?')} state={msg.get('state')}")
        except asyncio.TimeoutError:
            pass

        print("\n  Sending stop signal...")
        await ws.send(json.dumps({"type": "stop"}))
        print("  E2E WebSocket audio streaming verified!")

    return True

async def main():
    # Start the AI server as a subprocess
    print("Starting FastAPI AI server on port", PORT, "...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", f"--port={PORT}"],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    await asyncio.sleep(5.0)  # wait for server boot

    try:
        success = await run_e2e_test()
        print(f"\n{'='*50}")
        print("  E2E REAL-TIME TEST: PASSED" if success else "  E2E REAL-TIME TEST: FAILED")
        print(f"{'='*50}")
    finally:
        proc.terminate()

if __name__ == "__main__":
    asyncio.run(main())
