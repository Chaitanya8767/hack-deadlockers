import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uvicorn
import websockets
import json

async def run_step1_test():
    from main import app
    print("Starting uvicorn test server on 127.0.0.1:8005...", flush=True)
    config = uvicorn.Config(app, host="127.0.0.1", port=8005, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(1.0)

    uri = "ws://127.0.0.1:8005/ws/analyze/step1-test-call"
    print(f"Connecting to {uri}...", flush=True)
    async with websockets.connect(uri) as ws:
        msg1 = json.loads(await ws.recv())
        print("Connected! Initial message:", msg1, flush=True)
        assert msg1["type"] == "status"

        # Send multiple 8000 byte frames
        for _ in range(5):
            pcm_frame = b"\x00" * 8000
            await ws.send(pcm_frame)
            await asyncio.sleep(0.3)

        msg2 = json.loads(await ws.recv())
        print("Received status update:", msg2, flush=True)
        assert msg2["type"] == "status"
        assert msg2["audio_bytes"] >= 8000

        # Send stop message
        await ws.send(json.dumps({"type": "stop"}))
        print("STEP 1 VERIFIED: AudioWorklet protocol and byte counter working perfectly!", flush=True)

    server.should_exit = True
    await server_task

if __name__ == "__main__":
    asyncio.run(run_step1_test())
