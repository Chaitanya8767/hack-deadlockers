@echo off
echo ===================================================
echo Starting ScamChain WebRTC Signaling and AI Service
echo ===================================================

echo Starting Node.js WebRTC Signaling Server on port 3000...
start "Node Signaling Server (Port 3000)" cmd /k "node server.js"

echo Starting Python FastAPI AI Scam-Detection Service on port 8000...
start "FastAPI AI Engine (Port 8000)" cmd /k "cd ai && .venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000"

echo Both services launched in separate windows!
echo - WebRTC Calling Web App: http://localhost:3000
echo - AI Scam Shield Service: http://localhost:8000/health

