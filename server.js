// Signaling server: relays signaling between peers and proxies AI analysis to Python service
const express = require('express');
const http = require('http');
const os = require('os');
const { WebSocket, WebSocketServer } = require('ws');

const app = express();
// Proxy typed-text analysis to the Python AI service (demo fallback when the mic is noisy)
app.post('/analyze_text', express.json(), async (req, res) => {
  try {
    const r = await fetch('http://localhost:8000/analyze_text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body || {})
    });
    res.status(r.status).json(await r.json());
  } catch (e) {
    res.status(502).json({ error: 'AI server unreachable on port 8000' });
  }
});
app.post('/report/:callId', express.json(), async (req, res) => {
  try {
    const r = await fetch('http://localhost:8000/report/' + encodeURIComponent(req.params.callId), {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(req.body || {})
    });
    if (!r.ok) return res.status(r.status).json(await r.json());
    res.set('Content-Type', 'application/pdf');
    res.set('Content-Disposition', r.headers.get('content-disposition') || 'attachment; filename="scam-complaint.pdf"');
    const body = Buffer.from(await r.arrayBuffer()); res.send(body);
  } catch (e) { res.status(502).json({ error: 'AI server unreachable on port 8000' }); }
});
app.get('/config', async (_req, res) => {
  try { const r = await fetch('http://localhost:8000/config'); res.status(r.status).json(await r.json()); }
  catch (_) { res.json({ victim_side_detection: false }); }
});
app.use(express.static(require('path').join(__dirname, 'public')));
app.get('/api/ice', (_req, res) => {
  const iceServers = [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun.relay.metered.ca:80' }
  ];
  const turnUrls = (process.env.TURN_URLS || '').split(';').map(url => url.trim()).filter(Boolean);
  const turnConfigured = turnUrls.length > 0 && Boolean(process.env.TURN_USERNAME) && Boolean(process.env.TURN_CREDENTIAL);
  if (turnConfigured) {
    iceServers.push({
      urls: turnUrls,
      username: process.env.TURN_USERNAME,
      credential: process.env.TURN_CREDENTIAL
    });
  }
  res.set('Cache-Control', 'no-store').json({ iceServers });
});
const server = http.createServer(app);
const wss = new WebSocketServer({ server });
const peers = new Map(); // id -> ws

wss.on('connection', (ws, req) => {
  const url = req.url || '';

  // Proxy AI analysis WebSocket connections directly to Python FastAPI on port 8000
  if (url.startsWith('/ws/analyze/')) {
    const aiBackendUrl = 'ws://localhost:8000' + url;
    let aiWs;
    try {
      aiWs = new WebSocket(aiBackendUrl);
      const queue = [];

      ws.on('message', (data, isBinary) => {
        if (aiWs.readyState === WebSocket.OPEN) {
          aiWs.send(data, { binary: isBinary });
        } else if (aiWs.readyState === WebSocket.CONNECTING) {
          queue.push({ data, isBinary });
        }
      });

      aiWs.on('open', () => {
        while (queue.length > 0) {
          const item = queue.shift();
          aiWs.send(item.data, { binary: item.isBinary });
        }
      });

      aiWs.on('message', (data, isBinary) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(data, { binary: isBinary });
        }
      });

      aiWs.on('error', (err) => {
        console.warn('AI backend proxy error:', err.message);
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'status', state: 'error', message: 'AI backend offline' }));
        }
      });

      ws.on('close', () => {
        if (aiWs.readyState === WebSocket.OPEN || aiWs.readyState === WebSocket.CONNECTING) {
          aiWs.close();
        }
      });

      aiWs.on('close', () => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.close();
        }
      });
    } catch (err) {
      console.warn('Failed to connect to AI backend:', err.message);
    }
    return;
  }

  // Peer-to-peer WebRTC signaling logic
  ws.on('message', (raw) => {
    let m; try { m = JSON.parse(raw); } catch { return; }
    if (m.type === 'register') {
      if (peers.has(m.id) && peers.get(m.id) !== ws) return ws.send(JSON.stringify({ type: 'error', msg: 'ID already in use' }));
      ws.id = m.id; peers.set(m.id, ws);
      return ws.send(JSON.stringify({ type: 'registered', id: m.id }));
    }
    if (!ws.id) return;
    const target = peers.get(m.to);
    if (!target) return ws.send(JSON.stringify({ type: 'unavailable', to: m.to }));
    target.send(JSON.stringify({ ...m, from: ws.id })); // offer/answer/ice/reject/hangup
  });
  ws.on('close', () => { if (ws.id && peers.get(ws.id) === ws) peers.delete(ws.id); });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, '0.0.0.0', () => {
  console.log('Running on http://localhost:' + PORT);
  for (const addresses of Object.values(os.networkInterfaces())) {
    for (const address of addresses || []) {
      if (address.family === 'IPv4' && !address.internal) {
        console.log(`LAN address: http://${address.address}:${PORT}`);
      }
    }
  }
  console.log('Mobile browsers require HTTPS for microphone access; use an HTTPS tunnel for phone calls.');
  console.log(turnUrlsConfigured() ? 'Custom TURN relay configured.' : 'No TURN relay configured; calls may fail across restrictive networks.');
});

function turnUrlsConfigured() {
  return Boolean(process.env.TURN_URLS && process.env.TURN_USERNAME && process.env.TURN_CREDENTIAL);
}
