# CallApp – WebRTC in-app calling (mobile ↔ mobile)

## Run
    npm install
    npm start            # http://localhost:3000

## Test on two phones (mic needs HTTPS)
    npx cloudflared tunnel --url http://localhost:3000    # or: ngrok http 3000
Open the https URL on both phones, enter different IDs (e.g. 1001 / 1002), call each other.

## Audio access for later
See `onAudioChunk(who, blob)` and `tapStream()` in public/index.html.
