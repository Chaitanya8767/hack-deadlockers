import httpx, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

from config import settings

key = settings.llm.gemini_api_key
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}"
payload = {
    "contents": [{"parts": [{"text": "You are a scam detector. Reply with valid JSON only: {\"tactics\": []}"}]}],
    "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}
}
resp = httpx.post(url, json=payload, timeout=15.0)
print("Status:", resp.status_code)
if resp.status_code == 200:
    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    print("Response:", text.strip())
    print("GEMINI API WORKING SUCCESSFULLY!")
else:
    print("Error:", resp.text[:500])
