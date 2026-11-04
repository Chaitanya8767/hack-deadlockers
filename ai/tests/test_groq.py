"""
Test Groq API connection and run a live classification on a scam utterance.
"""
import httpx, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
from config import settings

key = settings.llm.groq_api_key
url = "https://api.groq.com/openai/v1/chat/completions"
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

# Test with the actual scam classifier system prompt
from tactic_classifier import SYSTEM_PROMPT

payload = {
    "model": "qwen/qwen3.8-27b",
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": '''CONVERSATION CONTEXT:
No previous utterances.

LATEST CALLER UTTERANCE:
[EN] "Hello sir, I am Senior Inspector Rajesh Verma calling from CBI Cyber Crime Branch New Delhi. A parcel with 150 grams of cocaine has been registered in your name and a non-bailable warrant has been issued against you."

Analyze for scam tactics. Respond with JSON:'''}
    ],
    "response_format": {"type": "json_object"},
    "temperature": 0.1
}

print("Testing Groq API with qwen/qwen3.8-27b on real scam utterance...")
resp = httpx.post(url, json=payload, headers=headers, timeout=20.0)
print(f"Status: {resp.status_code}")

if resp.status_code == 200:
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    tactics = parsed.get("tactics", [])
    print(f"\nDetected {len(tactics)} tactic(s):")
    for t in tactics:
        print(f"  - {t['tactic']} ({t['confidence']:.0%}): {t['evidence'][:50]}")
        print(f"    Explanation: {t['explanation']}")
    print(f"\nGROQ API WORKING SUCCESSFULLY!")
else:
    print(f"Error: {resp.text[:400]}")
