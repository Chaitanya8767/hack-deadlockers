import re
from typing import Tuple

MARATHI_MARKERS = {
    "आहे", "नाही", "काय", "कसा", "कशी", "करतो", "करते", "झाली", "झाले", "झाला",
    "गेला", "येतो", "येते", "माहिती", "सांगतो", "पाहिजे", "द्या", "करा", "पोलीस",
    "तुम्हाला", "तुमचा", "तुमचे", "आमचा", "माझा", "माझे", "होय", "नका", "करायचे",
    "बँक", "खाते", "पैसे", "तात्काळ", "अटक", "डिजिटल"
}

HINDI_MARKERS = {
    "है", "नहीं", "क्या", "कैसे", "करता", "करती", "हुआ", "हुई", "गया", "गई",
    "आता", "आती", "चाहिए", "दीजिए", "कीजिए", "आपको", "आपका", "आपके", "हमारा",
    "मेरा", "मेरे", "था", "थी", "रहे", "रहा", "पैसे", "खाता", "तुरंत"
}

HINGLISH_WORDS = {
    "aap", "aapka", "aapko", "karo", "kijiye", "bhejo", "paisa", "paise", "khata",
    "turant", "jaldi", "police", "arrest", "adhikari", "suno", "bolo", "batao",
    "karenge", "karega", "nahi", "nahin", "hai", "hain", "hoga", "kya", "kyu"
}

def is_devanagari(text: str) -> bool:
    """Check if the text contains Devanagari script characters."""
    return bool(re.search(r'[\u0900-\u097F]', text))

def detect_language(text: str, whisper_lang: str = "auto", whisper_prob: float = 0.9) -> Tuple[str, float]:
    """
    Detects language with script and lexical markers.
    Returns: (lang_code: 'en' | 'hi' | 'mr' | 'hinglish', confidence: float)
    """
    if not text or not text.strip():
        return (whisper_lang if whisper_lang != "auto" else "en", 0.5)

    clean_text = text.strip()
    words = [w.strip(".,!?:;\"'()[]{}") for w in clean_text.lower().split() if w.strip()]

    # Check for Devanagari script
    if is_devanagari(clean_text):
        marathi_hits = sum(1 for w in words if w in MARATHI_MARKERS)
        hindi_hits = sum(1 for w in words if w in HINDI_MARKERS)

        if marathi_hits > hindi_hits:
            return ("mr", min(0.98, 0.70 + 0.1 * marathi_hits))
        elif hindi_hits > marathi_hits:
            return ("hi", min(0.98, 0.70 + 0.1 * hindi_hits))
        else:
            # Fall back to Whisper's classification if it detected 'mr' or 'hi'
            if whisper_lang in ["mr", "hi"]:
                return (whisper_lang, whisper_prob)
            return ("hi", 0.75) # Default Devanagari to Hindi if ties

    # Latin script: check for Hinglish vs English
    latin_words = [w for w in words if re.match(r'^[a-z]+$', w)]
    if latin_words:
        hinglish_hits = sum(1 for w in latin_words if w in HINGLISH_WORDS)
        if hinglish_hits >= 2 or (len(latin_words) <= 4 and hinglish_hits >= 1):
            return ("hinglish", min(0.95, 0.65 + 0.15 * hinglish_hits))

    if whisper_lang and whisper_lang != "auto":
        return (whisper_lang, whisper_prob)
    return ("en", 0.85)

