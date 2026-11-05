"""
Standalone regex- and rule-based tactic classifier for ScamChain AI.
Provides fast (<5ms), offline, deterministic classification of social-engineering
scam tactics in voice-call transcripts with zero external API calls.

Supports English, Hindi, Marathi, and Hinglish.
Uses Noisy-OR combining of per-pattern weights, anti-scam negation handling,
and intelligent benign context vetoes (delivery PIN, branch KYC, personal emergencies).
"""

import logging
import math
import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Set

from conversation_manager import Turn, TacticEvent

logger = logging.getLogger("scamchain-regex-classifier")

VALID_TACTICS = {
    "authority_impersonation",
    "urgency_creation",
    "fear_manipulation",
    "otp_request",
    "financial_manipulation",
    "kyc_fraud",
    "digital_arrest",
    "remote_access",
    "reward_scam",
    "identity_manipulation",
}

# Pre-gate patterns preserved for backward compatibility
PRE_GATE_PATTERNS = [
    # English / Hinglish
    r"\b(cbi|police|customs|trai|fedex|courier|rbi|cyber\s*crime|inspector|officer|narcotics)\b",
    r"\b(arrest|fir|court|warrant|illegal|drugs|passport|blocked|suspended|penalt(y|ies))\b",
    r"\b(digital\s*arrest|stay\s*on\s*call|camera\s*on|don'?t\s*disconnect)\b",
    r"\b(otp|mpin|pin|verification\s*code|password|cvv)\b",
    r"\b(kyc|aadhaar|aadhar|pan\s*card|update|expire)\b",
    r"\b(urgent|immediately|right\s*now|within\s*\d+\s*(mins?|minutes?|hours?)|turant|jaldi)\b",
    r"\b(anydesk|teamviewer|rustdesk|quicksupport|screen\s*share|apk|download|app)\b",
    r"\b(transfer|safe\s*account|security\s*deposit|refund|lottery|reward|cashback|prize)\b",
    # Devanagari (Hindi / Marathi)
    r"(पोलीस|अधिकारी|सीबीआय|कस्टम|गुन्हा|अटक|वारंट|एफआयआर|बेकायदेशीर|डिजिटल\s*अटक)",
    r"(ओटीपी|खाते|बँक|पैसे|ट्रान्सफर|सुरक्षित\s*खाते|केवायसी|आधार|पॅन|अपडेट|बंद\s*होईल)",
    r"(तात्काळ|त्वरित|लवकर|लगेच|सध्या|तुरंत|जल्दी|अभी|धमकी|शिक्षा|कोर्ट)",
]

PRE_GATE_REGEX = re.compile("|".join(PRE_GATE_PATTERNS), re.IGNORECASE)

# Fast heuristic baseline dictionary for backwards compatibility
FAST_TACTIC_PATTERNS = {
    "authority_impersonation": (r"\b(?:calling|speaking|transferred|officer|inspector|branch|unit).{0,35}\b(?:cbi|police|customs|trai|rbi|cyber\s*crime|bank|court|fedex)\b|\b(?:cbi|police|customs|trai|rbi|cyber\s*crime|bank|court)\s+(?:officer|inspector|branch|department|unit|headquarters)\b", 0.88),
    "urgency_creation": (r"\b(?:urgent(?:ly)?|immediately|right now|within\s+\d+\s*(?:minutes?|mins?|hours?)|asap|turant|jaldi|abhi|tatkal|laghech|लगेच|तुरंत|अभी|तत्काल)\b", 0.86),
    "fear_manipulation": (r"\b(?:arrest|warrant|jail|prison|narcotics|illegal drugs|case filed|file an? fir|detained|account (?:will be )?(?:frozen|blocked)|account freeze|seized|penalty|अटक|वारंट|जेल|गिरफ्तार|खाता बंद)\b", 0.90),
    "digital_arrest": (r"\b(?:digital\s*arrest|digital custody|stay on (?:the )?(?:phone|call)|do not disconnect|don't disconnect|camera on|keep your camera on|do not tell (?:anyone|your family)|don't tell (?:anyone|your family))\b|किसी को मत बत|कॉल मत काट|डिजिटल\s*अटक", 0.94),
    "otp_request": (r"\b(?:share|tell|read|send|give|provide|confirm|बताओ|बताना|भेजो).{0,32}\b(?:otp|one.time password|verification code|pin|cvv)\b|\b(?:otp|verification code|pin|cvv).{0,32}\b(?:share|tell|read|send|give|provide|बताओ|बताना|भेजो)\b", 0.97),
    "financial_manipulation": (r"\b(?:safe account|security account|verification fee|clearance fee|security deposit|transfer.{0,35}(?:money|funds|rupees|savings|account)|send.{0,25}(?:money|funds|rupees)|pay.{0,25}(?:fee|deposit|charge))\b", 0.92),
    "kyc_fraud": (r"\b(?:kyc.{0,25}(?:update|expired|expire|suspend|block|verify|complete)|(?:update|verify|complete).{0,25}kyc)\b", 0.86),
    "remote_access": (r"\b(?:install|download|open|use|share your screen|screen share).{0,30}\b(?:anydesk|teamviewer|rustdesk|quicksupport|remote access|screen sharing|apk)\b|\b(?:anydesk|teamviewer|rustdesk|quicksupport)\b", 0.95),
    "reward_scam": (r"\b(?:you have won|you won|lottery prize|claim your prize|cashback.{0,25}(?:claim|fee|transfer)|reward.{0,25}(?:claim|fee|transfer))\b", 0.88),
    "identity_manipulation": (r"\b(?:share|send|tell|provide|read|confirm).{0,32}\b(?:aadhaar|aadhar|pan card|card number|date of birth)\b", 0.88),
}

# Devanagari digit transliteration
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Phonetic & Whisper mis-transcription standardizations
PHONETIC_NORMALIZATIONS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:oh\s+tee\s+pee|o\s+t\s+p|o\.t\.p|o-t-p)\b", re.IGNORECASE), "otp"),
    (re.compile(r"(?:ओ\s*टी\s*पी)"), "ओटीपी"),
    (re.compile(r"\b(?:any\s+desk|any-desk|ani\s*desk)\b", re.IGNORECASE), "anydesk"),
    (re.compile(r"(?:एनी\s*डेस्क|अ‍ॅनी\s*डेस्क)"), "anydesk"),
    (re.compile(r"\b(?:team\s+viewer|team-viewer)\b", re.IGNORECASE), "teamviewer"),
    (re.compile(r"(?:टीम\s*व्यूअर)"), "teamviewer"),
    (re.compile(r"\b(?:rust\s+desk|rust-desk)\b", re.IGNORECASE), "rustdesk"),
    (re.compile(r"\b(?:quick\s+support|quick-support)\b", re.IGNORECASE), "quicksupport"),
    (re.compile(r"\b(?:c\s*b\s*i)\b", re.IGNORECASE), "cbi"),
    (re.compile(r"(?:सी\s*बी\s*आई|सीबीआय)"), "cbi"),
    (re.compile(r"\b(?:r\s*b\s*i)\b", re.IGNORECASE), "rbi"),
    (re.compile(r"(?:आर\s*बी\s*आई|आरबीआय)"), "rbi"),
    (re.compile(r"\b(?:f\s*i\s*r)\b", re.IGNORECASE), "fir"),
    (re.compile(r"(?:एफ\s*आई\s*आर|एफआयआर)"), "fir"),
    (re.compile(r"\b(?:k\s*y\s*c)\b", re.IGNORECASE), "kyc"),
    (re.compile(r"(?:के\s*वाई\s*सी|केवायसी|केवाईसी)"), "kyc"),
    (re.compile(r"\b(?:pan\s+card|pan-card)\b", re.IGNORECASE), "pancard"),
    (re.compile(r"(?:पॅन\s*कार्ड|पैन\s*कार्ड)"), "pancard"),
    (re.compile(r"\b(?:aadhar|adhar)\b", re.IGNORECASE), "aadhaar"),
    (re.compile(r"\b(?:screen\s+share|screen-share|screenshare)\b", re.IGNORECASE), "screenshare"),
]

def normalize_text(text: str) -> str:
    """Normalize text for consistent regex evaluation while keeping Unicode words."""
    if not text:
        return ""
    text = text.translate(DEVANAGARI_DIGITS)
    text = unicodedata.normalize("NFKC", text)
    for pattern, repl in PHONETIC_NORMALIZATIONS:
        text = pattern.sub(repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Negation & Anti-scam advisory patterns (e.g. "bank will never ask for OTP")
NEGATION_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(?:never|do\s+not|don'?t|should\s+not|shouldn'?t|must\s+not)\s+(?:share|disclose|give|tell|enter|send|provide)\b.{0,30}\b(?:otp|password|pin|mpin|cvv|credentials?|card\s+details?|code)\b", re.IGNORECASE),
    re.compile(r"\b(?:bank|police|rbi|cbi|official|customer\s+care|state\s+bank)\s+(?:will\s+never|never|does\s+not|doesn'?t|do\s+not|don'?t)\s+(?:ask|demand|request|call\s+for)\b.{0,35}\b(?:otp|pin|password|cvv|transfer|funds|money)\b", re.IGNORECASE),
    re.compile(r"\b(?:beware\s+of|caution\s+against|fraud\s+alert|alert\s+regarding|stay\s+alert\s+against|warning)\b.{0,35}\b(?:fraud|scam|cheat|fake|callers?|fraudsters?)\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\s+(?:click|download|install|open\s+any)\b.{0,25}\b(?:link|apk|anydesk|unknown)\b", re.IGNORECASE),
    re.compile(r"\bnever\s+transfer\s+(?:money|funds)\b", re.IGNORECASE),
    re.compile(r"\b(?:share\s+mat|mat\s+share|mat\s+dena|mat\s+batao|na\s+kare)\b.{0,25}\b(?:otp|pin|password)\b", re.IGNORECASE),
    re.compile(r"\b(?:otp|pin|password)\b.{0,25}\b(?:share\s+mat|mat\s+share|mat\s+dena|mat\s+batao|na\s+kare)\b", re.IGNORECASE),
    re.compile(r"\b(?:kisi\s+ko\s*(?:bhi)?|kisi\s+se|kisi\s+ke\s+saath)\b.{0,25}\b(?:otp|pin|password)\b.{0,25}\b(?:share|batao|dena|karna)\b", re.IGNORECASE),
    re.compile(r"(?:कभी\s*भी|किसी\s*को\s*भी|किसी\s*के\s*साथ|किसी\s*को\s*भी\s*अपना)\s+.{0,25}(?:ओटीपी|पिन|पासवर्ड|सीवीवी)\s+(?:शेयर\s*ना\s*करें|मत\s*देना|मत\s*बताओ|मत\s*दीजिए|साझा\s*ना\s*करें|नहीं\s*मांगता|नहीं\s*पूछता)"),
    re.compile(r"(?:बैंक|पुलिस|आरबीआई)\s+(?:कभी\s*भी\s*)?(?:ओटीपी|पासवर्ड|पैसे)\s+(?:नहीं\s*मांगता|नहीं\s*पूछता|कॉल\s*नहीं\s*करता)"),
    re.compile(r"(?:ओटीपी|पिन|पासवर्ड)\s+(?:किसी\s*को\s*मत\s*दो|मत\s*बताना|शेयर\s*मत\s*करो|शेयर\s*ना\s*करें)"),
    re.compile(r"(?:कोणालाही|कुणालाही|कधीही).{0,35}(?:ओटीपी|पिन|पासवर्ड|सीव्हीव्ही|code)\s+(?:शेअर\s*करू\s*नका|सांगू\s*नका|देऊ\s*नका)"),
    re.compile(r"(?:बँक|पोलीस|आरबीआय|bank).{0,60}(?:कधीही|कधीच|never).{0,60}(?:मागत\s*नाही|विचारत\s*नाही)"),
    re.compile(r"(?:खबरदारीची\s*सूचना|सतर्कता\s*सूचना|सावधान\s*रहा)"),
    re.compile(r"\b(?:bank|police)\s+(?:kabhi\s+bhi|kabhi\s+nahi)\s+(?:otp|pin|password)\s+(?:nahi\s+mangti|nahi\s+mangta|nahi\s+puchta)\b", re.IGNORECASE),
]

# Patterns for each of the 10 tactics: List of (regex, base_weight, explanation)
TACTIC_RULES: Dict[str, List[Tuple[re.Pattern, float, str]]] = {
    "authority_impersonation": [
        (re.compile(r"\b(?:calling|speaking|transferred|this\s+is|here\s+is)\b.{0,35}\b(?:cbi|police|customs|trai|rbi|cyber\s*crime|supreme\s*court|high\s*court|fedex|telecom|narcotics|microsoft\s+technical|microsoft\s+security|mahavitaran|income\s*tax\s*(?:department|dept)?)\b", re.IGNORECASE), 0.90, "Caller claims affiliation with official authority or institution"),
        (re.compile(r"\b(?:i\s+am|myself|this\s+is)\s+(?:officer|inspector|head\s*constable|commissioner|manager|senior\s+inspector|agent)\s+[a-z]+", re.IGNORECASE), 0.85, "Caller introduces self with official title and name"),
        (re.compile(r"\b(?:cbi|police|customs|trai|rbi|cyber\s*crime|narcotics|telecom|mahavitaran|income\s*tax)\s+(?:department|headquarters|crime\s*branch|head\s*office|cell|official|inspector|officer|manager|dept)\b", re.IGNORECASE), 0.88, "Reference to law enforcement, government agency, or utility board"),
        (re.compile(r"\b(?:manager|officer)\s+(?:calling|speaking)\s+from\s+(?:the\s+)?(?:state\s+bank|hdfc|icici|axis|punjab\s+national|rbi|bank)\b", re.IGNORECASE), 0.86, "Caller claims to be a bank official or manager"),
        (re.compile(r"\btransferred\s+to\s+(?:cbi|crime\s*branch|police|customs)\b", re.IGNORECASE), 0.89, "Claims case was transferred to law enforcement branch"),
        # Hinglish authority patterns
        (re.compile(r"\b(?:main|hum)\b.{0,35}\b(?:bank|head\s*office|branch|police|cbi|customs|crime\s*branch|court|rbi|trai|telecom)\b.{0,35}\b(?:se|ka|ki)\s*(?:manager|officer|inspector|head)\s*(?:bol\s*raha|baat\s*kar\s*raha)\b", re.IGNORECASE), 0.90, "Hinglish official identification"),
        (re.compile(r"\b(?:manager|officer|inspector)\s*(?:bol\s*raha|baat\s*kar\s*raha)\s*(?:hu|hai)\b", re.IGNORECASE), 0.86, "Hinglish rank intro"),
        (re.compile(r"\b(?:delhi|mumbai)\s*airport\s*customs\b", re.IGNORECASE), 0.90, "Airport customs authority reference"),
        # Hindi
        (re.compile(r"(?:मैं|हम)\s+.{0,35}(?:सीबीआई|cbi|पुलिस|कस्टम|क्राइम\s*ब्रांच|मुख्यालय|आरबीआई|rbi|दूरसंचार|बैंक|आयकर).{0,25}(?:से\s+)?(?:इंस्पेक्टर|अधिकारी|मैनेजर|बोल\s*रहा|बात\s*कर\s*रहा)"), 0.92, "Caller claims official identity in Hindi"),
        (re.compile(r"(?:सीबीआई|cbi|पुलिस|कस्टम|क्राइम\s*ब्रांच|मुख्यालय|स्टेट\s*बैंक|आरबीआई|rbi|आयकर\s*विभाग)\s+(?:से\s+)?(?:सीनियर\s+)?(?:इंस्पेक्टर|मैनेजर|अधिकारी)"), 0.90, "Official agency or rank mentioned in Hindi"),
        # Marathi
        (re.compile(r"(?:मी|आम्ही)\s+.{0,35}(?:पोलीस|सायबर\s*सेल|मुख्यालय|कस्टम|महावितरण|वीज\s*विभाग|बँक).{0,25}(?:बोलत\s*आहे|बोलतोय)"), 0.92, "Caller claims official authority in Marathi"),
        (re.compile(r"(?:पोलीस\s*मुख्यालय|सायबर\s*सेल|महावितरण|वीज\s*विभाग|गुन्हे\s*शाखा)\s*(?:तून|कडून|तर्फे)?"), 0.88, "Official government or utility entity reference in Marathi"),
    ],

    "urgency_creation": [
        (re.compile(r"\b(?:immediately|right\s*now|urgently?|within\s+\d+\s*(?:minutes?|mins?|hours?|hrs?)|without\s+delay|at\s+once|asap)\b", re.IGNORECASE), 0.88, "High-urgency deadline imposed"),
        (re.compile(r"\b(?:turant|jaldi|abhi\s+ke\s+abhi|tatkal|fatafat)\b", re.IGNORECASE), 0.87, "Urgency coercion in Hinglish"),
        (re.compile(r"\b(?:today\s+itself|by\s+tonight|before\s+\d+\s*(?:pm|am))\b", re.IGNORECASE), 0.70, "Same-day deadline pressure"),
        (re.compile(r"\b(?:warna|otherwise)\s+.{0,25}\b(?:block|disconnect|freeze|arrest|action)\b", re.IGNORECASE), 0.86, "Conditional consequence forcing immediate compliance"),
        # Hindi
        (re.compile(r"(?:तुरंत|जल्दी|अभी|तत्काल|फटाफट|\d+\s*मिनट\s*के\s*अंदर|आज\s*ही|आज\s*रात|\d+\s*बजे\s*तक|आज\s*शाम\s*तक)"), 0.88, "Urgency cues in Hindi"),
        # Marathi
        (re.compile(r"(?:लगेच|तात्काळ|त्वरित|लवकर|आज\s*रात्री|\d+\s*मिनिटांत|आजच)"), 0.88, "Urgency cues in Marathi"),
    ],

    "fear_manipulation": [
        (re.compile(r"\b(?:arrest\s*warrant|non-bailable|warrant|ndps\s*act|illegal\s*contraband|narcotics|illegal\s*drugs|case\s*filed|file\s*(?:an?\s*)?fir|detained|jail|prison|go\s+to\s+jail)\b", re.IGNORECASE), 0.93, "Threat of arrest, jail, criminal case, or narcotics contraband"),
        (re.compile(r"\b(?:money\s*laundering|hawala|trojan|malicious\s*(?:activity|software)|cyber\s*crime\s*complaint|illegal\s*activities|police\s*(?:complaint|case)|case\s*se\s*bachne)\b", re.IGNORECASE), 0.89, "Accusations of illegal activities, money laundering, or complaints"),
        (re.compile(r"\b(?:account|sim|number|service|line)\b.{0,35}\b(?:will\s+be\s+)?(?:blocked|suspended|frozen|disconnected|terminated|permanent\s+block)\b", re.IGNORECASE), 0.88, "Threat of account or telecom line suspension"),
        (re.compile(r"\b(?:will\s+be\s+blocked|will\s+be\s+suspended|will\s+be\s+frozen|will\s+be\s+disconnected|will\s+be\s+terminated)\b", re.IGNORECASE), 0.88, "Threat of impending service or account termination"),
        (re.compile(r"\b(?:account\s+freeze|account\s+block|freeze\s+ho\s+jayega|block\s+ho\s+jayega|block\s+kar\s+diya\s+jayega)\b", re.IGNORECASE), 0.88, "Threat of account freeze or block in Hinglish"),
        (re.compile(r"\b(?:permanent\s*block|block\s*kar\s*(?:denge|diya|dega)|line\s*disconnect\s*ho\s*jayegi|arrest\s*ho\s*(?:jaoge|jayenge)|police\s*arrest\s*karegi)\b", re.IGNORECASE), 0.88, "Threat of immediate line termination or arrest in Hinglish"),
        (re.compile(r"\b(?:penalt(?:y|ies)|heavy\s*fine|legal\s*action|police\s*action|prosecution|chargesheet)\b", re.IGNORECASE), 0.85, "Threat of severe financial penalty or prosecution"),
        # Hindi
        (re.compile(r"(?:गिरफ्तारी|अरेस्ट|वारंट|जेल|अवैध|तस्करी|मनी\s*लॉन्ड्रिंग|हवाला|गैरकानूनी|मुकदमा|एफआईआर|fir|खाता\s*ब्लॉक|खाता\s*फ्रीज|बंद\s*कर\s*दिया\s*जाएगा|हमेशा\s*के\s*लिए\s*ब्लॉक|ब्लॉक\s*कर\s*दिया|बिजली\s*काट|जुर्माना|दण्ड|सजा|कार्रवाई)"), 0.92, "Legal threat, arrest warrant, or account freeze in Hindi"),
        # Marathi
        (re.compile(r"(?:गुन्हा\s*दाखल|अटक\s*केली\s*जाईल|अटक|वारंट|तस्करी|हवाला|बेकायदेशीर|औषध\s*तस्करी|कायमची\s*खंडित|वीज\s*कापली|कारवाई)"), 0.92, "Legal threats or utility cutoff threats in Marathi"),
    ],

    "digital_arrest": [
        (re.compile(r"\b(?:digital\s*arrest|digital\s*custody)\b", re.IGNORECASE), 0.98, "Explicit mention of digital arrest or custody"),
        (re.compile(r"\b(?:stay\s+on\s+(?:the\s+)?(?:call|phone|line)|do\s*n'?t\s*disconnect|do\s*not\s*disconnect|keep\s+(?:your\s+)?camera\s+on|turn\s+(?:your\s+)?camera\s+on)\b", re.IGNORECASE), 0.92, "Command to remain on call and keep camera active"),
        (re.compile(r"\b(?:do\s*not\s*tell|don'?t\s*tell|do\s*not\s*inform|don'?t\s*inform)\s+(?:anyone|your\s+family|family|friends|police)\b", re.IGNORECASE), 0.94, "Prohibiting contacting family, friends, or outside authorities"),
        (re.compile(r"\b(?:lock\s+your\s+door|close\s+the\s+door|isolate\s+yourself|sit\s+alone)\b", re.IGNORECASE), 0.91, "Instructing victim to isolate behind closed doors"),
        (re.compile(r"\b(?:non-bailable\s+arrest\s+warrant|arrest\s+warrant\s+has\s+been\s+issued)\b", re.IGNORECASE), 0.85, "Arrest warrant threat associated with digital detention"),
        # Digital arrest room, skype/video call confinement, digital warrant
        (re.compile(r"\b(?:arrest\s*room|custody\s*room|digital\s*custody\s*room)\b", re.IGNORECASE), 0.94, "Instructing victim into digital arrest custody room"),
        (re.compile(r"\b(?:digital|e-?|online)\s*warrant\b", re.IGNORECASE), 0.94, "Fake digital arrest warrant"),
        (re.compile(r"\b(?:skype|zoom|whatsapp|video)\s*(?:call|pe|par)\b.{0,45}\b(?:arrest|warrant|custody|cbi|police|investigation|officer|court)\b|\b(?:arrest|warrant|custody|cbi|police)\b.{0,45}\b(?:skype|zoom|whatsapp|video)\s*(?:call|pe|par)\b", re.IGNORECASE), 0.92, "Video call custody or digital arrest hearing"),
        (re.compile(r"\b(?:camera\s*on\s*(?:rakho|rakhna)|call\s*mat\s*(?:kaatna|katna|disconnect)|phone\s*cut\s*mat\s*(?:karna|karo)|kisi\s*ko\s*(?:mat\s*batana|bataya))\b", re.IGNORECASE), 0.93, "Confinement and isolation instructions in Hinglish"),
        # Hindi
        (re.compile(r"(?:डिजिटल\s*अरेस्ट|डिजिटल\s*कस्टडी|डिजिटल\s*वारंट)"), 0.98, "Explicit digital arrest in Hindi"),
        (re.compile(r"(?:कॉल\s*(?:चालू|ऑन)\s*रखें|कॉल\s*मत\s*काटना|कॉल\s*डिस्कनेक्ट\s*मत|कैमरा\s*ऑन\s*रखो|दरवाजा\s*बंद\s*करें|किसी\s*को\s*(?:भी\s*)?(?:मत\s*बताना|जानकारी\s*नहीं))"), 0.93, "Confinement and secrecy instructions in Hindi"),
        # Marathi
        (re.compile(r"(?:डिजिटल\s*अरेस्ट|डिजिटल\s*वारंट)"), 0.98, "Explicit digital arrest in Marathi"),
        (re.compile(r"(?:कॉल\s*डिस्कनेक्ट\s*करू\s*नका|खोलीतून\s*बाहेर\s*पडू\s*नका|कॉल\s*सुरू\s*ठेवा|(?:तपास|अटक|कॉल|कारवाई).{0,25}कोणालाही\s*सांगू\s*नका)"), 0.94, "Confinement and isolation instructions in Marathi"),
    ],

    "otp_request": [
        (re.compile(r"\b(?:share|tell|read|send|give|provide|confirm|enter|verify)\b.{0,35}\b(?:otp|one[- ]time\s*password|verification\s*code|mpin|pin|cvv|6[- ]digit\s*code)\b", re.IGNORECASE), 0.97, "Demand to share or provide confidential OTP/PIN"),
        (re.compile(r"\b(?:otp|verification\s*code|6[- ]digit\s*code)\b.{0,35}\b(?:share|tell|read|send|give|provide|batao|bataiye|batayein|batado|bhejo|aaya\s+hoga|aaya\s+hai)\b", re.IGNORECASE), 0.96, "Prompting for incoming OTP or verification code"),
        (re.compile(r"\b(?:share|tell|send|give|batao|bataiye|batayein|batado|bhejo|bhej|de\s*do|bolie)\b.{0,35}\b(?:otp|code|pin|password|mpin)\b", re.IGNORECASE), 0.96, "Hinglish verb demanding OTP or secret code"),
        (re.compile(r"\b(?:phone|mobile)\s*(?:par|pe)\s*(?:aaya\s*hua|aaya\s*hai|aaya\s*hoga)\s*(?:otp|code|pin)\b", re.IGNORECASE), 0.97, "Hinglish notification of incoming OTP to victim"),
        (re.compile(r"\baapke\s+phone\s+pe\s+(?:\d+[- ]digit\s+)?otp\s+aaya\s+hoga", re.IGNORECASE), 0.97, "Hinglish pattern stating OTP has arrived on phone"),
        # Aadhaar / Card / Account update + OTP ask
        (re.compile(r"\b(?:aadhaar|aadhar|biometric|card|account)\b.{0,35}\b(?:update|verification|activate|link)\b.{0,35}\b(?:otp|code|pin)\b", re.IGNORECASE), 0.95, "Harvesting OTP under pretext of Aadhaar or card update"),
        # Hindi
        (re.compile(r"(?:ओटीपी|otp|पिन|कोड|\d+\s*अंकों\s*का\s*(?:ओटीपी|otp)|वेरिफिकेशन\s*कोड)\s*.{0,35}(?:बताओ|दीजिए|भेजो|शेयर\s*करो|आया\s*होगा|आया\s*है|बताएं)"), 0.97, "Request for OTP or PIN in Hindi"),
        (re.compile(r"(?:बताओ|दीजिए|भेजो|शेयर\s*करो|बताएं)\s*.{0,35}(?:ओटीपी|otp|पिन|कोड)"), 0.96, "Request to send OTP in Hindi"),
        (re.compile(r"(?:आधार|बायोमेट्रिक|कार्ड|खाता).{0,30}(?:अपडेट|सक्रिय|वेरिफिकेशन).{0,30}(?:ओटीपी|otp|पिन|कोड)"), 0.95, "Aadhaar/card update OTP harvesting in Hindi"),
        # Marathi
        (re.compile(r"(?:ओटीपी|otp|पिन|सहा\s*अंकी\s*कोड)\s*.{0,35}(?:सांगा|पाठवा|द्या|शेअर\s*करा)"), 0.97, "Request for OTP or PIN in Marathi"),
        (re.compile(r"(?:आधार|बायोमेट्रिक|कार्ड|खाते).{0,30}(?:अपडेट|पडताळणी).{0,30}(?:ओटीपी|otp|पिन)"), 0.95, "Aadhaar/card update OTP harvesting in Marathi"),
    ],

    "financial_manipulation": [
        (re.compile(r"\b(?:rbi\s*safe\s*(?:verification\s*)?account|safe\s*account|security\s*account|government\s*account)\b", re.IGNORECASE), 0.95, "Directing funds to bogus government or safe account"),
        (re.compile(r"\b(?:transfer|send|deposit|pay)\b.{0,35}\b(?:funds?|money|rupees|amount|deposit|fees?|charges?|savings)\b", re.IGNORECASE), 0.88, "Instruction to transfer or deposit funds"),
        (re.compile(r"\b(?:security\s*deposit|verification\s*fee|clearance\s*fee|bail\s*deposit|refundable\s*amount|claim\s*fee|processing\s*fee|release\s*charge)\b", re.IGNORECASE), 0.91, "Requesting upfront fee or security deposit"),
        (re.compile(r"\b(?:customs?|courier|parcel|delivery|package)\b.{0,35}\b(?:clearance|release|processing|hold)\b.{0,35}\b(?:fee|charges?|transfer|pay|amount)\b", re.IGNORECASE), 0.92, "Demanding customs clearance or release fee for package"),
        # Hindi
        (re.compile(r"(?:सरकारी\s*खाते|सुरक्षित\s*खाते|आरबीआई\s*खाते|rbi\s*खाते).{0,25}(?:पैसे|रुपये|ट्रांसफर)"), 0.95, "Directing money transfer to safe/official account in Hindi"),
        (re.compile(r"(?:पैसे|रुपये|डिपॉजिट|फीस|रकम)\s*.{0,25}(?:ट्रांसफर\s*कीजिए|ट्रांसफर\s*करो|भेजें|जमा\s*करो|भुगतान\s*करें)"), 0.90, "Demand for money transfer in Hindi"),
        (re.compile(r"(?:सिक्योरिटी\s*डिपॉजिट|वेरिफिकेशन\s*फीस|जमानत\s*राशि|क्लियरेंस\s*फीस|प्रोसेसिंग\s*फीस)"), 0.91, "Demanding deposit or fees in Hindi"),
        # Marathi
        (re.compile(r"(?:पडताळणी\s*खात्यावर|सुरक्षित\s*खात्यावर)\s*.{0,25}(?:पैसे|रुपये)\s*(?:पाठवा|ट्रान्सफर)"), 0.95, "Transfer to verification account in Marathi"),
        (re.compile(r"(?:पैसे|रुपये|रक्कम)\s*.{0,25}(?:पाठवा|ट्रान्सफर\s*करा|पाठवून)"), 0.89, "Demand for money transfer in Marathi"),
        (re.compile(r"(?:नंबरवर\s+)?(?:\d+\s+)?(?:रुपये\s+पाठवून|पैसे\s+पाठवून)"), 0.88, "Instructing payment to a specific number in Marathi"),
    ],

    "kyc_fraud": [
        (re.compile(r"\bkyc\b.{0,35}\b(?:update|expired?|expire|suspend(?:ed)?|block(?:ed)?|verif(?:y|ication)|pending|incomplete|अपडेट|ब्लॉक|एक्सपायर|सस्पेंड|पेंडिंग)\b", re.IGNORECASE), 0.90, "Claims KYC is pending, expired, or causing account block"),
        (re.compile(r"\b(?:biometric\s*re-?verification|re-?verification|kyc\s*link|update\s*kyc)\b", re.IGNORECASE), 0.88, "Demands biometric re-verification or KYC updating"),
        (re.compile(r"\b(?:update|verify|complete|अपडेट)\b.{0,25}\bkyc\b", re.IGNORECASE), 0.87, "Instruction to complete KYC verification"),
        # Hindi
        (re.compile(r"(?:kyc|केवाईसी|के\s*वाई\s*सी)\s*.{0,35}(?:अपडेट|ब्लॉक|एक्सपायर|सस्पेंड|वेरिफिकेशन|पेंडिंग)"), 0.90, "KYC expiry or block threat in Hindi"),
        (re.compile(r"(?:बायोमेट्रिक\s*वेरिफिकेशन|kyc\s*अपडेट\s*करवाना)"), 0.88, "Biometric or KYC update demand in Hindi"),
        # Marathi
        (re.compile(r"(?:kyc|केवायसी|के\s*वाय\s*सी)\s*.{0,35}(?:अपडेट|पडताळणी|ब्लॉक|बंद)"), 0.90, "KYC verification or block in Marathi"),
    ],

    "remote_access": [
        (re.compile(r"\b(?:anydesk|teamviewer|rustdesk|quicksupport)\b", re.IGNORECASE), 0.95, "Explicit mention of remote-desktop application"),
        (re.compile(r"\b(?:install|download|open|run)\b.{0,30}\b(?:anydesk|teamviewer|rustdesk|quicksupport|screenshare|apk|support\s*app)\b", re.IGNORECASE), 0.96, "Command to install or launch remote access software"),
        (re.compile(r"\b(?:share\s+your\s+screen|screenshare|remote\s+access|give\s+access\s+to\s+your\s+device)\b", re.IGNORECASE), 0.94, "Instructing user to share screen or grant device access"),
        # Hindi
        (re.compile(r"(?:anydesk|teamviewer|rustdesk|quicksupport|स्क्रीन\s*शेयर|ऐप\s*डाउनलोड|इनस्टॉल\s*करें)"), 0.95, "Remote access or screen sharing command in Hindi"),
        # Marathi
        (re.compile(r"(?:anydesk|teamviewer|rustdesk|quicksupport|स्क्रीन\s*शेअर|अ‍ॅप\s*डाऊनलोड|डाऊनलोड\s*करा)"), 0.95, "Remote access or screen sharing command in Marathi"),
    ],

    "reward_scam": [
        (re.compile(r"\b(?:congratulations|you\s+have\s+won|you\s+won|winner\s+of)\b.{0,30}\b(?:lottery|prize|cashback|reward\s*points?|gift\s*card|bonus)\b", re.IGNORECASE), 0.92, "Announcing fake lottery winnings or unearned rewards"),
        (re.compile(r"\b(?:reward\s*points?|cashback|lottery\s*prize)\b.{0,25}\b(?:claim|expire|redeem|nikla\s+hai)\b", re.IGNORECASE), 0.89, "Urging victim to claim or redeem points/cashback"),
        (re.compile(r"\b\d+\s*(?:rupees?|rs\.?)\s*(?:ka\s+)?(?:reward|cashback|prize)\b", re.IGNORECASE), 0.88, "Specific reward or cashback amount announced"),
        # Lottery, lucky draw, online job reward, tax refund lures
        (re.compile(r"\b(?:inaam|inam|lucky\s*draw|lakh\s*jeete|jeet\s*gaye|bumper\s*prize|gift\s*voucher)\b", re.IGNORECASE), 0.90, "Hinglish lottery and prize stems"),
        (re.compile(r"\b(?:part-?time\s*(?:online\s*)?job|earn\s+\d+.*daily\s*(?:reward|cashback))\b", re.IGNORECASE), 0.89, "Part-time job earning lure"),
        (re.compile(r"\b(?:approved|sanctioned|pending)\b.{0,30}\b(?:tax\s*refund|income\s*tax\s*refund|refund)\b.{0,30}\b(?:claim|verify|receive|credit|prize)\b", re.IGNORECASE), 0.91, "Fake tax refund notification"),
        # Hindi
        (re.compile(r"(?:बधाई\s*हो|इनाम\s*जीता|लॉटरी\s*निकली|कैशबैक\s*मिला|रिवॉर्ड\s*पॉइंट्स\s*निकला|लकी\s*ड्रॉ|लाख\s*रुपये\s*का\s*इनाम)"), 0.92, "Lottery or cashback announcement in Hindi"),
        (re.compile(r"(?:रिवॉर्ड\s*पॉइंट्स|कैशबैक|इनाम|रिफंड)\s*.{0,25}(?:क्लेम\s*करना|प्राप्त\s*करने|एक्सपायर|दावा)"), 0.89, "Reward claiming instruction in Hindi"),
        # Marathi
        (re.compile(r"(?:अभिनंदन|बक्षीस\s*लागले|लॉटरी\s*लागली|कॅशबॅक\s*मिळाला|बक्षीस\s*जिंकले|रिवॉर्ड\s*पॉइंट|रिवॉर्ड\s*पॉइंट्स)"), 0.92, "Lottery or prize announcement in Marathi"),
        (re.compile(r"(?:रिवॉर्ड\s*पॉइंट्स?|बक्षीस|कॅशबॅक).{0,25}(?:रिडीम|क्लेम|एक्स्पायर)"), 0.89, "Reward claim in Marathi"),
    ],

    "identity_manipulation": [
        (re.compile(r"\b(?:share|tell|send|provide|give|enter)\b.{0,30}\b(?:aadhaar|pancard|pan\s*number|card\s*number|cvv|expiry\s*date|date\s*of\s*birth)\b", re.IGNORECASE), 0.90, "Demand to share Aadhaar, PAN, or debit/credit card number"),
        (re.compile(r"\b(?:card\s*number\s*(?:aur|and)\s*cvv|cvv\s*number)\b", re.IGNORECASE), 0.92, "Requesting card number and CVV details"),
        (re.compile(r"\b(?:share|provide|confirm)\s+(?:your\s+)?(?:aadhaar|pan|pancard)\s+(?:details|number)\b", re.IGNORECASE), 0.88, "Harvesting government identification numbers"),
        # Hindi & Multilingual
        (re.compile(r"(?:aadhaar|आधार|pancard|पैनकार्ड|पैन|पॅन|कार्ड\s*नंबर|सीवीवी)\b.{0,45}\b(?:बताओ|बताइए|दीजिए|भेजें|शेयर\s*कीजिए|शेयर\s*करें|सांगा|पाठवा|details|नंबर|number)"), 0.90, "Harvesting identity documents or card details in Hindi"),
        # Marathi
        (re.compile(r"(?:आधार\s*(?:कार्ड)?|पॅनकार्ड|कार्ड\s*नंबर)\s*.{0,25}(?:सांगा|पाठवा|द्या)"), 0.90, "Harvesting identity documents in Marathi"),
    ],
}

# General context check helpers
PRESSURE_MARKERS = re.compile(
    r"\b(cbi|police|customs|trai|rbi|arrest|warrant|fir|jail|narcotics|digital\s*arrest|court|illegal|freeze|blocked|suspended|turant|jaldi|immediately|urgent)\b|"
    r"(पोलीस|सीबीआय|कस्टम|गुन्हा|अटक|वारंट|डिजिटल\s*अटक|जेल|तुरंत|लगेच|तात्काळ)",
    re.IGNORECASE,
)

# Authority or Threat markers specifically used to decide if benign veto should be bypassed
AUTHORITY_OR_THREAT_MARKERS = re.compile(
    r"\b(cbi|police|customs|trai|rbi|inspector|officer|arrest|warrant|fir|jail|narcotics|digital\s*arrest|court|illegal|contraband|drugs|block|suspended|frozen|disconnect)\b|"
    r"(पोलीस|सीबीआय|कस्टम|गुन्हा|अटक|वारंट|डिजिटल\s*अटक|जेल|ब्लॉक|तस्करी|बेकायदेशीर)",
    re.IGNORECASE
)

# Benign delivery validation (e-commerce package arrival)
DELIVERY_BENIGN_MARKERS = re.compile(
    r"\b(?:delivery\s*(?:pin|otp|code)|delivery\s*person|standing\s*outside|at\s*(?:the\s+)?(?:gate|door)|with\s*(?:the\s+)?(?:parcel|order|package)|mark\s*(?:the\s+)?order\s*delivered|handover\s*(?:the\s+)?(?:parcel|package)|swiggy|zomato|amazon|flipkart|dunzo|zepto|courier\s*(?:boy|delivery))\b",
    re.IGNORECASE
)

# Benign branch visit instructions for KYC
BRANCH_VISIT_BENIGN_MARKERS = re.compile(
    r"\b(?:visit\s+(?:your\s+)?(?:nearest\s+)?(?:bank\s+)?branch|branch\s*(?:pe|par)\s*(?:jaiye|aaye|jao)|शाखेला\s*भेट\s*द्या|शाखेत\s*जाऊ|शाखा\s*में\s*जाएं)\b",
    re.IGNORECASE
)

# Benign emergency / personal assistance / medical assistance
BENIGN_ASSISTANCE_MARKERS = re.compile(
    r"\b(?:puncture|gadi|hospital|clinic|doctor|dentist|appointment|accident|dava|bhaiya|didi|kaka|mama|shaam\s+ko\s+(?:de\s+dunga|de\s+denge)|parat\s+karto|wapas\s+kar\s+dunga|reimburse|dinner|coffee|lunch)\b|"
    r"(?:गाडी|पंक्चर|दवाखाना|रुग्णालय|काका|संध्याकाळी\s*लगेच\s*परत|परत\s*करतो|संध्याकाळी\s*देतो)",
    re.IGNORECASE
)


class RegexTacticClassifier:
    """
    Regex and rule-based tactic classifier.
    Combines precompiled multilingual regular expressions with Noisy-OR aggregation,
    anti-scam negation suppression, and intelligent benign context vetoes.
    """

    def __init__(self):
        self.tactic_rules = TACTIC_RULES
        self.negation_patterns = NEGATION_PATTERNS

    def check_negation(self, text: str) -> bool:
        """Returns True if the utterance is an anti-scam warning or benign advisory."""
        return any(pattern.search(text) for pattern in self.negation_patterns)

    def apply_benign_vetoes(self, normalized_text: str, detected: Set[str]) -> Set[str]:
        """
        Applies general domain heuristics to suppress false positives on benign scenarios:
        1. Delivery PIN/OTP asked by delivery personnel outside door -> not credential phishing.
        2. Legitimate advice to visit physical bank branch for KYC -> not phishing link/fraud.
        3. Friendly/family roadside or medical emergency without authority or threat -> not coercion scam.
        """
        has_authority_or_threat = bool(AUTHORITY_OR_THREAT_MARKERS.search(normalized_text))

        # 1. Benign e-commerce delivery PIN veto
        if "otp_request" in detected and not has_authority_or_threat:
            if DELIVERY_BENIGN_MARKERS.search(normalized_text):
                detected.remove("otp_request")

        # 2. Benign physical branch KYC reminder veto
        if "kyc_fraud" in detected and not has_authority_or_threat:
            if BRANCH_VISIT_BENIGN_MARKERS.search(normalized_text):
                detected.remove("kyc_fraud")

        # 3. Benign roadside breakdown or personal family emergency veto on urgency
        if "urgency_creation" in detected and not has_authority_or_threat:
            if BENIGN_ASSISTANCE_MARKERS.search(normalized_text):
                detected.remove("urgency_creation")

        return detected

    def classify_regex(
        self,
        turn: Turn,
        context_str: str = "",
        has_prior_tactics: bool = False,
    ) -> List[TacticEvent]:
        """
        Synchronous core classification function.
        Returns a list of detected TacticEvents.
        """
        raw_text = turn.text or ""
        normalized = normalize_text(raw_text)
        if not normalized:
            return []

        # Check if caller is delivering an anti-fraud advisory ("never share OTP")
        is_negated = self.check_negation(normalized)
        if is_negated:
            # Genuine advisory / warning against fraud; suppress all scam tactics
            return []

        # Context cues: determine if caller was already applying pressure earlier
        prior_pressure = has_prior_tactics
        if not prior_pressure and context_str:
            prior_pressure = bool(PRESSURE_MARKERS.search(context_str))

        preliminary_events: Dict[str, Tuple[float, str, str]] = {}

        for tactic, rules in self.tactic_rules.items():
            matched_weights: List[float] = []
            matched_evidences: List[str] = []
            matched_explanations: List[str] = []

            for regex, base_weight, explanation in rules:
                match = regex.search(normalized)
                if match:
                    snippet = match.group(0).strip()
                    effective_weight = base_weight
                    # Context adjustment for sensitive requests
                    if tactic in {"otp_request", "financial_manipulation", "kyc_fraud", "remote_access"}:
                        if prior_pressure:
                            effective_weight = min(0.99, base_weight + 0.05)

                    if effective_weight > 0.0:
                        matched_weights.append(effective_weight)
                        matched_evidences.append(snippet)
                        matched_explanations.append(explanation)

            if not matched_weights:
                continue

            # Noisy-OR combination of matched rule weights:
            prob_none = 1.0
            for w in matched_weights:
                prob_none *= (1.0 - w)
            combined_conf = 1.0 - prob_none

            conf = round(min(1.0, max(0.0, combined_conf)), 2)

            if conf >= 0.40:
                evidence_text = matched_evidences[0] if matched_evidences else raw_text
                explanation_text = matched_explanations[0] if matched_explanations else f"Detected {tactic.replace('_', ' ')}"
                preliminary_events[tactic] = (conf, evidence_text, explanation_text)

        # Apply benign veto rules
        tactics_to_keep = self.apply_benign_vetoes(normalized, set(preliminary_events.keys()))

        detected_events: List[TacticEvent] = []
        for tactic in tactics_to_keep:
            conf, evidence_text, explanation_text = preliminary_events[tactic]
            detected_events.append(
                TacticEvent(
                    tactic=tactic,
                    confidence=conf,
                    evidence=evidence_text,
                    explanation=explanation_text,
                    t=turn.t1,
                    stage=0,
                    turn_id=turn.id,
                )
            )

        return detected_events

    async def classify_regex_async(
        self,
        turn: Turn,
        context_str: str = "",
        has_prior_tactics: bool = False,
    ) -> List[TacticEvent]:
        """Async wrapper matching the classify_utterance signature."""
        return self.classify_regex(turn, context_str=context_str, has_prior_tactics=has_prior_tactics)


# Global singleton instance
regex_classifier = RegexTacticClassifier()

def classify_regex(turn: Turn, context_str: str = "", has_prior_tactics: bool = False) -> List[TacticEvent]:
    return regex_classifier.classify_regex(turn, context_str=context_str, has_prior_tactics=has_prior_tactics)

async def classify_regex_async(turn: Turn, context_str: str = "", has_prior_tactics: bool = False) -> List[TacticEvent]:
    return await regex_classifier.classify_regex_async(turn, context_str=context_str, has_prior_tactics=has_prior_tactics)
