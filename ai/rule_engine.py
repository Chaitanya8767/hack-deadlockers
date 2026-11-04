"""
Local, word-based scam detector.

No LLM, no internet, no API key. It looks at the caller's transcribed
sentence and returns the same TacticEvent objects that the LLM classifier
produces, so the sequence analyzer, risk engine and UI work unchanged.

Languages: English, Hindi, Marathi, Hinglish (romanized Hindi).

Two kinds of rules:
  1. SENSITIVE ITEMS  - things a genuine caller never asks for
     (OTP, PIN, password, CVV, account number, card number, Aadhaar, PAN).
     Confidence depends on HOW it is asked:
        ask verb + item   ("tell me your OTP")        -> 0.95
        "your/aapka" + item ("aapka OTP")             -> 0.80
        item word alone                               -> 0.60
     Warnings such as "never share your OTP" are NOT flagged.
  2. TACTIC KEYWORDS - authority, fear, urgency, digital arrest, remote access,
     KYC, money transfer, reward.
"""
import re
from typing import List, Dict, Tuple

from conversation_manager import TacticEvent
import lexicon

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_L_BEFORE = r"(?<![a-z0-9])"
_L_AFTER = r"(?![a-z0-9])"


def _latin(*alts: str) -> str:
    """Whole-word match for Latin (English / romanized) words."""
    return _L_BEFORE + "(?:" + "|".join(alts) + ")" + _L_AFTER


def _mixed(latin: List[str], deva: List[str]) -> re.Pattern:
    """Latin alternatives with word boundaries + Devanagari alternatives as
    plain substrings (Python's \\b is unreliable around Devanagari matras)."""
    parts = []
    if latin:
        parts.append(_latin(*latin))
    if deva:
        parts.append("(?:" + "|".join(deva) + ")")
    return re.compile("|".join(parts), re.IGNORECASE)


def normalize(text: str) -> str:
    t = text.lower()
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# ---------------------------------------------------------------------------
# 1. SENSITIVE ITEMS  (what the caller wants from you)
# ---------------------------------------------------------------------------
# (key, tactic, human label, pattern)
SENSITIVE_ITEMS: List[Tuple[str, str, str, re.Pattern]] = [
    ("otp", "otp_request", "OTP / verification code", _mixed(
        [r"o\s*\.?\s*t\s*\.?\s*p", r"one[\s-]*time[\s-]*(?:password|pin|code)",
         r"verification\s*code", r"security\s*code", r"sms\s*code",
         r"(?:4|6|four|six)[\s-]*digit\s*(?:code|number|pin)", r"code\s*(?:on|received)"],
        [r"ओ\s*टी\s*पी", r"ओटीपी", r"वन\s*टाइम\s*पासवर्ड", r"वेरिफिकेशन\s*कोड", r"कोड"])),
    ("pin", "otp_request", "PIN", _mixed(
        [r"(?:m|upi|atm|debit|card|secret)[\s-]*pin", r"pin\s*(?:number|code)?", r"mpin"],
        [r"पिन", r"एमपिन"])),
    ("password", "otp_request", "password / login details", _mixed(
        [r"pass\s*word", r"passcode", r"passwd",
         r"net\s*banking\s*(?:id|password|details|login)",
         r"login\s*(?:id|details|credentials|password)",
         r"user\s*id\s*and\s*password"],
        [r"पासवर्ड", r"पासकोड", r"लॉगिन"])),
    ("cvv", "otp_request", "CVV / card expiry", _mixed(
        [r"cvv", r"cvc", r"card\s*verification", r"expiry\s*date", r"valid\s*(?:thru|till|upto)"],
        [r"सीव्हीव्ही", r"सीवीवी", r"एक्स्पायरी"])),
    ("account", "identity_manipulation", "bank account number", _mixed(
        [r"account\s*(?:no\.?|number|num|details)", r"a/?c\s*(?:no\.?|number)",
         r"bank\s*(?:account|details)", r"ifsc", r"khata\s*(?:number|no|sankhya)",
         r"account\s*ka\s*number", r"bank\s*account"],
        [r"खाते\s*(?:क्रमांक|नंबर|नं)", r"खाता\s*(?:संख्या|नंबर|नं)", r"अकाउंट\s*(?:नंबर|नं)",
         r"बँक\s*डिटेल", r"बैंक\s*डिटेल", r"बँक\s*खाते\s*(?:क्रमांक|नंबर)"])),
    ("card", "identity_manipulation", "card number", _mixed(
        [r"(?:debit|credit|atm)?\s*card\s*(?:number|no\.?|details|num)",
         r"16[\s-]*digit", r"card\s*ka\s*number"],
        [r"कार्ड\s*(?:नंबर|क्रमांक|नं|डिटेल)"])),
    ("aadhaar_pan", "identity_manipulation", "Aadhaar / PAN", _mixed(
        [r"aadh?a+r(?:\s*(?:number|no|card))?", r"pan\s*(?:card|number|no)"],
        [r"आधार", r"पॅन", r"पैन"])),
]

# ask verbs: caller is requesting something from you
ASK_VERBS = _mixed(
    [r"tell", r"share", r"give", r"send", r"provide", r"read\s*out", r"read", r"say",
     r"enter", r"type", r"confirm", r"verify", r"forward", r"dictate", r"need", r"require",
     r"batao", r"bataiye", r"bataye", r"btao", r"bata", r"bolo", r"boliye", r"dijiye",
     r"dedo", r"de\s*do", r"dena", r"bhejo", r"bhejiye", r"bhej\s*do", r"likho",
     r"sunao", r"chahiye", r"chaiye", r"sanga", r"sang", r"dya", r"dyaa", r"pathva",
     r"pathwa", r"pathav", r"sangaa", r"lagel", r"hava"],
    [r"बताओ", r"बताइए", r"बताएं", r"बता\s*दो", r"बता", r"बोलो", r"दीजिए", r"दीजिये", r"दे\s*दो",
     r"भेजो", r"भेजिए", r"भेजें", r"चाहिए", r"सांगा", r"सांग", r"द्या", r"द्यावा", r"पाठवा",
     r"सांगू", r"हवा", r"लागेल"])

POSSESSIVE = _mixed(
    [r"your", r"aapka", r"aapke", r"aapki", r"apna", r"apni", r"apka", r"tumcha", r"tumche",
     r"tumchi", r"tumhara", r"tera"],
    [r"आपका", r"आपके", r"आपकी", r"अपना", r"अपनी", r"तुमचा", r"तुमचे", r"तुमची", r"तुमचं",
     r"तुम्हारा", r"आपला"])

# "never share your OTP" / "bank never asks" -> warning, not a scam
WARNING = _mixed(
    [r"(?:never|don'?t|do\s*not|should\s*not|must\s*not|shouldn'?t)\s*(?:ever\s*)?"
     r"(?:share|give|disclose|reveal|provide|tell|send)",
     r"never\s*ask", r"(?:bank|we|banks)\s*(?:will\s*|would\s*)?never\s*ask",
     r"beware", r"fraud\s*alert", r"stay\s*alert", r"be\s*careful",
     r"(?:kisi\s*ko|kabhi\s*(?:bhi)?)\s*(?:mat|na|nahi|nahin)\s*(?:batana|batao|dena|dijiye|share)",
     r"(?:mat|na|nahi)\s*(?:batana|batao|dena|dijiye|share\s*karein|share\s*kare)",
     r"kabhi\s*nahi\s*(?:maangta|mangta|poochta|puchta)"],
    [r"कभी\s*(?:भी\s*)?(?:न|नहीं|मत)", r"किसी\s*को\s*(?:न|नहीं|मत)\s*(?:बताएं|बताना|दें|देना|बताइए)",
     r"(?:सांगू|देऊ|शेअर\s*करू)\s*नका", r"कोणालाही?\s*(?:सांगू|देऊ)\s*नका",
     r"सावध", r"सावधान"])

# if the caller says the warning but also "...to me/us", it is an ask
DIRECTED_AT_CALLER = _mixed(
    [r"to\s*(?:me|us)", r"mujhe", r"hume", r"humein", r"hamein", r"mala", r"amhala"],
    [r"मुझे", r"हमें", r"हमे", r"मला", r"आम्हाला"])


def _detect_sensitive(raw: str, norm: str, t: float) -> List[TacticEvent]:
    events: List[TacticEvent] = []
    is_warning = bool(WARNING.search(norm)) and not DIRECTED_AT_CALLER.search(norm)
    if is_warning:
        return events

    ask = ASK_VERBS.search(norm)
    poss = POSSESSIVE.search(norm)

    for key, tactic, label, pat in SENSITIVE_ITEMS:
        m = pat.search(norm)
        if not m:
            continue
        # "pin" alone is very short; require a context word to avoid noise like "pin code" of address
        if key == "pin" and not (ask or poss):
            continue
        if key == "otp" and m.group(0) in ("कोड",) and not (ask or poss):
            continue

        if ask:
            conf, how = 0.95, "asked you to give"
        elif poss:
            conf, how = 0.80, "mentioned your"
        else:
            conf, how = 0.60, "mentioned"

        start, end = m.start(), m.end()
        if ask:
            start, end = min(start, ask.start()), max(end, ask.end())
        evidence = _span_text(raw, norm, start, end, m.group(0))
        events.append(TacticEvent(
            tactic=tactic, confidence=conf, evidence=evidence, t=t,
            explanation=(f"Caller {how} {label}. Genuine banks, police or "
                         f"government offices never ask for this over a call."),
        ))
    return events


# ---------------------------------------------------------------------------
# 2. TACTIC KEYWORDS
# ---------------------------------------------------------------------------
# tactic -> list of (pattern, confidence, why)
TACTIC_RULES: Dict[str, List[Tuple[re.Pattern, float, str]]] = {
    "authority_impersonation": [
        (_mixed([r"(?:i\s*am|this\s*is|calling\s*from|speaking\s*from|main|mai|mein|hum)\s*"
                 r"(?:\w+\s*){0,3}?(?:cbi|police|customs|narcotics|ncb|rbi|trai|cyber\s*(?:crime|cell)|"
                 r"income\s*tax|enforcement\s*directorate|crime\s*branch|inspector|officer|fedex|dhl|"
                 r"bank\s*(?:manager|officer|executive)|customer\s*(?:care|support)|sbi|hdfc|icici|axis\s*bank|state\s*bank|pnb|paytm|amazon)"],
                [r"(?:सीबीआय|पोलीस|पुलिस|कस्टम|क्राइम\s*ब्रांच|आरबीआय|इन्कम\s*टॅक्स)\s*(?:मधून|से|विभाग|अधिकारी|ऑफिसर)",
                 r"(?:बँकेतून|बँकेकडून|बँकेच्या\s*कस्टमर\s*केअर|कस्टमर\s*केअर)\s*(?:मधून\s*)?बोल", r"बँक\s*(?:मॅनेजर|अधिकारी)"]),
         0.90, "Caller claims to be from an official agency or bank."),
        (_mixed([r"cbi", r"police", r"cyber\s*(?:crime|cell)", r"customs", r"narcotics", r"ncb",
                 r"enforcement\s*directorate", r"rbi", r"reserve\s*bank", r"trai",
                 r"income\s*tax", r"crime\s*branch", r"inspector", r"officer", r"supreme\s*court",
                 r"fedex", r"dhl", r"sebi", r"sbi", r"hdfc", r"icici", r"axis\s*bank", r"state\s*bank", r"pnb", r"customer\s*care", r"bank\s*manager", r"microsoft",
                 r"technical\s*support", r"tech\s*support", r"mahavitaran"],
                [r"महावितरण", r"स्टेट\s*बैंक", r"स्टेट\s*बँक", r"बैंक\s*ऑफ", r"बँक\s*ऑफ", r"विद्युत\s*विभाग", r"वीज\s*विभाग", r"सीबीआय", r"पोलीस", r"पुलिस", r"कस्टम", r"क्राइम\s*ब्रांच", r"आरबीआय", r"अधिकारी",
                 r"इन्कम\s*टॅक्स", r"सायबर", r"साइबर", r"न्यायालय", r"कोर्ट"]),
         0.60, "Caller mentions a police / government / bank authority."),
    ],
    "fear_manipulation": [
        (_mixed([r"arrest(?:ed)?", r"warrant", r"fir", r"jail", r"prison", r"criminal",
                 r"money\s*laundering", r"legal\s*action", r"case\s*(?:registered|filed|against|lagaya|darj)",
                 r"illegal", r"drugs?", r"seize[d]?", r"giraftar", r"gireftar", r"giraftari",
                 r"(?:account|sim|number|card|aadhaar)\s*(?:will\s*be\s*|ho\s*jayega\s*|has\s*been\s*)?"
                 r"(?:blocked|block|suspended|frozen|freeze|closed|deactivated|band)",
                 r"band\s*ho\s*jayega", r"block\s*ho\s*jayega", r"freeze\s*ho\s*jayega",
                 r"jail\s*(?:jana|hogi|mein)", r"penalty", r"suspended", r"police\s*complaint",
                 r"malicious", r"trojan", r"virus", r"malware", r"hacked", r"hacker",
                 r"(?:power|electricity|connection)\s*(?:will\s*be\s*)?(?:cut|disconnected)"],
                [r"गिरफ्तार", r"अटक", r"वारंट", r"गुन्हा", r"एफ\s*आय\s*आर", r"एफ\.?आई\.?आर", r"जेल", r"अरेस्ट",
                 r"तुरुंग", r"बेकायदेशीर", r"गैरकानूनी", r"ब्लॉक", r"बंद\s*(?:हो|होईल|होगा)", r"कारवाई",
                 r"मनी\s*लॉन्ड्रिंग", r"ड्रग", r"खंडित", r"कापली", r"कट\s*केली", r"कनेक्शन\s*कट"]),
         0.80, "Caller threatens arrest, legal action or blocking of your account."),
    ],
    "digital_arrest": [
        (_mixed([r"digital\s*arrest", r"stay\s*on\s*(?:the\s*)?(?:call|line|video)",
                 r"(?:don'?t|do\s*not|mat)\s*(?:disconnect|hang\s*up|cut|kaatna|kato)",
                 r"keep\s*(?:your\s*)?(?:camera|video)\s*on", r"camera\s*on",
                 r"(?:don'?t|do\s*not)\s*(?:tell|inform|contact|call)\s*(?:anyone|anybody|family|police|relatives)",
                 r"kisi\s*ko\s*(?:mat|na)\s*batana", r"under\s*(?:surveillance|investigation|observation)",
                 r"house\s*arrest", r"call\s*mat\s*kaat(?:na|o)"],
                [r"डिजिटल\s*(?:अटक|अरेस्ट|गिरफ्तारी)", r"अरेस्ट\s*कर", r"कॉल\s*(?:चालू|सुरू|चालु)", r"कट\s*(?:करू\s*नका|न\s*करें|मत)", r"कॉल\s*(?:कट\s*)?(?:मत|नका)", r"फोन\s*(?:कट\s*)?(?:मत|नका)",
                 r"कोणाला\s*(?:काही\s*)?सांगू\s*नका", r"किसी\s*को\s*(?:मत|न)\s*बता", r"कॅमेरा\s*चालू",
                 r"कैमरा\s*(?:चालू|ऑन)"]),
         0.92, "Caller keeps you on the line / isolates you: a classic 'digital arrest' scam move."),
    ],
    "urgency_creation": [
        (_mixed([r"urgent(?:ly)?", r"immediately", r"right\s*now", r"at\s*once", r"asap",
                 r"within\s*\d+\s*(?:min(?:ute)?s?|hours?)", r"last\s*chance", r"deadline",
                 r"before\s*it'?s\s*too\s*late", r"hurry", r"quickly", r"turant", r"jaldi",
                 r"foran", r"fatafat", r"abhi\s*(?:ke\s*abhi)?", r"lavkar", r"tatkal", r"lagech"],
                [r"तुरंत", r"जल्दी", r"अभी", r"तात्काळ", r"त्वरित", r"लगेच", r"लवकर", r"फौरन", r"आत्ताच", r"तातडी", r"ताबडतोब", r"लवकरात\s*लवकर", r"आत्ता\s*लगेच",
                 r"\d+\s*(?:मिनट|मिनिट|मिनिटांत|घंटे|तास)", r"आज\s*(?:रात्री|रात|शाम|शाम\s*तक)"]),
         0.70, "Caller is pushing you to act immediately."),
    ],
    "remote_access": [
        (_mixed([r"any\s*desk", r"team\s*viewer", r"rustdesk", r"quick\s*support", r"ammyy",
                 r"ultra\s*viewer", r"screen\s*shar(?:e|ing)", r"remote\s*(?:access|control|app|support)",
                 r"apk(?:\s*file)?",
                 r"(?:install|download)\s*(?:this\s*|the\s*|an?\s*)?(?:app|application|software|apk|file)",
                 r"(?:app|apk|software)\s*(?:install|download)\s*(?:karo|kijiye|kare|karna)",
                 r"click\s*(?:on\s*)?(?:the\s*|this\s*)?link", r"link\s*(?:par|pe|per)\s*click"],
                [r"एनी\s*डेस्क", r"एनीडेस्क", r"टीम\s*व्ह्यूअर", r"टीम\s*व्यूअर", r"स्क्रीन\s*शेअर",
                 r"स्क्रीन\s*शेयर", r"डाउनलोड\s*(?:करा|करें|करो|कर)", r"इन्स्टॉल", r"इंस्टॉल",
                 r"लिंक\s*(?:वर|पर)\s*क्लिक"]),
         0.90, "Caller wants you to install an app or share your screen (remote-access fraud)."),
    ],
    "kyc_fraud": [
        (_mixed([r"kyc", r"know\s*your\s*customer", r"re[\s-]*verif(?:y|ication)", r"biometric",
                 r"(?:sim|number)\s*(?:will\s*be\s*)?(?:deactivated|blocked|disconnected)"],
                [r"केवायसी", r"केवाईसी", r"केवाइसी", r"के\.?\s*वाय\.?\s*सी"]),
         0.80, "Caller claims your KYC is pending or expired."),
    ],
    "financial_manipulation": [
        (_mixed([r"safe\s*account", r"secure\s*account", r"rbi\s*account", r"government\s*account",
                 r"(?:verification|processing|clearance|registration|release)\s*(?:fee|amount|charges?)",
                 r"security\s*deposit", r"(?:transfer|send|deposit)\s*(?:the\s*|your\s*)?"
                 r"(?:money|amount|funds|balance|rs\.?|rupees|payment)",
                 r"pay\s*(?:the\s*)?(?:fine|penalty|fee|amount|charges?)",
                 r"(?:paise|paisa|rupaye|amount)\s*(?:transfer|bhejo|bhejiye|dalo|jama)",
                 r"gift\s*cards?", r"bitcoin", r"crypto", r"scan\s*(?:the\s*)?qr",
                 r"upi\s*(?:payment|request|collect)", r"collect\s*request"],
                [r"सुरक्षित\s*खाते", r"सेफ\s*अकाउंट", r"पैसे\s*(?:ट्रान्सफर|ट्रांसफर|भेजो|भेजें|पाठवा|जमा)",
                 r"ट्रान्सफर\s*करा", r"रुपये\s*(?:पाठवून|पाठवा|भेजो|भेजें)", r"पाठवून", r"ट्रांसफर\s*करो", r"रक्कम", r"जमा\s*करा", r"जमा\s*करें",
                 r"दंड\s*भरा", r"जुर्माना"]),
         0.88, "Caller asks you to move money to a 'safe' account or pay a fee."),
    ],
    "reward_scam": [
        (_mixed([r"lottery", r"jackpot", r"prize", r"you\s*(?:have\s*)?won", r"congratulations",
                 r"lucky\s*(?:draw|winner|customer)", r"cash\s*back", r"cashback", r"reward\s*points",
                 r"free\s*gift", r"kbc", r"inaam", r"inam"],
                [r"इनाम", r"लॉटरी", r"बक्षीस", r"जिंकले", r"जीते", r"जीता", r"कैशबैक", r"रिवॉर्ड"]),
         0.80, "Caller offers a prize, lottery or reward you never applied for."),
    ],
}



# ---------------------------------------------------------------------------
# 2b. EXTRA SCAM PHRASES  (Marathi / Hindi / Hinglish / English)
# ---------------------------------------------------------------------------
# To add more words: put them in the matching list below, OR (no code, no
# restart) call POST /rules/add, OR edit ai/custom_rules.json.
def _extra(tactic: str, latin: List[str], deva: List[str], conf: float, why: str) -> None:
    TACTIC_RULES.setdefault(tactic, []).append((_mixed(latin, deva), conf, why))


_APP = r"(?:ॲप|अ‍ॅप|अॅप|एप|ऐप|एप्लिकेशन|ॲप्लिकेशन)"

# --- who the caller pretends to be -----------------------------------------
_extra("authority_impersonation",
       [r"head\s*office", r"government\s*(?:officer|official|department)", r"investigation\s*(?:officer|team|department)",
        r"cyber\s*police", r"mumbai\s*police", r"delhi\s*police", r"nodal\s*officer", r"judge", r"advocate",
        r"courier\s*(?:company|office|service)", r"customs\s*department", r"electricity\s*(?:board|department|office)",
        r"insurance\s*(?:company|agent|advisor)", r"lic\s*(?:agent|office)", r"telecom\s*department", r"dot\s*department",
        r"aadhaar\s*(?:office|department)", r"passport\s*office", r"army\s*(?:officer|man)", r"i\s*am\s*(?:from|calling\s*from)\s*(?:the\s*)?bank",
        r"main\s*bank\s*se\s*bol", r"bank\s*se\s*bol(?:\s*raha|\s*rahi)?", r"police\s*station\s*se", r"cyber\s*cell\s*se"],
       [r"बँकेचा\s*प्रतिनिधी", r"बँकेचे\s*प्रतिनिधी", r"बँकेतून\s*बोल", r"बँकेकडून\s*बोल", r"बैंक\s*से\s*बोल",
        r"हेड\s*ऑफिस", r"हेड\s*ऑफिसमधून", r"सरकारी\s*अधिकारी", r"तपास\s*अधिकारी", r"न्यायाधीश", r"वकील",
        r"रिझर्व्ह\s*बँक", r"रिजर्व\s*बैंक", r"सायबर\s*क्राईम", r"सायबर\s*क्राइम", r"साइबर\s*क्राइम", r"सायबर\s*पोलीस",
        r"मुंबई\s*पोलीस", r"क्राईम\s*ब्रँच", r"क्राइम\s*ब्रांच", r"ईडी\s*(?:चे|से|कडून|विभाग)", r"एनसीबी", r"टेलिकॉम\s*विभाग",
        r"दूरसंचार\s*विभाग", r"कुरिअर\s*(?:कंपनी|ऑफिस|सर्व्हिस)", r"कुरियर\s*(?:कंपनी|ऑफिस)", r"फेडेक्स", r"डीएचएल",
        r"कस्टमर\s*केअर", r"कस्टमर\s*केयर", r"ग्राहक\s*सेवा", r"विमा\s*(?:कंपनी|प्रतिनिधी)", r"एलआयसी",
        r"पोलीस\s*स्टेशन", r"पोलिस\s*स्टेशन", r"थाण्यातून", r"थाने\s*से", r"वीज\s*(?:मंडळ|कंपनी)", r"बिजली\s*(?:विभाग|कंपनी)"],
       0.75, "Caller claims to represent a bank, police, courier, court or government office.")

# --- fear / threats ---------------------------------------------------------
_extra("fear_manipulation",
       [r"case\s*(?:has\s*been\s*)?(?:registered|filed)", r"(?:your\s*)?(?:sim|number|account)\s*(?:will\s*be\s*)?(?:blocked|closed|deactivated)",
        r"legal\s*(?:notice|proceedings|trouble)", r"court\s*(?:notice|case|summons)", r"summons", r"bail", r"kidnap(?:ped)?",
        r"accident", r"in\s*(?:the\s*)?hospital", r"hawala", r"smuggl(?:ing|ed)", r"narcotic", r"parcel\s*(?:contains|mein|me)",
        r"frozen", r"under\s*investigation", r"trouble\s*(?:mein|me)\s*pad", r"kanooni\s*karrawai", r"karwai", r"case\s*darj",
        r"aapka\s*(?:beta|son|bacha|bachcha|ladka)\s*(?:pakda|accident|hospital)", r"(?:bijli|vij)\s*(?:kat|cut)"],
       [r"खाते\s*बंद\s*होणार", r"खाते\s*ब्लॉक", r"खाते\s*गोठव", r"खाते\s*सस्पेंड", r"खाते\s*निलंबित", r"सिम\s*बंद", r"नंबर\s*बंद",
        r"सिम\s*कार्ड\s*बंद", r"गुन्हा\s*दाखल", r"केस\s*दाखल", r"तक्रार\s*दाखल", r"अटक\s*वॉरंट", r"अटक\s*वारंट", r"अटक\s*होईल",
        r"अटक\s*केली\s*जाईल", r"कायदेशीर\s*(?:कारवाई|नोटीस)", r"कोर्ट\s*(?:नोटीस|केस)", r"न्यायालयात", r"समन्स", r"जामीन",
        r"अपघात", r"हॉस्पिटलमध्ये", r"रुग्णालयात", r"अपहरण", r"तस्करी", r"हवाला", r"अमली\s*पदार्थ", r"ड्रग्ज", r"मनी\s*लाँड्रिंग",
        r"पार्सल\s*मध्ये", r"पार्सलमध्ये", r"संशयास्पद", r"गोठवले", r"गोठवण्यात", r"शिक्षा\s*होईल", r"तुरुंगात",
        r"खाता\s*बंद\s*हो", r"खाता\s*ब्लॉक", r"खाता\s*फ्रीज", r"गिरफ्तारी\s*वारंट", r"केस\s*दर्ज", r"कानूनी\s*(?:कार्रवाई|नोटिस)",
        r"सिम\s*बंद\s*हो", r"नंबर\s*बंद\s*हो", r"बेटे\s*का\s*एक्सीडेंट", r"बेटा\s*पकड़ा", r"अपहरण\s*कर", r"जमानत",
        r"बिजली\s*कट", r"बिल\s*(?:भरलेले|भरला)\s*नाही", r"बिल\s*नहीं\s*भरा", r"वीज\s*कनेक्शन\s*(?:कट|बंद|खंडित)"],
       0.78, "Caller threatens you or a family member, or says your account / SIM / power will be cut.")

# --- urgency ----------------------------------------------------------------
_extra("urgency_creation",
       [r"today\s*only", r"by\s*(?:today|tonight|evening)", r"or\s*else", r"otherwise", r"no\s*time", r"don'?t\s*delay",
        r"do\s*not\s*delay", r"time\s*(?:is\s*)?(?:running|up|limited)", r"few\s*minutes", r"in\s*(?:the\s*)?next\s*\d+\s*(?:min|minutes|hours)",
        r"aaj\s*hi", r"warna", r"varna", r"nahi\s*to", r"der\s*mat", r"der\s*na\s*karein", r"abhi\s*abhi", r"aajach", r"nahitar", r"vel\s*nahi"],
       [r"आजच", r"आताच", r"आत्ताच", r"लगेचच", r"ताबडतोब", r"काही\s*मिनिटांत", r"थोड्या\s*वेळात", r"दोन\s*तासांत", r"एका\s*तासात",
        r"अन्यथा", r"नाहीतर", r"नाही\s*तर", r"वेळ\s*नाही", r"वेळ\s*कमी", r"शेवटची\s*संधी", r"उशीर\s*करू\s*नका", r"आज\s*रात्रीपर्यंत",
        r"आजच\s*करा", r"आता\s*लगेच", r"आज\s*ही", r"वरना", r"नहीं\s*तो", r"आख़िरी\s*मौका", r"आखिरी\s*मौका", r"देर\s*मत", r"जल्द\s*से\s*जल्द",
        r"अंतिम\s*तारीख", r"अंतिम\s*मुदत", r"मुदत\s*संपत"],
       0.68, "Caller is rushing you or giving a deadline.")

# --- OTP / PIN / password ---------------------------------------------------
_extra("otp_request",
       [r"code\s*(?:jo\s*)?(?:aaya|aayi|aaye|received|sent)", r"message\s*(?:mein|me|mai)\s*(?:aaya|aayi)", r"sms\s*(?:mein|me)\s*(?:aaya|aayi)",
        r"(?:6|six)[\s-]*digit", r"(?:4|four)[\s-]*digit", r"upi\s*pin", r"atm\s*pin", r"secret\s*code", r"authentication\s*code",
        r"verification\s*number", r"phone\s*pe\s*(?:aaya|aayi)", r"mobile\s*pe\s*(?:aaya|aayi)", r"number\s*(?:jo\s*)?(?:aaya|aayi)"],
       [r"मेसेज\s*मध्ये\s*आलेला", r"मेसेजमध्ये\s*आलेला", r"संदेशात\s*आलेला", r"मोबाईलवर\s*आलेला", r"फोनवर\s*आलेला", r"आलेला\s*कोड",
        r"आलेला\s*नंबर", r"एसएमएस\s*कोड", r"सहा\s*अंकी", r"चार\s*अंकी", r"[४6]\s*अंकी", r"६\s*अंकी", r"छह\s*अंक", r"छः\s*अंक", r"चार\s*अंक",
        r"यूपीआय\s*पिन", r"युपीआय\s*पिन", r"एटीएम\s*पिन", r"गुप्त\s*(?:कोड|क्रमांक|पिन)", r"मैसेज\s*में\s*आया", r"फोन\s*पर\s*आया",
        r"कोड\s*बताओ", r"कोड\s*सांगा", r"पासवर्ड\s*सांगा", r"पिन\s*सांगा", r"ओटीपी\s*(?:नंबर|क्रमांक)"],
       0.85, "Caller wants a code / PIN / password that was sent to your phone.")

# --- digital arrest / isolation ---------------------------------------------
_extra("digital_arrest",
       [r"stay\s*in\s*(?:the\s*|your\s*)?room", r"don'?t\s*(?:leave|go\s*out)", r"do\s*not\s*(?:leave|go\s*out)", r"don'?t\s*tell\s*(?:your\s*)?(?:family|wife|husband|parents|anyone|bank)",
        r"keep\s*(?:this|it)\s*(?:secret|confidential)", r"confidential", r"video\s*call\s*(?:on|chalu|start)", r"on\s*video\s*call", r"whatsapp\s*video",
        r"investigation\s*(?:is\s*)?(?:going\s*on|ongoing|complete)", r"till\s*(?:the\s*)?investigation", r"line\s*pe\s*bane\s*rahiye", r"kisi\s*se\s*baat\s*mat",
        r"ghar\s*walo\s*ko\s*(?:mat|na)", r"kamre\s*(?:mein|me)\s*(?:rahiye|raho)", r"ghari\s*kunala\s*sangu\s*naka"],
       [r"कॉल\s*कट\s*करू\s*नका", r"कॉल\s*बंद\s*करू\s*नका", r"फोन\s*ठेवू\s*नका", r"फोन\s*कट\s*करू\s*नका", r"लाईन\s*वर\s*रहा", r"लाइन\s*वर\s*रहा",
        r"व्हिडिओ\s*कॉल\s*(?:चालू|सुरू|वर)", r"व्हिडीओ\s*कॉल", r"कॅमेरा\s*चालू\s*ठेवा", r"खोलीतून\s*बाहेर", r"खोलीत\s*(?:थांबा|राहा)", r"बाहेर\s*जाऊ\s*नका",
        r"घरच्यांना\s*सांगू\s*नका", r"घरच्यांना\s*काही\s*सांगू", r"कुटुंबाला\s*सांगू\s*नका", r"कोणालाही\s*सांगू\s*नका", r"कोणाशीही\s*बोलू\s*नका",
        r"गोपनीय", r"गुप्त\s*ठेवा", r"तपास\s*पूर्ण\s*होईपर्यंत", r"चौकशी\s*पूर्ण\s*होईपर्यंत", r"देखरेखीखाली", r"निगराणी", r"नजरकैद",
        r"लाइन\s*पर\s*बने\s*रहिए", r"लाइन\s*पर\s*रहिए", r"कॉल\s*मत\s*काट", r"फोन\s*मत\s*काट", r"घरवालों\s*को\s*मत", r"किसी\s*से\s*बात\s*मत",
        r"कमरे\s*से\s*बाहर", r"जांच\s*पूरी\s*होने\s*तक", r"निगरानी\s*में"],
       0.90, "Caller isolates you or keeps you on the call (digital-arrest style).")

# --- remote access / malicious apps / links ---------------------------------
_extra("remote_access",
       [r"play\s*store", r"app\s*store", r"install\s*kar", r"download\s*kar", r"link\s*(?:bheja|sent|send|bhej)", r"i\s*(?:have\s*)?sent\s*(?:you\s*)?a\s*link",
        r"open\s*(?:the\s*)?link", r"click\s*(?:on\s*)?(?:the\s*)?(?:link|button)", r"share\s*(?:your\s*)?screen", r"screen\s*dikha", r"apk\s*file",
        r"quick\s*support", r"alpemix", r"aweray", r"airdroid", r"hoxx", r"mobile\s*(?:ka\s*)?control", r"access\s*(?:to\s*)?(?:your\s*)?(?:phone|mobile|computer)",
        r"allow\s*(?:the\s*)?(?:permission|access)", r"accessibility\s*(?:permission|service)"],
       [_APP + r"\s*(?:डाउनलोड|इन्स्टॉल|इंस्टॉल|इन्स्टाल)", r"(?:डाउनलोड|इन्स्टॉल|इंस्टॉल)\s*(?:करा|करें|करो|कर)", r"प्ले\s*स्टोअर", r"प्ले\s*स्टोर",
        r"एपीके", r"एनी\s*डेस्क", r"क्विक\s*सपोर्ट", r"टीम\s*व्ह्यू", r"लिंक\s*पाठवली", r"लिंक\s*पाठवला", r"लिंकवर\s*क्लिक", r"लिंक\s*वर\s*क्लिक",
        r"लिंक\s*भेजी", r"लिंक\s*भेजा", r"लिंक\s*खोला", r"लिंक\s*उघडा", r"स्क्रीन\s*(?:दाखवा|शेअर|शेयर|दिखाओ)", r"मोबाईलचा\s*ताबा", r"मोबाइल\s*का\s*कंट्रोल",
        r"परवानगी\s*(?:द्या|दे)", r"एक्सेस\s*(?:द्या|दें|दो)", r"ॲक्सेसिबिलिटी", r"रिमोट"],
       0.85, "Caller asks you to open a link, install an app or give access to your phone.")

# --- KYC / document update --------------------------------------------------
_extra("kyc_fraud",
       [r"pan\s*(?:card\s*)?(?:update|link|verification)", r"aadhaar\s*(?:update|link|verification|seeding)", r"(?:bank\s*)?account\s*(?:update|verification|verify)",
        r"documents?\s*(?:verification|update|expired?)", r"sim\s*(?:verification|re[\s-]*verification)", r"(?:account|card)\s*(?:expired|expiry)",
        r"update\s*(?:your\s*)?(?:details|information|profile|kyc|pan|aadhaar)", r"link\s*(?:your\s*)?(?:pan|aadhaar)", r"net\s*banking\s*(?:expired|update)",
        r"e[\s-]*kyc", r"video\s*kyc", r"kyc\s*(?:pending|incomplete|expired?)"],
       [r"केवायसी\s*(?:अपडेट|पूर्ण|पेंडिंग|एक्स्पायर|संपली|करा|करणे|करें|करो)", r"केवाईसी\s*(?:अपडेट|पूर्ण)", r"पॅन\s*(?:कार्ड\s*)?(?:अपडेट|लिंक)", r"पैन\s*(?:कार्ड\s*)?(?:अपडेट|लिंक)",
        r"आधार\s*(?:अपडेट|लिंक|व्हेरिफिकेशन|वेरिफिकेशन)", r"खाते\s*(?:अपडेट|व्हेरिफाय|पडताळणी|तपासणी)", r"खाता\s*(?:अपडेट|वेरिफाई|वेरिफिकेशन)", r"कागदपत्र(?:े)?\s*(?:पडताळणी|अपडेट|सादर)",
        r"दस्तावेज\s*(?:सत्यापन|अपडेट)", r"सिम\s*(?:व्हेरिफिकेशन|पडताळणी|वेरिफिकेशन)", r"माहिती\s*अपडेट", r"तपशील\s*अपडेट", r"डिटेल\s*अपडेट",
        r"ओळखपत्र\s*(?:पडताळणी|अपडेट)", r"नेट\s*बँकिंग\s*(?:अपडेट|एक्स्पायर)", r"मुदत\s*संपली", r"एक्स्पायर\s*झाले", r"पेंडिंग\s*आहे", r"पडताळणी\s*(?:पूर्ण|करा|बाकी)"],
       0.78, "Caller says your KYC, PAN, Aadhaar or SIM details must be updated.")

# --- money movement, fees, refunds ------------------------------------------
_extra("financial_manipulation",
       [r"refund", r"wrong\s*(?:transfer|payment)", r"(?:sent|transferred)\s*(?:money\s*)?by\s*mistake", r"mistakenly\s*(?:sent|transferred)", r"galti\s*se\s*(?:paise|amount|transfer)",
        r"google\s*pay", r"gpay", r"phone\s*pe", r"phonepe", r"paytm", r"bhim", r"scan\s*(?:this\s*)?qr", r"qr\s*code", r"processing\s*fee", r"clearance\s*(?:fee|charge)",
        r"customs\s*(?:duty|charge|fee)", r"delivery\s*(?:charge|fee)", r"loan\s*(?:approved|sanctioned|offer)", r"pre[\s-]*approved\s*loan", r"insurance\s*(?:premium|bonus|maturity)",
        r"policy\s*(?:bonus|maturity|lapsed)", r"invest(?:ment)?\s*(?:scheme|plan|opportunity|returns?)", r"double\s*(?:your\s*)?money", r"guaranteed\s*(?:returns?|profit)", r"trading\s*(?:app|group|tips)",
        r"stock\s*(?:tips|group)", r"fixed\s*returns?", r"pay\s*(?:\d+|some|small|little)\s*(?:rs|rupees|amount)?", r"registration\s*(?:fee|charge)",
        r"bank\s*(?:me|mein)\s*(?:paise|amount)\s*(?:daal|dal|dalo)", r"account\s*(?:me|mein)\s*(?:paise|amount)\s*(?:daal|dal|dalo)"],
       [r"पैसे\s*पाठवा", r"पैसे\s*पाठवून", r"रक्कम\s*पाठवा", r"रक्कम\s*जमा", r"पैसे\s*जमा\s*करा", r"खात्यात\s*पैसे", r"खात्यावर\s*पैसे", r"नंबरवर\s*पैसे", r"नंबरवर\s*रुपये",
        r"सुरक्षित\s*खात्यात", r"सरकारी\s*खात्यात", r"आरबीआय\s*खात्यात", r"पडताळणी\s*(?:शुल्क|फी|खाते)", r"प्रोसेसिंग\s*(?:फी|फीस|शुल्क)", r"प्रक्रिया\s*शुल्क", r"क्लिअरन्स\s*(?:फी|शुल्क)",
        r"सिक्युरिटी\s*डिपॉझिट", r"अनामत\s*रक्कम", r"दंड\s*भरा", r"दंड\s*भरावा", r"कस्टम\s*(?:शुल्क|ड्युटी)", r"डिलिव्हरी\s*(?:चार्ज|शुल्क)", r"नोंदणी\s*(?:शुल्क|फी)",
        r"परतावा", r"रिफंड", r"चुकून\s*(?:पैसे|रक्कम)", r"चुकीने\s*(?:पैसे|रक्कम)", r"गुगल\s*पे", r"फोन\s*पे", r"फोनपे", r"पेटीएम", r"यूपीआय", r"युपीआय",
        r"क्यूआर\s*(?:कोड)?", r"गिफ्ट\s*कार्ड", r"बिटकॉइन", r"क्रिप्टो", r"कर्ज\s*मंजूर", r"लोन\s*मंजूर", r"विमा\s*(?:बोनस|मॅच्युरिटी|हप्ता)", r"पॉलिसी\s*(?:बोनस|बंद)", r"गुंतवणूक\s*(?:योजना|करा)",
        r"दुप्पट\s*पैसे", r"दुप्पट\s*रक्कम", r"हमखास\s*(?:नफा|परतावा)", r"निश्चित\s*(?:नफा|परतावा)", r"ट्रेडिंग\s*(?:ॲप|ग्रुप|टिप्स)",
        r"पैसे\s*भेजो", r"पैसे\s*भेजें", r"रकम\s*जमा", r"सेफ\s*अकाउंट", r"प्रोसेसिंग\s*फीस", r"क्यूआर\s*कोड", r"लोन\s*पास", r"गलती\s*से\s*(?:पैसे|रकम)", r"दोगुना\s*(?:पैसा|पैसे|रकम)",
        r"पक्का\s*मुनाफा", r"गारंटीड\s*(?:रिटर्न|मुनाफा)", r"बीमा\s*(?:बोनस|मैच्योरिटी)", r"जुर्माना\s*भरो", r"शुल्क\s*भरो"],
       0.82, "Caller asks for money, a fee, a UPI/QR payment, or offers a loan / investment / refund.")

# --- prizes, jobs, freebies -------------------------------------------------
_extra("reward_scam",
       [r"work\s*from\s*home", r"part[\s-]*time\s*(?:job|work)", r"online\s*(?:job|task|earning)", r"earn\s*(?:money\s*)?(?:daily|per\s*day|from\s*home|online)", r"like\s*(?:and|&)\s*(?:subscribe|earn)",
        r"task\s*(?:complete|based)", r"youtube\s*(?:video|like)", r"free\s*(?:mobile|phone|laptop|gift|recharge|offer)", r"you\s*(?:are\s*)?selected", r"you\s*have\s*been\s*selected",
        r"lucky\s*(?:draw|winner|customer|number)", r"coupon", r"gift\s*(?:voucher|hamper)", r"bumper\s*(?:prize|draw)", r"congrats", r"aapko\s*(?:inaam|lottery|gift)\s*(?:mila|mili|laga|lagi)",
        r"inaam\s*(?:nikla|laga)", r"lakh\s*(?:rupees?|rs)?\s*(?:won|jeet|jitne)", r"won\s*(?:rs\.?|rupees|\d+\s*lakh)", r"(?:rs\.?|rupees)\s*[\d,]+\s*(?:cashback|reward|prize)"],
       [r"बक्षीस\s*(?:लागले|जिंकले|मिळाले)", r"लॉटरी\s*(?:लागली|जिंकली)", r"तुम्ही\s*जिंकले", r"तुम्हाला\s*(?:बक्षीस|इनाम|भेट)", r"भेटवस्तू", r"मोफत\s*(?:मोबाईल|फोन|रिचार्ज|भेट)",
        r"लकी\s*ड्रॉ", r"भाग्यवान\s*(?:विजेते|ग्राहक)", r"कूपन", r"गिफ्ट\s*(?:व्हाउचर|हॅम्पर)", r"अभिनंदन", r"केबीसी", r"कौन\s*बनेगा\s*करोड़पती", r"कौन\s*बनेगा\s*करोडपती",
        r"घरबसल्या", r"घर\s*बैठे", r"पार्ट\s*टाइम\s*(?:जॉब|काम|नोकरी)", r"पार्ट\s*टाईम", r"ऑनलाइन\s*(?:कमाई|नोकरी|काम)", r"ऑनलाईन\s*(?:कमाई|नोकरी|काम)", r"दररोज\s*कमवा", r"रोज\s*कमाओ", r"रोज\s*कमाएं",
        r"टास्क\s*(?:पूर्ण|पूरा)", r"निवड\s*झाली", r"तुमची\s*निवड", r"आपका\s*चयन", r"इनाम\s*(?:निकला|लगा|मिला)", r"आपको\s*इनाम", r"लाख\s*(?:रुपये|रुपए)?\s*(?:जिंकले|जीते)", r"रिवॉर्ड\s*पॉइंट", r"बक्षिस"],
       0.72, "Caller offers a prize, a free gift or easy online earning you never asked for.")


# ---------------------------------------------------------------------------
# 2c. GENERIC "asks for something personal" (catches items NOT in our list)
# ---------------------------------------------------------------------------
# ask verb + "number / code / details" + your/me  ->  suspicious even if the
# exact item (e.g. "ATM card back number") was never listed.
THING_WORDS = _mixed(
    [r"number", r"no\.", r"code", r"digits?", r"details", r"detail", r"information", r"info", r"id", r"credentials", r"photo", r"selfie", r"signature",
     r"nambar", r"number\s*bata", r"jaankari", r"jankari", r"mahiti", r"tapshil"],
    [r"नंबर", r"नं\.", r"क्रमांक", r"कोड", r"अंक", r"डिटेल", r"तपशील", r"माहिती", r"जानकारी", r"आयडी", r"फोटो", r"सेल्फी", r"सही", r"स्वाक्षरी", r"कागदपत्र", r"दस्तावेज"])
TO_ME = _mixed([r"to\s*(?:me|us)", r"me\s*(?:the|your)", r"mujhe", r"hume", r"humein", r"hamein", r"mala", r"amhala", r"mere\s*ko"],
               [r"मुझे", r"हमें", r"हमे", r"मला", r"आम्हाला", r"मेरे\s*को", r"मला\s*द्या"])


def _detect_generic_ask(raw: str, norm: str, t: float) -> List[TacticEvent]:
    if WARNING.search(norm) and not DIRECTED_AT_CALLER.search(norm):
        return []
    ask, thing = ASK_VERBS.search(norm), THING_WORDS.search(norm)
    if not (ask and thing):
        return []
    mine, to_me = POSSESSIVE.search(norm), TO_ME.search(norm)
    if not (mine or to_me):
        return []
    s, e = min(ask.start(), thing.start()), max(ask.end(), thing.end())
    return [TacticEvent(
        tactic="identity_manipulation", confidence=0.60 if (mine and to_me) else 0.52, t=t,
        evidence=_span_text(raw, norm, s, e, thing.group(0)),
        explanation="Caller is asking you to hand over a number, code or personal detail. "
                    "Real banks and officials do not collect these on a call.")]


def _detect_fuzzy(raw: str, norm: str, t: float, is_warning: bool) -> List[TacticEvent]:
    """Spelling variants / near-misses of scam words (ASR errors, typos)."""
    events: List[TacticEvent] = []
    ask = ASK_VERBS.search(norm)
    poss = POSSESSIVE.search(norm)
    for word, tactic, sensitive, label, conf in lexicon.fuzzy_hits(norm):
        if sensitive:
            if is_warning:
                continue
            if ask or poss:
                conf = max(conf, 0.85 if (ask and conf >= 0.7) else 0.65)
            else:
                conf = min(conf, 0.55)
        events.append(TacticEvent(
            tactic=tactic, confidence=conf, evidence=word, t=t,
            explanation=f"Word that sounds like '{label}' was heard (possible spelling variant)."))
    return events


def _detect_custom(raw: str, norm: str, t: float, is_warning: bool) -> List[TacticEvent]:
    events: List[TacticEvent] = []
    for phrase, tactic, conf, note in lexicon.custom_rules.match(norm):
        if is_warning and tactic in ("otp_request", "identity_manipulation"):
            continue
        events.append(TacticEvent(
            tactic=tactic, confidence=conf, evidence=phrase, t=t,
            explanation=note or "Matched a phrase added to the custom scam list."))
    return events


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def _span_text(raw: str, norm: str, start: int, end: int, fallback: str) -> str:
    """Return the exact text from the original sentence (so the UI can
    highlight it). Falls back to the matched string."""
    if len(raw) == len(norm):
        return raw[start:end].strip()
    idx = raw.lower().find(fallback.lower())
    return raw[idx:idx + len(fallback)] if idx >= 0 else fallback


def analyze_text(text: str, t: float = 0.0) -> List[TacticEvent]:
    """Analyze ONE caller sentence. Returns zero or more TacticEvent."""
    if not text or not text.strip():
        return []

    raw = re.sub(r"\s+", " ", text.strip())
    norm = normalize(raw)

    found: Dict[str, TacticEvent] = {}

    # 1. what the caller is asking for (OTP, password, account number, ...)
    for ev in _detect_sensitive(raw, norm, t):
        prev = found.get(ev.tactic)
        if prev is None or ev.confidence > prev.confidence:
            found[ev.tactic] = ev

    # 2. tactic keywords (original + extra phrase lists)
    for tactic, rules in TACTIC_RULES.items():
        for pat, conf, why in rules:
            m = pat.search(norm)
            if not m:
                continue
            prev = found.get(tactic)
            if prev is None or conf > prev.confidence:
                found[tactic] = TacticEvent(
                    tactic=tactic, confidence=conf, t=t, explanation=why,
                    evidence=_span_text(raw, norm, m.start(), m.end(), m.group(0)),
                )

    # 3. safety nets for words we never listed
    is_warning = bool(WARNING.search(norm)) and not DIRECTED_AT_CALLER.search(norm)
    extra: List[TacticEvent] = []
    extra += _detect_custom(raw, norm, t, is_warning)          # team-added phrases
    extra += _detect_fuzzy(raw, norm, t, is_warning)           # spelling variants
    if "identity_manipulation" not in found:
        extra += _detect_generic_ask(raw, norm, t)             # "give me your <anything>"
    for ev in extra:
        prev = found.get(ev.tactic)
        if prev is None or ev.confidence > prev.confidence:
            found[ev.tactic] = ev

    # "never tell anyone your OTP" is a SAFETY warning, not 'don't tell your family'
    if is_warning and any(p.search(norm) for _, _, _, p in SENSITIVE_ITEMS):
        found.pop("digital_arrest", None)

    return list(found.values())
