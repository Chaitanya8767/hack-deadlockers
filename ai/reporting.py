import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def canonical_payload(call):
    return {"events": call.get("events", []), "risk_timeline": call.get("risk_timeline", []), "transcript": call.get("transcript", [])}


def report_hash(call):
    raw = json.dumps(canonical_payload(call), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), raw


def create_report(call, scammer_identifier, output_dir):
    # Imported here so integrity verification stays dependency-light.
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    digest, raw = report_hash(call); stamp = datetime.now(timezone.utc).isoformat()
    base = output_dir / ("complaint-" + call["call_id"])
    json_path = base.with_suffix(".json"); pdf_path = base.with_suffix(".pdf")
    json_path.write_bytes(raw)
    c = canvas.Canvas(str(pdf_path), pagesize=A4); width, height = A4; y = height - 45
    def line(text, bold=False):
        nonlocal y
        if y < 55: c.showPage(); y = height - 45
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 10)
        for part in str(text).split("\n"):
            c.drawString(42, y, part[:130]); y -= 14
    def heading(text):
        nonlocal y
        y -= 8; line(text, True); y -= 3
    line("FIR-READY SUSPECTED CYBER-FRAUD COMPLAINT", True)
    line(f"Generated: {stamp}"); line(f"Scammer identifier (provided by reporter): {scammer_identifier or 'Not provided (WebRTC has no phone number)'}")
    line(f"Call started: {call.get('started_at')} | Duration: {call.get('duration_seconds', 0):.0f}s")
    st = call.get("scam_type", {}); heading("Scam type and evidence")
    line(f"{st.get('type', 'Unknown/Benign')} ({st.get('confidence', 0):.0%} confidence)")
    for e in st.get("evidence", []): line(f"[{e.get('t', 0):.1f}s] {e.get('tactic')}: {e.get('text')}")
    heading("Risk timeline")
    for item in call.get("risk_timeline", []): line(f"[{item.get('t', 0):.1f}s] {item.get('score')}% — {item.get('level')}")
    heading("Timestamped caller transcript")
    sensitive_sentences = {e.get("sentence", "") for e in call.get("events", []) if e.get("tactic") in {"otp_request", "identity_manipulation", "financial_manipulation"}}
    for t in call.get("transcript", []):
        text = t.get("text", "")
        marker = "[SENSITIVE DEMAND] " if text in sensitive_sentences else ""
        line(f"[{t.get('t0', 0):.1f}s] {marker}{text}")
    heading("Data already shared")
    for event in call.get("data_shared", []): line(f"[{event.get('t', '00:00')}] {event.get('kind', 'other')}")
    if not call.get("data_shared"): line("None recorded.")
    heading("Suggestions only, not legal advice")
    line("Potential references: BNS 318, 319, 204; IT Act 66C, 66D.")
    heading("Immediate reporting")
    line("Call 1930 immediately and report at https://cybercrime.gov.in.")
    heading("Evidence integrity")
    line(f"SHA-256 (canonical transcript + events + risk timeline): {digest}")
    line(f"Hash timestamp (UTC): {stamp}")
    c.save(); return pdf_path, json_path, digest
