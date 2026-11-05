import hashlib
from scripts.verify_report import verify
from reporting import report_hash

def test_report_hash_and_verify(tmp_path):
    call = {"transcript": [{"text": "hello"}], "events": [], "risk_timeline": []}
    digest, raw = report_hash(call)
    path = tmp_path / "report.json"; path.write_bytes(raw)
    assert verify(path, digest) == (True, digest)
    assert hashlib.sha256(raw).hexdigest() == digest
