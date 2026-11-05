"""Verify a saved report JSON against its expected SHA-256 hash."""
import argparse
import hashlib
from pathlib import Path

def verify(path, expected):
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return actual == expected, actual

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("json_path"); parser.add_argument("sha256")
    args = parser.parse_args(); ok, actual = verify(args.json_path, args.sha256)
    print(actual); raise SystemExit(0 if ok else 1)
