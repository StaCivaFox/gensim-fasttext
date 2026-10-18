import hashlib
import json
from pathlib import Path

SEED = 42
# Predefined before any model is evaluated. Labels are page IDs, not topic names.
PAGE_ORDER = ["01", "06", "02", "08", "03", "04", "05", "07"]
TASKS = {k: PAGE_ORDER[:k] for k in (2, 4, 8)}


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stable_hash(word):
    return int.from_bytes(hashlib.md5(word.encode("utf-8")).digest()[:8], "little")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

