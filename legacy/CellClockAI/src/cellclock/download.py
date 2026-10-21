"""Download the exact public processed matrices used in the empirical pilot."""
import hashlib
import json
import urllib.request
from pathlib import Path

FILES = {
    "GSE165177_Log2_RPM_Transient_reprogramming.txt.gz":
    "e6f055a491900a8f8563a339d1f18d99a92690af53d8ff44d7cc9a164b66209c",
    "GSE165177_Log2_RPM_Transient_reprogramming_part2_170621.txt.gz":
    "fca65676b19599491e139c583971a88f6bf2d177592d49558e57689e18e067e8",
}
BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE165nnn/GSE165177/suppl/"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(root):
    root = Path(root)
    dest = root / "data/raw"
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for name, digest in FILES.items():
        path = dest / name
        if not path.exists():
            tmp = path.with_suffix(".download")
            try:
                with urllib.request.urlopen(BASE + name, timeout=120) as r, tmp.open("wb") as f:
                    for block in iter(lambda: r.read(1024 * 1024), b""):
                        f.write(block)
                if sha256(tmp) != digest:
                    raise ValueError(f"Checksum mismatch for {name}; source changed")
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
        if sha256(path) != digest:
            raise ValueError(f"Checksum mismatch for cached {name}")
        records.append({"accession": "GSE165177", "file": name, "url": BASE + name,
                        "sha256": digest, "bytes": path.stat().st_size,
                        "source_type": "public_observed_RNAseq", "accessed": "2026-10-13"})
    (root / "data/provenance.json").write_text(json.dumps(records, indent=2) + "\n")
    return records
