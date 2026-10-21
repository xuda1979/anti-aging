#!/usr/bin/env python3
"""Fetch only ~7.6 MB of free public processed data; validate pinned SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

FILES = {
    "GSE113957_fpkm.txt.gz": ("suppl", "999c9d720ff7f0dfe59675f347803d3a0181231313aace2f8406f08528aa123d"),
    "GSE113957-GPL16791_series_matrix.txt.gz": ("matrix", "3144bef9ef61ca818e16aaf9a7a70e68b082a70a81f5737517cd009d307ea90a"),
    "GSE113957-GPL18573_series_matrix.txt.gz": ("matrix", "1764a9926545a7c4aecc68f423e9338245e859dfe14eb38a8f4bfb88fb0c80ec"),
}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="data/raw"); args=parser.parse_args()
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True); manifest=[]
    for filename,(folder,expected) in FILES.items():
        path=output/filename
        url=f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE113nnn/GSE113957/{folder}/{filename}"
        if not path.exists():
            with urllib.request.urlopen(url, timeout=120) as response:
                data=response.read()
            if hashlib.sha256(data).hexdigest()!=expected:
                raise ValueError(f"Source content changed: {url}; review a new data version before use")
            path.write_bytes(data)
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected: raise ValueError(f"Hash mismatch for {filename}")
        manifest.append({"filename":filename,"url":url,"sha256":actual,"bytes":path.stat().st_size})
        print(f"Verified {filename}: {path.stat().st_size} bytes")
    (output/"source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")

if __name__=="__main__": main()
