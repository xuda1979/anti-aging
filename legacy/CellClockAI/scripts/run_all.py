"""Run the exact pilots. No remote model service or credentials are used."""
import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cellclock import empirical, synthetic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["all", "public", "synthetic"], default="all")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--permutations", type=int, default=100)
    a = p.parse_args()
    if a.seeds < 2 or a.permutations < 1:
        p.error("Use at least two simulation seeds and one permutation")
    for folder in ["results", "data/processed", "figures"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    start = time.time()
    if a.only in {"all", "public"}:
        print(json.dumps(empirical.run(ROOT, a.permutations), indent=2))
    if a.only in {"all", "synthetic"}:
        synthetic.run(ROOT, a.seeds)
    metadata = {"python": sys.version, "platform": platform.platform(), "elapsed_seconds": time.time() - start,
                "arguments": vars(a), "packages": {n: importlib.metadata.version(n) for n in
                   ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]}}
    (ROOT / f"results/runtime_{a.only}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"completed in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
