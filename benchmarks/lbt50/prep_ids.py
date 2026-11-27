#!/usr/bin/env python3
"""Calibration (fit) and held-out token files for the LBT-50 probes. Sources are repository text only; none of it is evaluation material.

  python3 benchmarks/lbt50/prep_ids.py --gen bin/veltra-gen --model M.vlt --out DIR [--tokens 300]
"""
import argparse
import json
import os
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tokens", type=int, default=300)
a = ap.parse_args()
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
os.makedirs(a.out, exist_ok=True)
# (name, file, character offset) - fit and held-out use different files
FIT = [("fit_en", "docs/qk_format.md", 0), ("fit_code", "engine/chat_template.cpp", 0), ("fit_nl", "docs/UNPUBLISHED_NOTES.md", 0),
       ("fit_py", "python/lbt_stream.py", 0), ("fit_en2", "docs/WINDOWS_BUILD.md", 0), ("fit_cpp", "engine/search_verifier.cpp", 3000)]
HELD = [("held_en", "docs/bitplane_spec.md", 0), ("held_code", "engine/platform.cpp", 0), ("held_nl", "docs/UNPUBLISHED_NOTES.md", 0)]
for name, path, off in FIT + HELD:
    with open(os.path.join(ROOT, path), encoding="utf-8", errors="replace") as f:
        text = f.read()[off: off + a.tokens * 12]
    req = os.path.join(a.out, "enc.jsonl")
    with open(req, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": text}) + "\n")
    p = subprocess.run([a.gen, "--model", a.model, "--encode-file", req], capture_output=True, text=True, check=True)
    ids = json.loads(p.stdout.strip().splitlines()[-1])[: a.tokens]
    assert len(ids) >= a.tokens * 3 // 4, (name, len(ids))
    open(os.path.join(a.out, name + ".ids"), "w").write(",".join(map(str, ids)))
    print(name, len(ids), "tokens from", path)
os.remove(os.path.join(a.out, "enc.jsonl"))
