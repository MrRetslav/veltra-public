#!/usr/bin/env python3
"""Decision rule for the shadow-expert quality run, fixed before the probe: of the mask variants with at most 5.5 expert reads take the one with the
lowest held-out KL; run the 160-task set only when that KL is below the one of uniform --moe-k 6 (6 expert reads, so more bytes).
  python3 pick_shadow.py shadow_kl.json        -> prints "<R> <mask file name>" or exits 1
"""
import json
import sys

READS = lambda r, s: r + (8 - r) * (2 * s * 2.625 + s * 4.625) / (2 * 768 * 2.625 + 768 * 3.66)
v = {x["variant"]: x for x in json.load(open(sys.argv[1]))["variants"]}
cand = []
for name, x in v.items():
    if name.startswith("s:"):
        _, r, f = name.split(":", 2)
        s = int(f.rsplit("_s", 1)[1].split(".")[0])
        if READS(int(r), s) <= 5.5:
            cand.append((x["kl"], READS(int(r), s), int(r), f))
kl, reads, r, f = min(cand)
print("best at <=5.5 reads: s:%d:%s reads %.2f KL %.4f; uniform moe-k 6 KL %.4f" % (r, f, reads, kl, v["k:6"]["kl"]), file=sys.stderr)
if kl >= v["k:6"]["kl"]:
    sys.exit(1)
print(r, f)
