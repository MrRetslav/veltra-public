#!/usr/bin/env python3
"""Merges the interleaved rounds of speed_battle.sh (speed_<engine>_r<N>.json, rss_<engine>_r<N>.txt) into one speed_<engine>.json (median over
all repetitions of all rounds) and rss_<engine>.txt (largest peak). Directories are searched in the order given; a later directory replaces an
engine that an earlier one already had.

  python3 benchmarks/merge_speed.py --out benchmarks/results/raw DIR [DIR ...]
"""
import argparse
import glob
import json
import os
import re
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("dirs", nargs="+")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
engines = {}
for d in a.dirs:
    found = {}
    for path in sorted(glob.glob(os.path.join(d, "speed_*_r[0-9].json"))):
        name = re.match(r"speed_(.+)_r[0-9]\.json", os.path.basename(path)).group(1)
        found.setdefault(name, []).append(path)
    for name, paths in found.items():
        engines[name] = (d, paths)
for name, (d, paths) in sorted(engines.items()):
    runs = [json.load(open(p, encoding="utf-8")) for p in paths]
    out = {"label": name, "model": runs[0]["model"], "rounds": len(runs), "reps": sum(r["reps"] for r in runs), "max_tokens": runs[0]["max_tokens"],
           "load_start": runs[0]["load_start"], "load_end": runs[-1]["load_end"], "contexts": {}}
    for ctx in runs[0]["contexts"]:
        rows = [r["contexts"][ctx] for r in runs]
        decode = [x for r in rows for x in r["decode_all"]]
        out["contexts"][ctx] = {"prompt_tokens": rows[0]["prompt_tokens"], "completion_tokens": rows[0]["completion_tokens"],
                                "decode_tok_s": round(statistics.median(decode), 2), "decode_all": decode,
                                # prompt rate: the benchmark keeps only the median per round, so this is the median of the round medians
                                "pp_tok_s": round(statistics.median(r["pp_tok_s"] for r in rows), 1)}
    json.dump(out, open(os.path.join(a.out, "speed_%s.json" % name), "w", encoding="utf-8"), indent=2)
    rss = []
    for p in glob.glob(os.path.join(d, "rss_%s_r[0-9].txt" % name)):
        text = open(p).read().strip()
        if text:
            rss.append(int(text))
    if rss:
        open(os.path.join(a.out, "rss_speed_%s.txt" % name), "w").write("%d\n" % max(rss))
    c = out["contexts"]
    print("%-14s rounds %d | decode %s | pp %s | peak RSS %s GB" % (name, len(runs), " / ".join("%.2f" % c[k]["decode_tok_s"] for k in c),
          " / ".join("%.1f" % c[k]["pp_tok_s"] for k in c), "%.2f" % (max(rss) / 1048576) if rss else "?"))
