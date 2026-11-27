#!/usr/bin/env python3
"""Collects raw run files (eval_<run>.json, speed_<run>.json, rss_<run>.txt) into pareto_sweep.json and battle_3way.json. No numbers are typed in
by hand: a missing run stays missing (null) and is listed under "not_measured".

  python3 benchmarks/assemble.py --raw benchmarks/results/raw --manifest benchmarks/results/manifest.json --out benchmarks/results
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
man = json.load(open(a.manifest, encoding="utf-8"))


def load(name):
    path = os.path.join(a.raw, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f) if name.endswith(".json") else f.read().strip()


entries, missing = [], []
for e in man["entries"]:
    r = dict(e)
    ev = load("eval_%s.json" % e["eval"]) if e.get("eval") else None
    if ev and ev.get("complete") and not ev.get("errors"):
        r["quality"] = {"passed": ev["passed"], "total": ev["total"], "pct": round(100.0 * ev["passed"] / ev["total"], 1),
                        "by_category": ev["by_category"], "completion_tokens": ev["completion_tokens"], "wall_total_s": ev["wall_total_s"],
                        "failed": [x["id"] for x in ev["results"] if not x["pass"]]}
    else:
        r["quality"] = None
        if e.get("eval"):
            missing.append("%s: quality (%s)" % (e["id"], "incomplete or request errors" if ev else "no run"))
    sp = load("speed_%s.json" % e["speed"]) if e.get("speed") else None
    r["speed"] = {k: {x: v[x] for x in ("prompt_tokens", "completion_tokens", "decode_tok_s", "pp_tok_s", "decode_all")}
                  for k, v in sp["contexts"].items()} if sp else None
    if sp:
        r["speed_load"] = [sp["load_start"], sp["load_end"]]
    elif e.get("speed"):
        missing.append("%s: speed" % e["id"])
    # peak_rss_gb comes from the speed run (the same short requests for every engine); the 160-task run is reported separately,
    # because server defaults such as llama-server's host prompt cache grow with the number of requests.
    peaks = {}
    for key in ("speed_rss", "eval_rss"):
        v = load("rss_%s.txt" % e[key]) if e.get(key) else None
        peaks[key] = round(int(v) / 1048576.0, 2) if v else None
    r["peak_rss_gb"] = peaks["speed_rss"] or peaks["eval_rss"]
    r["eval_peak_rss_gb"] = peaks["eval_rss"]
    # identical-output check against another run (lossless claims)
    if e.get("same_output_as") and ev:
        other = load("eval_%s.json" % e["same_output_as"])
        if other:
            o = {x["id"]: x["output"] for x in other["results"]}
            same = sum(o.get(x["id"]) == x["output"] for x in ev["results"])
            r["outputs_identical_to"] = {"run": e["same_output_as"], "identical": same, "of": len(ev["results"])}
    entries.append(r)

for group, fname in (("sweep", "pareto_sweep.json"), ("battle", "battle_3way.json")):
    rows = [r for r in entries if group in r.get("groups", [])]
    ids = {r["id"] for r in rows}
    with open(os.path.join(a.out, fname), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"hardware": man["hardware"], "versions": man["versions"], "eval_set": man["eval_set"], "entries": rows,
                   "not_measured": [m for m in missing if m.split(":")[0] in ids] + man.get("not_run", [])}, f, indent=2, ensure_ascii=False)
    print("%s: %d entries" % (fname, len(rows)))
for m in missing:
    print("not measured:", m)
