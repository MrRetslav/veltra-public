#!/usr/bin/env python3
"""Apex experiment C: how much wall time the LM head takes on hf30 in decode: unscreened int8 head, 4-bit screened head, 2-bit screened head.
Uses the engine's own per-part profile (LB_PROFILE); decode tok/s comes from separate runs without profiling. Configurations are interleaved.

  python3 benchmarks/apex_proto/head_profile.py --gen bin/veltra-gen --model M.vlt --out head.json
"""
import argparse
import json
import os
import re
import statistics
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))


def encode(obj):
    path = a.out + ".enc.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
    p = subprocess.run([a.gen, "--model", a.model, "--encode-file", path], capture_output=True, text=True, check=True)
    os.remove(path)
    return json.loads(p.stdout.strip().splitlines()[-1])


ids = ",".join(map(str, encode({"messages": [{"role": "user", "content": "Explain in detail how a four-stroke combustion engine works."}], "thinking": 0})))
CONF = {"unscreened": [], "screen4": ["--screen-head", "1"], "screen2": ["--screen-head", "2"]}
rows = {k: {"tps": [], "decode_s": [], "head_ms": [], "gen": []} for k in CONF}
for _ in range(a.reps):
    for name, flags in CONF.items():
        for prof in (False, True):
            env = dict(os.environ)
            if prof:
                env["VELTRA_PROFILE"] = "1"
            p = subprocess.run(["nice", "-n", "10", a.gen, "--model", a.model, "--threads", "8", "--ctx", "2048", "--ids", ids, "--n", "128", "--spec", "0"] + flags,
                               capture_output=True, text=True, env=env, check=True)
            s = json.loads([x for x in p.stdout.splitlines() if x.startswith("{")][-1])["stats"]
            if not prof:
                rows[name]["tps"].append(s["tokens_per_s"])
                continue
            line = [x for x in p.stderr.splitlines() if x.startswith("prof ms total")][-1]
            rows[name]["head_ms"].append(float(re.search(r" head ([0-9.]+) norm", line).group(1)))
            rows[name]["decode_s"].append(s["decode_s"])
            rows[name]["gen"].append(s["generated_tokens"])
out = {"model": os.path.basename(a.model), "reps": a.reps, "load": os.getloadavg()[0], "configs": {}}
for name, r in rows.items():
    gen = statistics.median(r["gen"])
    head_per_token = statistics.median(r["head_ms"]) / gen
    token_ms_profiled = statistics.median(r["decode_s"]) * 1000 / gen
    out["configs"][name] = {"decode_tok_s": round(statistics.median(r["tps"]), 2), "decode_all": [round(x, 2) for x in r["tps"]],
                            "generated_tokens": gen, "head_ms_per_token": round(head_per_token, 2),
                            "token_ms_while_profiling": round(token_ms_profiled, 2), "head_share_pct": round(100 * head_per_token / token_ms_profiled, 1)}
    c = out["configs"][name]
    print("%-11s decode %6.2f tok/s %s | head %.2f ms of %.2f ms per token = %.1f%%" % (name, c["decode_tok_s"], c["decode_all"], c["head_ms_per_token"],
                                                                                    c["token_ms_while_profiling"], c["head_share_pct"]), flush=True)
with open(a.out, "w") as f:
    json.dump(out, f, indent=2)
