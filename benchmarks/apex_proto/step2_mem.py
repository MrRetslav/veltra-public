#!/usr/bin/env python3
"""Apex step 2.1: (a) the stand-alone bandwidth probe (membw.cpp) and (b) the real engine with and without thread pinning, interleaved.

  python3 benchmarks/apex_proto/step2_mem.py --gen bin/veltra-gen --model M.vlt --out apex_step2_mem.json
"""
import argparse
import json
import os
import statistics
import subprocess
import tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
out = {"load_start": os.getloadavg()[0]}

with tempfile.TemporaryDirectory() as tmp:
    exe = os.path.join(tmp, "membw")
    subprocess.run(["g++", "-O3", "-mavx2", "-std=c++20", "-pthread", os.path.join(HERE, "membw.cpp"), "-o", exe], check=True)
    p = subprocess.run(["nice", "-n", "10", exe, "2.0", "5"], capture_output=True, text=True, check=True)
    out["membw"] = [json.loads(x) for x in p.stdout.splitlines()]
for r in out["membw"]:
    print("%-26s %6.2f GB/s (min %.2f max %.2f)" % (r["config"], r["gb_s_median"], r["gb_s_min"], r["gb_s_max"]), flush=True)


def encode(obj):
    path = a.out + ".enc.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
    p = subprocess.run([a.gen, "--model", a.model, "--encode-file", path], capture_output=True, text=True, check=True)
    os.remove(path)
    return json.loads(p.stdout.strip().splitlines()[-1])


text = open(os.path.join(HERE, "..", "..", "docs", "qk_format.md"), encoding="utf-8", errors="replace").read()[:1500]
ids = ",".join(map(str, encode({"messages": [{"role": "user", "content": "Explain these notes:\n\n" + text}], "thinking": 0})))
PINS = {"unpinned": None, "pin-4+4-ccd": "0,1,2,3,6,7,8,9", "pin-2x4-ccx": "0,1,3,4,6,7,9,10"}
rows = {k: [] for k in PINS}
for _ in range(a.reps):
    for name, pin in PINS.items():
        env = dict(os.environ)
        env.pop("VELTRA_PIN", None)
        if pin:
            env["VELTRA_PIN"] = pin
        p = subprocess.run(["nice", "-n", "10", a.gen, "--model", a.model, "--threads", "8", "--ctx", "2048", "--ids", ids, "--n", "96", "--spec", "0",
                            "--screen-head", "1"], capture_output=True, text=True, env=env, check=True)
        s = json.loads([x for x in p.stdout.splitlines() if x.startswith("{")][-1])["stats"]
        rows[name].append((s["tokens_per_s"], s["prompt_tokens"] / s["prefill_s"], s["generated_tokens"]))
out["engine"] = {k: {"decode_tok_s": round(statistics.median(x[0] for x in v), 2), "decode_all": [round(x[0], 2) for x in v],
                     "prefill_tok_s": round(statistics.median(x[1] for x in v), 1), "generated": v[0][2]} for k, v in rows.items()}
for k, v in out["engine"].items():
    print("engine %-12s decode %6.2f tok/s %s prefill %6.1f (%d tokens)" % (k, v["decode_tok_s"], v["decode_all"], v["prefill_tok_s"], v["generated"]), flush=True)
out["bytes_per_token_note"] = "engine GB/s = tok/s x about 1.24 GB per token (screened head)"
out["load_end"] = os.getloadavg()[0]
with open(a.out, "w") as f:
    json.dump(out, f, indent=2)
