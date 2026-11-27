#!/usr/bin/env python3
"""Apex experiment A: the size of the union of the routed experts over m consecutive tokens, per layer, on hf30 (teacher-forced on given text,
routing dump of the engine: LB_DUMP_LAYERS writes per layer and token the K expert ids). Reports U_m for m = 1..5.

  python3 benchmarks/apex_proto/expert_overlap.py --gen bin/veltra-gen --model M.vlt --work DIR --out overlap.json
"""
import argparse
import json
import os
import random
import struct
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=400)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "eval_public"))
from reference_solutions import REFERENCE  # noqa: E402

os.makedirs(a.work, exist_ok=True)
K, LAYERS, EXPERTS = 8, 48, 128


def encode(text):
    path = os.path.join(a.work, "enc.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": text}) + "\n")
    p = subprocess.run([a.gen, "--model", a.model, "--encode-file", path], capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


def read(path, n=9000):
    with open(os.path.join(ROOT, path), encoding="utf-8", errors="replace") as f:
        return f.read()[:n]


tasks = [json.loads(x) for x in open(os.path.join(ROOT, "benchmarks", "eval_public", "tasks.jsonl"), encoding="utf-8")]
rng = random.Random(3)
math_text = "".join("%s\nStep by step: we compute the parts and add them.\nAnswer: %s\n\n" % (t["messages"][0]["content"], t["check"]["value"])
                    for t in tasks if t["cat"] == "math")
json_text = "".join("%s\n%s\n\n" % (t["messages"][-1]["content"], json.dumps(t["check"]["value"])) for t in tasks if t["cat"] == "json")
texts = {"python": "\n\n".join(REFERENCE), "math": math_text, "json": json_text, "english": read("docs/WINDOWS_BUILD.md"),
         "dutch": read("docs/UNPUBLISHED_NOTES.md")}
out = {"model": os.path.basename(a.model), "k": K, "experts": EXPERTS, "layers": LAYERS, "texts": {}}
for name, text in texts.items():
    ids = encode(text)[: a.max_tokens]
    T = len(ids)
    prefix = os.path.join(a.work, "dump_" + name)
    for s in (".x", ".xa", ".e"):
        if os.path.exists(prefix + s):
            os.remove(prefix + s)
    env = dict(os.environ, VELTRA_DUMP_LAYERS=prefix)
    subprocess.run(["nice", "-n", "10", a.gen, "--model", a.model, "--threads", "8", "--ctx", "2048", "--ids", ",".join(map(str, ids)),
                    "--dump-logits", os.path.join(a.work, "logits.bin")], check=True, env=env, capture_output=True)
    raw = open(prefix + ".e", "rb").read()
    rec = 4 * K + 4
    assert len(raw) == rec * T * LAYERS, "unexpected routing dump size %d for %d tokens" % (len(raw), T)
    layers = []
    for l in range(LAYERS):
        rows = [set(struct.unpack_from("<%di" % K, raw, rec * (l * T + t))) for t in range(T)]
        assert all(len(r) == K and min(r) >= 0 and max(r) < EXPERTS for r in rows)
        layers.append(rows)
    res = {"tokens": T}
    for m in range(1, 6):
        vals = [len(set().union(*layers[l][t:t + m])) for l in range(LAYERS) for t in range(0, T - m + 1)]
        res["U%d" % m] = round(sum(vals) / len(vals), 2)
        res["U%d_p90" % m] = sorted(vals)[int(0.9 * len(vals))]
    # control: the same routing sets with the token order shuffled (no locality between neighbours)
    order = list(range(T))
    rng.shuffle(order)
    res["U3_shuffled"] = round(sum(len(layers[l][order[t]] | layers[l][order[t + 1]] | layers[l][order[t + 2]])
                                   for l in range(LAYERS) for t in range(T - 2)) / (LAYERS * (T - 2)), 2)
    out["texts"][name] = res
    print("%-8s T=%4d  U1 %.2f  U2 %.2f  U3 %.2f  U4 %.2f  U5 %.2f | U3 shuffled %.2f" % (name, T, res["U1"], res["U2"], res["U3"], res["U4"],
                                                                                         res["U5"], res["U3_shuffled"]), flush=True)
    for s in (".x", ".xa", ".e"):
        if os.path.exists(prefix + s):
            os.remove(prefix + s)
os.remove(os.path.join(a.work, "logits.bin"))
mean = {k: round(sum(v[k] for v in out["texts"].values()) / len(out["texts"]), 2) for k in ("U1", "U2", "U3", "U4", "U5")}
out["mean"] = mean
# bytes per verify pass with the measured unions: 572 MB shared (attention, screened head, router) + 85.3 MB per expert-in-every-layer
out["byte_model_mb_per_pass"] = {k: round(572 + 85.3 * v) for k, v in mean.items()}
print("mean", json.dumps(mean), "| MB per pass", json.dumps(out["byte_model_mb_per_pass"]))
with open(a.out, "w") as f:
    json.dump(out, f, indent=2)
