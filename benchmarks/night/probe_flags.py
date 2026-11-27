#!/usr/bin/env python3
"""Phase-1 timing probe of the CPU accelerator flags through veltra-gen (no server). One process per run, median of --reps.

  python3 benchmarks/night/probe_flags.py --gen bin/veltra-gen --model M.vlt --out probe.json [--reps 3] [--only base,moek6,...]
"""
import argparse
import json
import os
import re
import statistics
import struct
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--only", default="")
ap.add_argument("--extra", default="", help="flags added to every run")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")


def tensor_table(path):
    with open(path, "rb") as f:
        _, _, n, meta_off, meta_len, table_off, _, _, _, _ = struct.unpack("<8sIIQQQQQII", f.read(64))
        f.seek(meta_off)
        meta = json.loads(f.read(meta_len))
        f.seek(table_off)
        rows = {}
        for _ in range(n):
            name, _, _, _, _, _, _, _, nbytes = struct.unpack("<96sII4QQQ8x", f.read(160))
            rows[name.rstrip(b"\0").decode()] = nbytes
    return meta, rows


def byte_budget():
    """Weight bytes streamed per decoded token, from the file's own tensor sizes: dense parts, head, and one routed expert per layer."""
    meta, rows = tensor_table(a.model)
    expert = sum(v for k, v in rows.items() if re.match(r"blk\.\d+\.e\d+\.", k))
    layers_moe = len({k.split(".")[1] for k in rows if re.match(r"blk\.\d+\.moe_gate$", k)})
    ne = meta.get("num_experts", 0)
    per_expert_layer = expert / (ne * layers_moe) if ne and layers_moe else 0
    dense = sum(v for k, v in rows.items() if k.startswith("blk.") and not re.match(r"blk\.\d+\.e\d+\.", k))
    head = rows.get("output", rows.get("tok_embeddings", 0))
    k = meta.get("num_experts_per_tok", 0)
    return {"top_k": k, "moe_layers": layers_moe, "expert_set_mb": round(per_expert_layer * layers_moe / 1e6, 1),
            "dense_per_token_mb": round(dense / 1e6, 1), "head_mb_full": round(head / 1e6, 1),
            "per_token_mb_by_k": {str(i): round((dense + head + i * per_expert_layer * layers_moe) / 1e6, 1) for i in (8, 7, 6, 5) if i <= k}}


def encode(obj):
    path = a.out + ".enc.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
    p = subprocess.run([a.gen, "--model", a.model, "--encode-file", path], capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


def run(ids, flags, env=None):
    cmd = ["nice", "-n", "10", a.gen, "--model", a.model, "--threads", "8", "--ctx", "4096", "--ids", ",".join(map(str, ids)), "--n", str(a.n)] + flags + a.extra.split()
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, **(env or {})))
    if p.returncode != 0:
        raise SystemExit("failed: %s\n%s" % (" ".join(flags), p.stderr[-1200:]))
    return json.loads([x for x in p.stdout.splitlines() if x.startswith("{")][-1]), p.stderr


text = open(os.path.join(ROOT, "docs", "VELTRA_V3_LOG.md"), encoding="utf-8", errors="replace").read()[:1700]
prose = encode({"messages": [{"role": "user", "content": "Read these engineering notes:\n\n" + text + "\n\nExplain the main findings in your own words."}], "thinking": 0})
repeat = encode({"messages": [{"role": "user", "content": "Repeat the following text exactly, word for word:\n\n" + text[:900]}], "thinking": 0})

CONFIGS = {
    "base": ["--spec", "0"],
    "moek7": ["--spec", "0", "--moe-k", "7"],
    "moek6": ["--spec", "0", "--moe-k", "6"],
    "moek5": ["--spec", "0", "--moe-k", "5"],
    "screen": ["--spec", "0", "--screen-head", "1"],
    "chunk512": ["--spec", "0", "--prefill-chunk", "512"],
    "teal20": ["--spec", "0", "--teal-sparsity", "0.2"],
    "teal30": ["--spec", "0", "--teal-sparsity", "0.3"],
    "teal40": ["--spec", "0", "--teal-sparsity", "0.4"],
    "lookup_default": [],
    "lookup4": ["--lookup-n", "4"],
    "lookup8_repeat": ["--lookup-n", "8"],
    "base_repeat": ["--spec", "0"],
    "paretomax": ["--pareto-max", "1"],
}
only = [x for x in a.only.split(",") if x]
out = {"model": os.path.basename(a.model), "reps": a.reps, "n": a.n, "prompt_tokens": len(prose), "repeat_prompt_tokens": len(repeat),
       "load_start": os.getloadavg()[0], "bytes": byte_budget(), "runs": {}}
print("bytes", json.dumps(out["bytes"]), flush=True)
reference = {}
# interleave configurations across repetitions so that load drift on the shared box hits all of them alike
rows = {name: [] for name in CONFIGS if not only or name in only}
for rep in range(a.reps):
    for name in rows:
        ids = repeat if name.endswith("repeat") else prose
        j, err = run(ids, CONFIGS[name])
        rows[name].append(j)
for name, rs in rows.items():
    s = [r["stats"] for r in rs]
    tok = rs[0]["tokens"]
    kind = "repeat" if name.endswith("repeat") else "prose"
    reference.setdefault(kind, tok if name in ("base", "base_repeat") else None)
    res = {"flags": " ".join(CONFIGS[name]), "decode_tok_s": round(statistics.median(x["tokens_per_s"] for x in s), 2),
           "prefill_tok_s": round(len(repeat if kind == "repeat" else prose) / statistics.median(x["prefill_s"] for x in s), 1),
           "drafted": s[0]["drafted"], "accepted": s[0]["accepted"], "generated": s[0]["generated_tokens"], "finish": s[0]["finish_reason"], "tokens": tok}
    out["runs"][name] = res
base = {"prose": out["runs"].get("base", {}).get("tokens"), "repeat": out["runs"].get("base_repeat", {}).get("tokens")}
for name, res in out["runs"].items():
    ref = base["repeat" if name.endswith("repeat") else "prose"]
    tok = res.pop("tokens")
    res["same_tokens_as_base"] = None if ref is None else tok == ref
    if ref is not None and tok != ref:
        res["first_difference_at"] = next((i for i, (x, y) in enumerate(zip(tok, ref)) if x != y), min(len(tok), len(ref)))
    print(name, json.dumps(res), flush=True)

# separate runs (they cost time themselves, so never used for tok/s): TEAL block statistics and the per-part profile by k
for name in ("teal20", "teal30", "teal40"):
    if name in rows:
        _, err = run(prose, CONFIGS[name], env={"VELTRA_TEAL_STATS": "1"})
        line = [x for x in err.splitlines() if x.startswith("teal P=")]
        out["runs"][name]["blocks"] = line[-1] if line else "no statistics line"
        print(name, out["runs"][name]["blocks"], flush=True)
for name in ("base", "moek6", "moek5"):
    if name in rows:
        _, err = run(prose, CONFIGS[name], env={"VELTRA_PROFILE": "1"})
        line = [x for x in err.splitlines() if x.startswith("prof ms total")]
        out["runs"][name]["profile"] = line[-1] if line else ""
        print(name, out["runs"][name]["profile"], flush=True)
out["load_end"] = os.getloadavg()[0]
with open(a.out, "w") as f:
    json.dump(out, f, indent=2)
print("wrote", a.out)
