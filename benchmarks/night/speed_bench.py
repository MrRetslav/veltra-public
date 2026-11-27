#!/usr/bin/env python3
"""Server-side speed benchmark that treats every engine alike (veltra-server, llama-server): /v1/chat/completions, greedy, one request at a time.

Per context size (about 0 / 500 / 2000 prompt tokens): prompt-processing rate (pp) and decode rate, read from the server's own timings and also
from the client clock (time to first streamed token is not used, so both engines report through their `timings`). Median of --reps. Every request
starts with a different random line, so no prefix cache can skip the prompt.

  python3 speed_bench.py --url http://127.0.0.1:8097 --out speed.json [--reps 3] [--n 128] [--label X]
"""
import argparse
import json
import os
import random
import statistics
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8097")
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--label", default="")
a = ap.parse_args()
model = json.loads(urllib.request.urlopen(a.url + "/v1/models", timeout=30).read())["data"][0]["id"]

WORDS = ("harbour lantern copper meadow signal orchard granite ribbon compass willow ember quarry sailor timber glacier violet anchor "
         "thimble falcon pepper marble canyon beacon saddle cedar pebble kettle monsoon barley cobble").split()


def filler(rng, n_sent):
    out = []
    for _ in range(n_sent):
        w = rng.sample(WORDS, 7)
        out.append("The %s near the %s kept a %s beside the %s while a %s carried the %s past the %s." % tuple(w))
    return " ".join(out)


def request(prompt):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": a.n, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.loads(r.read())
    return j, time.time() - t0


ASK = ("\n\nWrite a long, detailed travel diary of at least four hundred words about a walk through an old port town. Describe the streets, "
       "the people, the weather and the food. Do not summarise the text above.")
out = {"label": a.label, "model": model, "reps": a.reps, "max_tokens": a.n, "load_start": os.getloadavg()[0], "contexts": {}}
request("Say hello.")   # warm-up: pages in the weights
# about 22 tokens per filler sentence
sizes = {"ctx0": 0, "ctx500": 20, "ctx2000": 88}
rows = {k: [] for k in sizes}
for rep in range(a.reps):   # contexts interleaved so that load drift on a shared box hits all of them alike
    for name, ns in sizes.items():
        rng = random.Random(1000 * rep + ns + 7)
        head = "Log entry %d.\n" % rng.randrange(10 ** 8)
        j, wall = request(head + filler(rng, ns) + ASK)
        u, t = j.get("usage", {}), j.get("timings", {})
        n_prompt, n_out = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        if "predicted_per_second" in t:    # llama-server / ik_llama.cpp
            decode, pp, prefill_s = t["predicted_per_second"], t["prompt_per_second"], t["prompt_ms"] / 1000.0
        else:                              # veltra-server
            decode, prefill_s = t["tokens_per_second"], t["prefill_s"]
            pp = (n_prompt - t.get("cached_tokens", 0)) / prefill_s if prefill_s > 0 else 0
        rows[name].append({"decode": decode, "pp": pp, "prefill_s": prefill_s, "prompt_tokens": n_prompt, "completion_tokens": n_out, "wall_s": wall})
for name, rs in rows.items():
    med = lambda k: round(statistics.median(r[k] for r in rs), 2)
    out["contexts"][name] = {"prompt_tokens": med("prompt_tokens"), "completion_tokens": med("completion_tokens"),
                             "decode_tok_s": med("decode"), "pp_tok_s": med("pp"), "prefill_s": med("prefill_s"), "wall_s": med("wall_s"),
                             "decode_all": [round(r["decode"], 2) for r in rs], "completion_all": [r["completion_tokens"] for r in rs]}
    c = out["contexts"][name]
    print("%-7s prompt %5s out %4s | decode %6.2f tok/s %s | pp %7.1f tok/s" % (name, c["prompt_tokens"], c["completion_tokens"],
          c["decode_tok_s"], c["decode_all"], c["pp_tok_s"]), flush=True)
out["load_end"] = os.getloadavg()[0]
with open(a.out, "w") as f:
    json.dump(out, f, indent=2)
