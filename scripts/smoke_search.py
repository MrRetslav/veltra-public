#!/usr/bin/env python3
"""API smoke test of test-time search and the disconnect log against a running veltra-server (loopback test port only).

  python3 scripts/smoke_search.py --port 8097 --log <VELTRA_DATA_DIR>/veltra.log
"""
import argparse
import json
import os
import socket
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8097)
ap.add_argument("--log", required=True)
a = ap.parse_args()
assert a.port != 8099 and a.port != 8098, "never the live ports"
BASE = "http://127.0.0.1:%d" % a.port
MODEL = json.loads(urllib.request.urlopen(BASE + "/v1/models", timeout=10).read())["data"][0]["id"]
fails = []


def check(ok, what, detail=""):
    print("%s %s %s" % ("ok  " if ok else "FAIL", what, detail))
    if not ok:
        fails.append(what)


def body(content, **kw):
    return dict(model=MODEL, messages=[{"role": "user", "content": content}], temperature=0, **kw)


def post(b, timeout=600):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode()


KEYS = ("branches_evaluated", "valid_branches", "consensus_ratio", "pruned_branches", "winner", "total_tokens", "branches")
j = json.loads(post(body("What is 12*12? Answer with the number.", search_n=4, max_tokens=48)))
s = j["timings"].get("search", {})
check(all(k in s for k in KEYS) and s.get("branches_evaluated") == 4 and len(s["branches"]) == 4, "search_n=4 metadata",
      "valid %s consensus %s winner %s" % (s.get("valid_branches"), s.get("consensus_ratio"), s.get("winner")))
first = j["choices"][0]["message"]["content"]
check(bool(first.strip()) and "<think>" not in first, "search answer text", repr(first[:60]))
again = json.loads(post(body("What is 12*12? Answer with the number.", search_n=4, max_tokens=48)))
check(again["choices"][0]["message"]["content"] == first, "same seed gives the same answer")

raw = post(body("Name two colours.", search_n=4, max_tokens=32, stream=True))
chunks = [json.loads(x[6:]) for x in raw.splitlines() if x.startswith("data: {")]
text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c.get("choices"))
check(bool(text.strip()) and any("search" in c.get("timings", {}) for c in chunks) and "[DONE]" in raw, "streamed search reply", repr(text[:60]))

j = json.loads(post(body('Return a JSON object with the keys "city" and "country" for the capital of France. Only JSON.', search_n=4,
                         max_tokens=64, response_format={"type": "json_object"})))
out = j["choices"][0]["message"]["content"].strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
s = j["timings"]["search"]
try:
    parsed = isinstance(json.loads(out), dict)
except ValueError:
    parsed = False
check(parsed or s["valid_branches"] == 0, "response_format gate: winner parses as JSON (or no branch was valid)",
      "valid %d/%d %r" % (s["valid_branches"], s["branches_evaluated"], out[:60]))

j = json.loads(post(body("Say hello.", max_tokens=16)))
check("search" not in j["timings"], "search_n absent -> normal decoder")


def abort(b, wait_for_bytes, delay):
    payload = json.dumps(b).encode()
    c = socket.create_connection(("127.0.0.1", a.port), timeout=120)
    c.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(payload) + payload)
    got = 0
    while got < wait_for_bytes:
        got += len(c.recv(4096))
    time.sleep(delay)
    c.close()


def logged(needle, since, seconds=120):
    end = time.time() + seconds
    while time.time() < end:
        if os.path.exists(a.log):
            with open(a.log, encoding="utf-8", errors="replace") as f:
                f.seek(since)
                hit = [x for x in f.read().splitlines() if needle in x]
            if hit:
                return hit[-1]
        time.sleep(1)
    return ""


size = os.path.getsize(a.log) if os.path.exists(a.log) else 0
abort(body("Write a long story about the sea.", max_tokens=200, stream=True), 1500, 0)
line = logged("client disconnected after", size)
check(bool(line), "aborted stream is logged in veltra.log", line)

size = os.path.getsize(a.log) if os.path.exists(a.log) else 0
t = time.time()
abort(body("Write a long story about the mountains.", search_n=8, max_tokens=400), 0, 6)
line = logged("during test-time search", size)
check(bool(line), "aborted search cancels all branches and is logged", "%s (after %.0f s)" % (line, time.time() - t))

j = json.loads(post(body("What is 2+2?", max_tokens=16)))
check(bool(j["choices"][0]["message"]["content"].strip()), "server still answers after both aborts")
print("SMOKE %s: %d failures" % ("FAILED" if fails else "OK", len(fails)))
sys.exit(1 if fails else 0)
