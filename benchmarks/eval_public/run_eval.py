#!/usr/bin/env python3
"""Runs the 160-task public eval against any OpenAI-compatible server (veltra-server, llama-server) with greedy decoding. Standard library only.

  python3 run_eval.py --url http://127.0.0.1:8097 --report out.json [--only math,code,json,long] [--model ID] [--label NAME]
  python3 run_eval.py --validate          # proves the checkers and the sandboxed unit tests with reference answers (no model)

Code tasks execute model-written Python only inside sandbox.py (bwrap or systemd-run); without a working sandbox the run is refused.
"""
import argparse
import json
import os
import re
import secrets
import sys
import time
import urllib.request

import sandbox

HERE = os.path.dirname(os.path.abspath(__file__))


def load_tasks():
    with open(os.path.join(HERE, "tasks.jsonl"), encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def visible(text):
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()


def check_number(want, text):
    t = visible(text).replace(" ", "").replace("**", "").replace("$", "")
    m = re.findall(r"(?i)answer\s*(?:is)?\s*[:=]?\s*(-?\d[\d,]*(?:\.\d+)?)", t) or re.findall(r"-?\d[\d,]*(?:\.\d+)?", t)
    if not m:
        return False, "no number"
    got = m[-1].replace(",", "")
    try:
        ok = abs(float(got) - want) < 1e-9
    except ValueError:
        return False, "unparsable " + got
    return ok, "ok" if ok else "got %s, expected %s" % (got, want)


def extract_code(text):
    t = visible(text)
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", t, re.S)
    if blocks:
        return max(blocks, key=len)
    i = min([x for x in (t.find("def "), t.find("import "), t.find("from ")) if x >= 0], default=-1)
    return t[i:] if i >= 0 else ""


def check_python(tests, text, code=None):
    code = extract_code(text) if code is None else code
    if "def " not in code:
        return False, "no function found"
    nonce = "PASS-" + secrets.token_hex(12)
    # the tests run in a second namespace-free step: the model code cannot know the nonce, so only reaching the last line prints it
    program = code + "\n\n" + tests + "\nprint(%r)\n" % nonce
    rc, out, err = sandbox.run_untrusted(program)
    if rc == 0 and nonce in out:
        return True, "ok"
    if rc is None:
        return False, "timeout"
    return False, (err.strip().splitlines() or ["exit %s" % rc])[-1][:160]


def same_json(want, got):
    if isinstance(want, bool) or isinstance(got, bool):
        return want is got
    if isinstance(want, (int, float)):
        return isinstance(got, (int, float)) and abs(want - got) < 1e-9
    if isinstance(want, str):
        return isinstance(got, str) and got.strip().casefold() == want.strip().casefold()
    if isinstance(want, list):
        if not isinstance(got, list) or len(want) != len(got):
            return False
        if all(isinstance(x, str) for x in want):   # name lists: order is not part of the task
            return all(isinstance(x, str) for x in got) and sorted(x.strip().casefold() for x in got) == sorted(x.strip().casefold() for x in want)
        return all(same_json(a, b) for a, b in zip(want, got))
    if isinstance(want, dict):
        return isinstance(got, dict) and set(want) == set(got) and all(same_json(v, got[k]) for k, v in want.items())
    return want == got


def check_json(want, text):
    t = visible(text)
    m = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", t, re.S)
    if m:
        t = m.group(1).strip()
    try:
        got = json.loads(t)
    except ValueError as e:
        return False, "not exactly one JSON value (%s)" % str(e)[:60]
    ok = same_json(want, got)
    return ok, "ok" if ok else "wrong schema or value"


def check_value(want, text):
    t = visible(text).replace("**", "").replace(",", "")
    ok = re.search(r"(?<![\w-])" + re.escape(want.replace(",", "")) + r"(?![\w-])", t, re.I) is not None and len(t) <= 300
    return ok, "ok" if ok else "expected value not found"


def check(task, text):
    c = task["check"]
    if c["type"] == "number":
        return check_number(c["value"], text)
    if c["type"] == "python":
        return check_python(c["tests"], text)
    if c["type"] == "json":
        return check_json(c["value"], text)
    return check_value(c["value"], text)


def validate():
    from code_tasks import CODE_TASKS
    from reference_solutions import REFERENCE
    print(sandbox.self_test())
    tasks = load_tasks()
    bad = []
    code_i = 0
    for t in tasks:
        c = t["check"]
        if c["type"] == "python":
            ok, why = check_python(c["tests"], "```python\n%s\n```" % REFERENCE[code_i])
            wrong, _ = check_python(c["tests"], "```python\ndef %s(*a, **k):\n    return None\n```" % CODE_TASKS[code_i][0].split("(")[0])
            code_i += 1
        elif c["type"] == "number":
            ok, why = check(t, "so...\nAnswer: %s" % c["value"])
            wrong, _ = check(t, "Answer: %s" % (c["value"] + 1))
        elif c["type"] == "json":
            ok, why = check(t, "```json\n%s\n```" % json.dumps(c["value"]))
            wrong = check(t, "Here it is: " + json.dumps(c["value"]))[0] or check(t, "{}")[0]
        else:
            ok, why = check(t, str(c["value"]))
            wrong, _ = check(t, "unknown")
        if not ok or wrong:
            bad.append((t["id"], why, "accepts a wrong answer" if wrong else ""))
    print("validate: %d tasks, %d reference code solutions, %d problems" % (len(tasks), code_i, len(bad)))
    for b in bad:
        print("  ", b)
    return 1 if bad else 0


def ask(a, model, task):
    body = {"model": model, "messages": task["messages"], "temperature": 0, "max_tokens": task["max_tokens"], "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=a.timeout) as r:
        j = json.loads(r.read().decode("utf-8", errors="replace"))
    return j, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8097")
    ap.add_argument("--report", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--regrade", default="", help="re-run the checkers on the outputs stored in an earlier report")
    a = ap.parse_args()
    if a.validate:
        sys.exit(validate())
    tasks = load_tasks()
    if a.only:
        tasks = [t for t in tasks if t["cat"] in a.only.split(",")]
    if any(t["cat"] == "code" for t in tasks):
        print(sandbox.self_test(), flush=True)
    old = {}
    if a.regrade:
        old = {r["id"]: r for r in json.load(open(a.regrade, encoding="utf-8"))["results"]}
    model = a.model
    if not model and not a.regrade:
        model = json.loads(urllib.request.urlopen(a.url.rstrip("/") + "/v1/models", timeout=30).read())["data"][0]["id"]
    rows = []
    started = time.time()
    for n, t in enumerate(tasks):
        try:
            if a.regrade:
                prev = old[t["id"]]
                text, wall, ntok, ptok, tps = prev["output"], prev["wall_s"], prev["completion_tokens"], prev["prompt_tokens"], prev["tok_s"]
            else:
                j, wall = ask(a, model, t)
                text = j["choices"][0]["message"]["content"] or ""
                u, tm = j.get("usage", {}), j.get("timings", {})
                ntok, ptok = u.get("completion_tokens", 0), u.get("prompt_tokens", 0)
                tps = tm.get("tokens_per_second") or tm.get("predicted_per_second") or 0
            ok, why = check(t, text)
            err = None
        except (OSError, ValueError, KeyError, IndexError) as e:
            text, wall, ntok, ptok, tps, ok, why, err = "", 0, 0, 0, 0, False, "request failed: %s" % str(e)[:200], str(e)[:300]
        rows.append({"id": t["id"], "cat": t["cat"], "pass": bool(ok), "reason": why, "output": text, "wall_s": round(wall, 2),
                     "completion_tokens": ntok, "prompt_tokens": ptok, "tok_s": round(float(tps), 2), "error": err})
        if (n + 1) % 20 == 0 or n + 1 == len(tasks):
            print("%3d/%d pass %d (%.0f s)" % (n + 1, len(tasks), sum(r["pass"] for r in rows), time.time() - started), flush=True)
        if a.report and ((n + 1) % 10 == 0 or n + 1 == len(tasks)):
            cats = {}
            for r in rows:
                c = cats.setdefault(r["cat"], [0, 0])
                c[0] += r["pass"]
                c[1] += 1
            with open(a.report, "w", encoding="utf-8") as f:
                json.dump({"label": a.label, "model": model, "complete": n + 1 == len(tasks), "passed": sum(r["pass"] for r in rows), "total": len(rows),
                           "by_category": cats, "errors": sum(r["error"] is not None for r in rows), "wall_total_s": round(time.time() - started, 1),
                           "completion_tokens": sum(r["completion_tokens"] for r in rows), "results": rows}, f, indent=1, ensure_ascii=False)
    npass = sum(r["pass"] for r in rows)
    cats = {}
    for r in rows:
        c = cats.setdefault(r["cat"], [0, 0])
        c[0] += r["pass"]
        c[1] += 1
    print("TOTAL %d/%d = %.1f%% | %s | errors %d" % (npass, len(rows), 100.0 * npass / max(1, len(rows)),
          ", ".join("%s %d/%d" % (k, v[0], v[1]) for k, v in cats.items()), sum(r["error"] is not None for r in rows)))


if __name__ == "__main__":
    main()
