#!/usr/bin/env python3
"""sanity10: the 10 prompts in scripts/sanity10.json against an OpenAI-compatible chat server (lb_server).

  python scripts/run_sanity10.py --url http://127.0.0.1:8099 [--model M] [--json scripts/sanity10.json]
                                 [--max-tokens 250] [--report out.json]

Greedy (temperature 0), single user message per prompt, POST /v1/chat/completions, stop string <|im_end|>.
Checks: contains_all (values, optional ignore_case), regex (re.search), json_keys (first JSON object in the answer must have the keys).
A closed <think> block is stripped before checking, as in eval_raptor.py. Prints X/10 and the failed ids. Standard library only.
"""
import argparse
import json
import os
import re
import urllib.request

IM_E = "<|im_end|>"


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def visible(text):
    return text.split("</think>")[-1] if "</think>" in text else text


def json_object(text):
    t = re.sub(r"```(?:json)?", "", visible(text)).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except ValueError:
        return None


def check(c, text):
    t = visible(text)
    kind = c["type"]
    if kind == "contains_all":
        fold = c.get("ignore_case", False)
        hay = t.lower() if fold else t
        miss = [v for v in c["values"] if (v.lower() if fold else v) not in hay]
        return not miss, ("missing " + ", ".join(miss)) if miss else "ok"
    if kind == "regex":
        ok = re.search(c["pattern"], t) is not None
        return ok, "ok" if ok else "pattern not found"
    if kind == "json_keys":
        obj = json_object(text)
        if not isinstance(obj, dict):
            return False, "no JSON object"
        miss = [k for k in c["keys"] if k not in obj]
        return not miss, ("missing keys " + ", ".join(miss)) if miss else "ok"
    return False, "unknown check type " + kind


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="<dir>/q30b_n4.lbt")
    ap.add_argument("--json", default=os.path.join(here, "sanity10.json"))
    ap.add_argument("--max-tokens", type=int, default=250)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--report", default="", help="write the full outputs as JSON")
    args = ap.parse_args()

    with open(args.json, encoding="utf-8") as f:
        prompts = json.load(f)
    rows = []
    for p in prompts:
        body = {"model": args.model, "messages": [{"role": "user", "content": p["prompt"]}],
                "max_tokens": args.max_tokens, "temperature": 0, "stop": [IM_E]}
        try:
            data = post(args.url.rstrip("/") + "/v1/chat/completions", body, args.timeout)
            text = data["choices"][0]["message"]["content"] or ""
            ok, why = check(p["check"], text)
            err = None
        except (OSError, ValueError, KeyError, IndexError) as e:
            text, ok, why, err = "", False, "request failed: %s" % e, str(e)
        rows.append({"id": p["id"], "pass": bool(ok), "reason": why, "output": text, "error": err})
        print("%-7s %s  %s" % (p["id"], "PASS" if ok else "FAIL", why))

    npass = sum(r["pass"] for r in rows)
    failed = [r["id"] for r in rows if not r["pass"]]
    print("SANITY10: %d/%d passed; failed: %s" % (npass, len(rows), ", ".join(failed) if failed else "none"))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"url": args.url, "model": args.model, "max_tokens": args.max_tokens, "passed": npass,
                       "total": len(rows), "results": rows}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
