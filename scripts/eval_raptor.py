#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAPTOR quick eval: 20 hard prompts with automatic pass/fail against a running lb_server.

  python scripts/eval_raptor.py --url http://127.0.0.1:PORT [--model /content/hf17] [--max-tokens 250]
                                [--think off|on] [--only F1,T3] [--report out.json] [--mode completions|chat]
  python scripts/eval_raptor.py --self-test        # validates the checkers against a canned fake server (no model needed)

Groups: F1-F5 facts, D1-D5 Dutch, R1-R5 reasoning/math/code, T1-T5 RAPTOR tool calls (output must parse as JSON).

Thinking handling (documented choice): by default (--think off) the assistant turn is pre-filled with the empty think block
`<think>\\n\\n</think>\\n\\n`, exactly what Qwen3's chat template does for enable_thinking=false, and requests go to
/v1/completions with the ChatML built here (stop string <|im_end|>, temperature 0). With --think on the assistant turn is left
open, the model may emit <think>...</think>; the checker strips a closed think block, and FAILS tool-call prompts whose think
block is not closed before the JSON (an unclosed block means the JSON would never be parsed by an agent). With --mode chat the
server's own chat template is used (/v1/chat/completions).
Standard library only. Exit status 0 if the harness ran (pass count is reported, not enforced) unless --min-pass is given.
"""
import argparse
import http.server
import json
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

IM_S, IM_E = "<|im_start|>", "<|im_end|>"
EMPTY_THINK = "<think>\n\n</think>\n\n"
TOOLS = {"bash", "read_file", "write_file", "http_get", "python", "final"}

TOOL_SYSTEM = ("You are RAPTOR, an operations agent. You can use these tools by replying with ONE JSON object on its own line, "
               "after a short plan: {\"tool\":\"<name>\",\"args\":{...}}.\n"
               "Tools: bash(cmd), read_file(path), write_file(path, content), http_get(url), python(code), final(answer).\n"
               "Tool results arrive in <tool_response> tags. When you are done, call final with the answer. Never invent tool output.")
SCHEMA_SYSTEM = (
    "You are RAPTOR, an operations agent. You act ONLY by calling tools. Reply with exactly one JSON object "
    "{\"tool\":NAME,\"args\":{...}} and nothing else: no plan, no explanation, no markdown, no code fences.\n"
    "Tool schemas (use these exact tool names and argument names):\n"
    "- bash: {\"cmd\": string}  run a shell command\n"
    "- read_file: {\"path\": string}  read a file\n"
    "- write_file: {\"path\": string, \"content\": string}  write a file\n"
    "- http_get: {\"url\": string}  HTTP GET a URL\n"
    "- python: {\"code\": string}  run Python code\n"
    "- final: {\"answer\": string}  finish with the answer\n"
    "Examples:\n"
    "User: List the files in /srv.\n"
    "{\"tool\":\"bash\",\"args\":{\"cmd\":\"ls /srv\"}}\n"
    "User: Save the text 'ok' to /tmp/status.txt.\n"
    "{\"tool\":\"write_file\",\"args\":{\"path\":\"/tmp/status.txt\",\"content\":\"ok\"}}")
DUTCH_SYSTEM = "Je bent een behulpzame assistent. Antwoord kort en duidelijk in het Nederlands."
GENERAL_SYSTEM = "You are a helpful assistant."

# ---------------------------------------------------------------- generic helpers


def strip_think(text):
    """Returns (visible_text, think_ok). think_ok is False if a <think> is opened and never closed."""
    t = text
    if "<think>" in t and "</think>" not in t:
        return t, False
    if "</think>" in t:
        t = t.split("</think>")[-1]
    return t.strip(), True


DUTCH_STOP = {"de", "het", "een", "en", "is", "van", "in", "dat", "te", "op", "voor", "met", "niet", "zijn", "ik", "je", "jij", "u", "dit", "om", "als",
              "maar", "ze", "ook", "aan", "door", "er", "naar", "bij", "of", "kan", "kunt", "wordt", "worden", "heeft", "hebben", "gaat", "hoe", "wat", "dus",
              "dan", "jouw", "uw", "mijn", "graag", "helaas", "goede", "goedemorgen", "ben", "bent", "waar", "omdat", "lucht", "licht", "heel", "meer", "geen"}


def dutch_words(text):
    return [w for w in re.findall(r"[a-zA-Zàâäçèéêëîïôöùûüÿ']+", text.lower()) if w in DUTCH_STOP]


def garbage(text):
    """Reason string if the text looks like garbage tokens, else None."""
    if not text.strip():
        return "empty output"
    if "\ufffd" in text:
        return "replacement char"
    if re.search(r"[\u3000-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]", text):
        return "non-latin script"
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
        return "control chars"
    if "<|" in text.replace("<|im_end|>", ""):
        return "special token leaked"
    n = len(text)
    odd = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in ".,;:!?'\"()[]{}<>/\\-_+=*&%$#@^|~`€°…–—‘’“”\u00a0" or ord(c) < 128))
    if n >= 20 and odd / n > 0.10:
        return "too many odd characters"
    words = re.findall(r"\S+", text)
    if len(words) >= 12:
        grams = {}
        for i in range(len(words) - 2):
            g = tuple(words[i:i + 3])
            grams[g] = grams.get(g, 0) + 1
        if max(grams.values()) >= 4:
            return "repetition loop"
        long_words = sum(1 for w in words if len(w) > 25)
        if long_words >= 3:
            return "very long tokens"
    return None


def last_number(text):
    nums = re.findall(r"-?\d[\d.,]*", text)
    return nums[-1].rstrip(".,") if nums else None


def find_json_call(text):
    """First JSON object in text that has a 'tool' key. Returns (obj or None, error)."""
    dec = json.JSONDecoder()
    pos = 0
    first_err = "no JSON object found"
    while True:
        i = text.find("{", pos)
        if i < 0:
            return None, first_err
        try:
            obj, end = dec.raw_decode(text[i:])
            if isinstance(obj, dict) and "tool" in obj:
                return obj, None
            pos = i + 1
        except json.JSONDecodeError as e:
            first_err = f"invalid JSON: {e.msg}"
            pos = i + 1


# ---------------------------------------------------------------- checkers: (raw_output) -> (ok, reason)


def chk_contains(*alts, forbid=()):
    def f(raw):
        t, ok = strip_think(raw)
        if not ok:
            return False, "unclosed <think>"
        g = garbage(t)
        if g:
            return False, g
        for a in alts:
            if not re.search(a, t, re.I if a.islower() else 0):
                return False, f"missing /{a}/"
        for a in forbid:
            if re.search(a, t):
                return False, f"unexpected /{a}/"
        return True, "ok"
    return f


def chk_dutch(keys_all=(), keys_any=(), min_stop=2, forbid=()):
    def f(raw):
        t, ok = strip_think(raw)
        if not ok:
            return False, "unclosed <think>"
        g = garbage(t)
        if g:
            return False, g
        low = t.lower()
        for k in keys_all:
            if not re.search(k, low):
                return False, f"missing /{k}/"
        if keys_any and not any(re.search(k, low) for k in keys_any):
            return False, "missing any of " + "|".join(keys_any)
        for k in forbid:
            if re.search(k, low):
                return False, f"unexpected /{k}/"
        n = len(dutch_words(t))
        if n < min_stop:
            return False, f"not Dutch ({n} Dutch function words)"
        return True, "ok"
    return f


def chk_number(expected):
    def f(raw):
        t, ok = strip_think(raw)
        if not ok:
            return False, "unclosed <think>"
        g = garbage(t)
        if g:
            return False, g
        nums = [n.replace(",", "") for n in re.findall(r"-?\d[\d,]*\.?\d*", t)]
        nums = [n.rstrip(".") for n in nums]
        if str(expected) in nums:
            return True, "ok"
        return False, f"expected {expected}, numbers in output: {nums[:6]}"
    return f


def chk_last_number(expected):
    def f(raw):
        t, ok = strip_think(raw)
        if not ok:
            return False, "unclosed <think>"
        g = garbage(t)
        if g:
            return False, g
        n = last_number(t)
        return (n == str(expected), "ok" if n == str(expected) else f"last number {n}, expected {expected}")
    return f


def chk_yes(raw):
    t, ok = strip_think(raw)
    if not ok:
        return False, "unclosed <think>"
    g = garbage(t)
    if g:
        return False, g
    m = re.search(r"[A-Za-z]+", t)
    if m and m.group(0).lower() in ("yes", "ja"):
        return True, "ok"
    return False, "answer does not start with yes"


PRIME_TESTS = """
assert is_prime(2) and is_prime(3) and is_prime(97) and is_prime(7919)
assert not is_prime(0) and not is_prime(1) and not is_prime(91) and not is_prime(100) and not is_prime(-7)
print("PRIME_OK")
"""


def extract_code(text):
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1)
    i = text.find("def ")
    return text[i:] if i >= 0 else ""


def chk_prime(raw):
    t, ok = strip_think(raw)
    if not ok:
        return False, "unclosed <think>"
    code = extract_code(t)
    if "def is_prime" not in code:
        return False, "no def is_prime found"
    try:
        p = subprocess.run([sys.executable, "-I", "-c", code + "\n" + PRIME_TESTS], capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return False, "code timed out"
    if p.returncode == 0 and "PRIME_OK" in p.stdout:
        return True, "ok (asserts passed)"
    err = (p.stderr.strip().splitlines() or ["failed"])[-1]
    return False, "code failed: " + err[:100]


def tool_checker(accept):
    """accept(obj) -> bool for the parsed call. Also requires: think closed, valid JSON, known tool, dict args."""
    def f(raw):
        if "<think>" in raw and "</think>" not in raw:
            return False, "unclosed <think> before JSON"
        t = raw.split("</think>")[-1] if "</think>" in raw else raw
        obj, err = find_json_call(t)
        if obj is None:
            return False, err
        if obj.get("tool") not in TOOLS:
            return False, f"unknown tool {obj.get('tool')!r}"
        if not isinstance(obj.get("args"), dict):
            return False, "args is not an object"
        try:
            if accept(obj):
                return True, "ok"
        except Exception as e:  # noqa: BLE001
            return False, f"checker error {e}"
        return False, "wrong tool/args: " + json.dumps(obj)[:100]
    return f


def _s(o, k):
    v = o["args"].get(k, "")
    return v if isinstance(v, str) else json.dumps(v)


def acc_df(o):
    return o["tool"] == "bash" and re.search(r"\bdf\b", _s(o, "cmd")) and "/var" in _s(o, "cmd")


def acc_read(o):
    return (o["tool"] == "read_file" and _s(o, "path").strip() == "/etc/hostname") or (o["tool"] == "bash" and re.search(r"\bcat\b", _s(o, "cmd")) and "/etc/hostname" in _s(o, "cmd"))


def acc_curl(o):
    u = "127.0.0.1:8080/health"
    return (o["tool"] == "bash" and "curl" in _s(o, "cmd") and u in _s(o, "cmd")) or (o["tool"] == "http_get" and u in _s(o, "url"))


def acc_py(o):
    c = _s(o, "code") if o["tool"] == "python" else (_s(o, "cmd") if o["tool"] == "bash" and "python" in _s(o, "cmd") else "")
    return bool(c) and "12345" in c and "6789" in c


def acc_write(o):
    if o["tool"] == "write_file":
        return _s(o, "path") == "/tmp/greeting.txt" and "hello" in _s(o, "content").lower()
    return o["tool"] == "bash" and "/tmp/greeting.txt" in _s(o, "cmd") and "hello" in _s(o, "cmd").lower()


# id, group, system, user, checker
PROMPTS = [
    ("F1", "fact", GENERAL_SYSTEM, "What is the capital of France? Answer in one sentence.", chk_contains(r"\bParis\b")),
    ("F2", "fact", GENERAL_SYSTEM, "What is the capital of the Netherlands? Answer in one sentence.", chk_contains(r"\bAmsterdam\b")),
    ("F3", "fact", GENERAL_SYSTEM, "What is the chemical symbol of gold?", chk_contains(r"\bAu\b")),
    ("F4", "fact", GENERAL_SYSTEM, "Who wrote the play Hamlet?", chk_contains(r"Shakespeare")),
    ("F5", "fact", GENERAL_SYSTEM, "How many days does a leap year have? Answer with the number.", chk_number(366)),
    ("D1", "dutch", DUTCH_SYSTEM, "Vertaal naar het Nederlands: 'Good morning, how are you today?'",
     chk_dutch(keys_any=[r"goedemorgen", r"goede morgen"], min_stop=1)),
    ("D2", "dutch", DUTCH_SYSTEM, "Leg in twee zinnen uit waarom de lucht overdag blauw is.", chk_dutch(keys_all=[r"blauw", r"licht|zon"], min_stop=5)),
    ("D3", "dutch", DUTCH_SYSTEM, "Noem de vier jaargetijden in het Nederlands.",
     chk_dutch(keys_all=[r"lente|voorjaar", r"zomer", r"herfst|najaar", r"winter"], min_stop=0)),
    ("D4", "dutch", DUTCH_SYSTEM, "Schrijf een korte, beleefde zin waarmee ik een afspraak afzeg.",
     chk_dutch(keys_any=[r"afspraak"], keys_all=[r"afzeg|annuler|kan.{0,20}niet|helaas|verzet|verplaats"], min_stop=3)),
    ("D5", "dutch", DUTCH_SYSTEM, "Wat is het verschil tussen een hond en een kat? Antwoord kort in het Nederlands.",
     chk_dutch(keys_all=[r"\bhond", r"\bkat"], min_stop=5)),
    ("R1", "reason", GENERAL_SYSTEM, "What is 17*23? Answer with just the number.", chk_number(391)),
    ("R2", "reason", GENERAL_SYSTEM, "Write a Python function is_prime(n) that returns True if n is prime and False otherwise. Reply with only the code in a ```python block.", chk_prime),
    ("R3", "reason", GENERAL_SYSTEM, "A train travels at 60 km per hour for 2.5 hours. How many kilometres does it travel? Answer with a number.", chk_number(150)),
    ("R4", "reason", GENERAL_SYSTEM, "Tom has 5 apples, gives 2 to Anna and then buys 4 more. How many apples does Tom have now? Answer with a number.", chk_last_number(7)),
    ("R5", "reason", GENERAL_SYSTEM, "All bloops are razzies and all razzies are lazzies. Are all bloops lazzies? Answer yes or no, then explain in one sentence.", chk_yes),
    ("T1", "tool", TOOL_SYSTEM, "Show how much disk space is free on /var.", tool_checker(acc_df)),
    ("T2", "tool", TOOL_SYSTEM, "Read the file /etc/hostname.", tool_checker(acc_read)),
    ("T3", "tool", TOOL_SYSTEM, "Check the HTTP status code of http://127.0.0.1:8080/health.", tool_checker(acc_curl)),
    ("T4", "tool", TOOL_SYSTEM, "Use python to compute 12345 * 6789.", tool_checker(acc_py)),
    ("T5", "tool", TOOL_SYSTEM, "Write the text 'hello' to the file /tmp/greeting.txt.", tool_checker(acc_write)),
]


def build_prompt(system, user, think_off):
    return f"{IM_S}system\n{system}{IM_E}\n{IM_S}user\n{user}{IM_E}\n{IM_S}assistant\n" + (EMPTY_THINK if think_off else "")


# ---------------------------------------------------------------- client


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def run_one(args, pid, system, user):
    if getattr(args, "system_schema", False) and system == TOOL_SYSTEM:
        system = SCHEMA_SYSTEM
    t0 = time.time()
    if args.mode == "chat":
        body = {"model": args.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": args.max_tokens, "temperature": 0, "stop": [IM_E]}
        data = post(args.url.rstrip("/") + "/v1/chat/completions", body, args.timeout)
        text = data["choices"][0]["message"]["content"]
    else:
        body = {"model": args.model, "prompt": build_prompt(system, user, args.think == "off"), "max_tokens": args.max_tokens,
                "temperature": 0, "stop": [IM_E]}
        data = post(args.url.rstrip("/") + "/v1/completions", body, args.timeout)
        ch = data["choices"][0]
        text = ch.get("text") if ch.get("text") is not None else ch.get("message", {}).get("content", "")
    wall = time.time() - t0
    usage = data.get("usage", {})
    tim = data.get("timings", {})
    ntok = usage.get("completion_tokens") or tim.get("completion_tokens") or 0
    tps = tim.get("tokens_per_second")
    if not tps:
        tps = ntok / wall if wall > 0 and ntok else 0.0
    return text, ntok, float(tps), wall


def evaluate(args):
    rows = []
    only = set(args.only.split(",")) if args.only else None
    for pid, group, system, user, checker in PROMPTS:
        if only and pid not in only:
            continue
        try:
            text, ntok, tps, wall = run_one(args, pid, system, user)
            ok, why = checker(text)
            err = None
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            text, ntok, tps, wall, ok, why, err = "", 0, 0.0, 0.0, False, f"request failed: {e}", str(e)
        rows.append({"id": pid, "group": group, "pass": bool(ok), "reason": why, "completion_tokens": ntok, "tok_s": round(tps, 2),
                     "wall_s": round(wall, 2), "output": text, "error": err})
    return rows


def print_report(rows, out=sys.stdout):
    out.write(f"{'id':3} {'res':4} {'tok':>4} {'tok/s':>7}  {'why':28} output (first 120 chars)\n")
    for r in rows:
        o = r["output"][:120].replace("\n", "\\n")
        out.write(f"{r['id']:3} {'PASS' if r['pass'] else 'FAIL':4} {r['completion_tokens']:4d} {r['tok_s']:7.1f}  {r['reason'][:28]:28} {o}\n")
    npass = sum(r["pass"] for r in rows)
    meas = [r["tok_s"] for r in rows if r["tok_s"] > 0]
    mean = sum(meas) / len(meas) if meas else 0.0
    groups = {}
    for r in rows:
        g = groups.setdefault(r["group"], [0, 0])
        g[0] += r["pass"]
        g[1] += 1
    out.write("by group: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in groups.items()) + "\n")
    out.write(f"SUMMARY: {npass}/{len(rows)} passed, mean {mean:.1f} tok/s\n")
    return npass, mean


# ---------------------------------------------------------------- self test with a canned fake server

GOOD = {
    "F1": "The capital of France is Paris.", "F2": "The capital of the Netherlands is Amsterdam.", "F3": "The chemical symbol of gold is Au.",
    "F4": "Hamlet was written by William Shakespeare.", "F5": "A leap year has 366 days.",
    "D1": "Goedemorgen, hoe gaat het vandaag met je?", "D2": "De lucht is overdag blauw omdat het zonlicht door de atmosfeer wordt verstrooid. Blauw licht verstrooit het sterkst, en daarom zien we een blauwe lucht.",
    "D3": "De vier jaargetijden zijn lente, zomer, herfst en winter.", "D4": "Helaas kan ik niet naar onze afspraak komen, kunnen we een nieuwe datum plannen?",
    "D5": "Een hond is meestal trouw en actief, en een kat is zelfstandiger. De hond wil graag wandelen, de kat is vaak liever binnen.",
    "R1": "391", "R2": "```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True\n```",
    "R3": "The train travels 150 km.", "R4": "5 - 2 = 3, then 3 + 4 = 7. Tom has 7 apples.", "R5": "Yes. Every bloop is a razzie, and every razzie is a lazzie.",
    "T1": "I will check the disk usage.\n{\"tool\":\"bash\",\"args\":{\"cmd\":\"df -h /var\"}}",
    "T2": "<think>\nread the file\n</think>\n\n{\"tool\":\"read_file\",\"args\":{\"path\":\"/etc/hostname\"}}",
    "T3": "Plan: curl the endpoint.\n{\"tool\":\"bash\",\"args\":{\"cmd\":\"curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health\"}}",
    "T4": "{\"tool\":\"python\",\"args\":{\"code\":\"print(12345 * 6789)\"}}",
    "T5": "{\"tool\":\"write_file\",\"args\":{\"path\":\"/tmp/greeting.txt\",\"content\":\"hello\"}}",
}
BAD = {
    "F1": "The capital of France is Lyon.", "F2": "Den Haag", "F3": "Ag", "F4": "Christopher Marlowe", "F5": "365",
    "D1": "Good morning, how are you today?", "D2": "\ufffd\ufffd \u5929\u7a7a Blauw ### zzzz", "D3": "Spring, summer, autumn and winter.", "D4": "I cannot come.", "D5": "dog cat dog cat dog cat",
    "R1": "392", "R2": "```python\ndef is_prime(n):\n    return n % 2 == 1\n```", "R3": "100", "R4": "Tom has 3 apples.", "R5": "No, they are not.",
    "T1": "{\"tool\":\"bash\",\"args\":{\"cmd\":\"ls /tmp\"}}", "T2": "{\"tool\":\"read_file\",\"args\":{\"path\":\"/etc/passwd\"}}",
    "T3": "<think>\nI should curl it\n{\"tool\":\"bash\",\"args\":{\"cmd\":\"curl http://127.0.0.1:8080/health\"}}",
    "T4": "{\"tool\":\"python\",\"args\":{\"code\":\"print(1+1\"}}", "T5": "I would write hello to the file.",
}


SEEN_PROMPTS = []


def make_fake_server(answers):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            prompt = body.get("prompt") or json.dumps(body.get("messages"))
            SEEN_PROMPTS.append(prompt)
            out = ""
            for pid, _g, _s_, user, _c in PROMPTS:
                if json.dumps(user)[1:-1] in prompt or user in prompt:
                    out = answers[pid]
            resp = {"choices": [{"index": 0, "text": out, "message": {"role": "assistant", "content": out}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 30, "completion_tokens": max(1, len(out) // 4), "total_tokens": 40},
                    "timings": {"tokens_per_second": 25.0, "completion_tokens": max(1, len(out) // 4)}}
            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def self_test():
    ns = argparse.Namespace(system_schema=False, model="x", max_tokens=50, think="off", only=None, mode="completions", timeout=30, url="")
    fails = 0
    for name, answers, want in (("good", GOOD, True), ("bad", BAD, False)):
        srv = make_fake_server(answers)
        ns.url = f"http://127.0.0.1:{srv.server_address[1]}"
        rows = evaluate(ns)
        srv.shutdown()
        print(f"--- fake server: {name} answers ---")
        npass, mean = print_report(rows)
        for r in rows:
            if r["pass"] != want:
                print(f"SELF-TEST FAIL: {r['id']} expected {'pass' if want else 'fail'} ({r['reason']})")
                fails += 1
        if abs(mean - 25.0) > 0.01:
            print("SELF-TEST FAIL: tok/s parsing")
            fails += 1
    # --system-schema: tool prompts must carry SCHEMA_SYSTEM (and only they), graders unchanged
    srv = make_fake_server(GOOD)
    ns.url = f"http://127.0.0.1:{srv.server_address[1]}"
    ns.system_schema = True
    del SEEN_PROMPTS[:]
    rows = evaluate(ns)
    srv.shutdown()
    ns.system_schema = False
    nsch = sum(1 for p in SEEN_PROMPTS if "Tool schemas" in p)
    if nsch != 5 or not all(r["pass"] for r in rows):
        print(f"SELF-TEST FAIL: --system-schema (schema prompts={nsch}, pass={sum(r['pass'] for r in rows)})")
        fails += 1
    # prompt building and checker edge cases
    assert build_prompt("S", "U", True).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert build_prompt("S", "U", False).endswith("<|im_start|>assistant\n")
    assert find_json_call('plan {"a":1} then {"tool":"bash","args":{}}')[0]["tool"] == "bash"
    assert tool_checker(acc_df)("{'tool': 'bash'}")[0] is False
    print("SELF-TEST", "FAILED" if fails else "OK", f"({fails} problems)")
    return 1 if fails else 0


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="/content/hf17")
    ap.add_argument("--mode", choices=["completions", "chat"], default="completions")
    ap.add_argument("--think", choices=["off", "on"], default="off")
    ap.add_argument("--max-tokens", type=int, default=250)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--only", default="")
    ap.add_argument("--report", default="")
    ap.add_argument("--min-pass", type=int, default=-1, help="exit 1 if fewer prompts pass")
    ap.add_argument("--system-schema", action="store_true", help="for the T1-T5 tool prompts replace the system message with a strict tool-schema prompt (SCHEMA_SYSTEM)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--regrade", default="", help="re-run the checkers on the outputs stored in an earlier --report JSON (no server needed)")
    args = ap.parse_args()
    if args.regrade:
        old = json.load(open(args.regrade, encoding="utf-8"))
        by_id = {p[0]: p for p in PROMPTS}
        rows = []
        for r in old["results"]:
            ok, why = by_id[r["id"]][4](r["output"])
            rows.append(dict(r, **{"pass": bool(ok), "reason": why}))
        print_report(rows)
        return
    if args.self_test:
        sys.exit(self_test())
    rows = evaluate(args)
    npass, mean = print_report(rows)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"url": args.url, "model": args.model, "mode": args.mode, "think": args.think, "system_schema": args.system_schema, "max_tokens": args.max_tokens,
                       "passed": npass, "total": len(rows), "mean_tok_s": round(mean, 2), "results": rows}, f, indent=2, ensure_ascii=False)
    sys.exit(1 if args.min_pass >= 0 and npass < args.min_pass else 0)


if __name__ == "__main__":
    main()
