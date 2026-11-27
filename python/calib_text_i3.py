#!/usr/bin/env python3
"""Calibration / held-out token sequences for the INT3-attention Hessians.

Never use evaluation prompts as calibration text. Put your own text into <src>/nl, <src>/en and <src>/code (the text used for the measured
model file is not published):
  nl    Dutch and mixed-language documents (*.md)
  en    English documents (*.md, *.txt)
  code  source files (any extension)
  json  synthetic JSON records, Qwen-style tool-call chats (<tool_call>/<tool_response>), short Dutch/English chat turns
Writes <out>/cal_XX.ids (2048 tokens each, comma separated, 8 per category) and <out>/held_<cat>.ids (1024 tokens each).
Fixed seed, deterministic: windows of a pool are shuffled, the LAST window of every pool is the held-out one.
usage: python calib_text_i3.py --tok <model dir>/tokenizer.json --out <out dir> [--src <source dir>]"""
import argparse, glob, json, os, random, re
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--tok', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--src', default='calib_src'); ap.add_argument('--seqs', type=int, default=8)
ap.add_argument('--seqlen', type=int, default=2048); ap.add_argument('--heldlen', type=int, default=1024)
a = ap.parse_args()
tok = Tokenizer.from_file(a.tok)
os.makedirs(a.out, exist_ok=True)
rnd = random.Random(20241124)

def rd(p, lim=None):
    try:
        t = open(p, encoding='utf-8', errors='replace').read()
    except OSError:
        return ''
    return t[:lim] if lim else t

def files(pats):
    out = []
    for p in pats: out += sorted(glob.glob(p, recursive=True))
    return out

# ---------------- pools ----------------
nl = [rd(p) for p in files([a.src + '/nl/**/*.md'])]
en = [rd(p) for p in files([a.src + '/en/**/*.md', a.src + '/en/**/*.txt'])]
code_files = [p for p in files([a.src + '/code/**/*']) if os.path.isfile(p)]
code = [rd(p, 40000) for p in code_files]

def tools_pool():
    return [
        ('get_weather', 'Get current weather for a city', {'city': 'string', 'unit': 'string'}, lambda: {'city': rnd.choice(['Amsterdam', 'Utrecht', 'Berlin', 'Lyon', 'Gent', 'Porto']), 'unit': rnd.choice(['celsius', 'fahrenheit'])}),
        ('search_web', 'Search the web', {'query': 'string', 'limit': 'integer'}, lambda: {'query': rnd.choice(['beste fietsroute Zeeland', 'nginx rate limit config', 'postgres vacuum full vs analyze', 'train strike Belgium', 'weer morgen Rotterdam']), 'limit': rnd.choice([3, 5, 10])}),
        ('read_file', 'Read a text file', {'path': 'string'}, lambda: {'path': rnd.choice(['/etc/hosts', '/var/log/syslog', 'src/main.rs', 'config/settings.yaml', 'README.md'])}),
        ('run_shell', 'Run a shell command', {'cmd': 'string'}, lambda: {'cmd': rnd.choice(['uptime', 'df -h', 'ls -la /var/www', 'systemctl status nginx', 'git log --oneline -5'])}),
        ('send_email', 'Send an email', {'to': 'string', 'subject': 'string', 'body': 'string'}, lambda: {'to': rnd.choice(['jan@example.org', 'info@bedrijf.nl', 'support@example.com']), 'subject': rnd.choice(['Offerte', 'Afspraak donderdag', 'Invoice 2043', 'Status update']), 'body': rnd.choice(['Hierbij de offerte.', 'Kunnen we donderdag om 14:00 afspreken?', 'Please find the invoice attached.', 'Alles loopt volgens planning.'])}),
        ('create_event', 'Create a calendar event', {'title': 'string', 'start': 'string', 'minutes': 'integer'}, lambda: {'title': rnd.choice(['Stand-up', 'Klantgesprek', 'Review', 'Lunch']), 'start': '2026-%02d-%02dT%02d:00:00' % (rnd.randint(1, 12), rnd.randint(1, 28), rnd.randint(8, 17)), 'minutes': rnd.choice([15, 30, 45, 60])}),
        ('query_db', 'Run a read-only SQL query', {'sql': 'string'}, lambda: {'sql': rnd.choice(['SELECT count(*) FROM orders WHERE status = \'open\'', 'SELECT id, email FROM users ORDER BY created_at DESC LIMIT 5', 'SELECT sum(total) FROM invoices WHERE paid = false'])}),
        ('calculator', 'Evaluate an arithmetic expression', {'expression': 'string'}, lambda: {'expression': rnd.choice(['(1249 * 0.21) + 1249', '37 * 41 - 19', '2 ** 16 / 8', '1500 / 12'])}),
    ]
TOOLS = tools_pool()
PY2J = {'string': 'string', 'integer': 'integer'}
SYS = 'You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n%s\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
USERS = ['Wat is het weer in {city}?', 'Zoek even op: {query}', 'Lees {path} en vat het samen.', 'Draai {cmd} en laat me de uitvoer zien.', 'Stuur een mailtje naar {to} met onderwerp "{subject}".', 'Plan "{title}" in op {start}.', 'Hoeveel openstaande orders hebben we? ({sql})', 'Reken {expression} voor me uit.',
         'What is the weather in {city}?', 'Please search for: {query}', 'Open {path} and tell me what it does.', 'Run `{cmd}` and report back.', 'Mail {to} about "{subject}".', 'Book "{title}" at {start}.', 'Run this query: {sql}', 'Compute {expression}.']
FINAL = ['Dit is het resultaat: ', 'Klaar. De uitkomst is hierboven te zien; ', 'Gedaan! ', 'Here is what I found: ', 'Done. ', 'Het is gelukt, ']

def tool_chat():
    k = rnd.randint(2, 4)
    tl = rnd.sample(TOOLS, k)
    defs = '\n'.join(json.dumps({'type': 'function', 'function': {'name': n, 'description': d, 'parameters': {'type': 'object', 'properties': {kk: {'type': PY2J[v]} for kk, v in p.items()}, 'required': list(p)}}}) for n, d, p, _ in tl)
    out = '<|im_start|>system\n' + SYS % defs + '<|im_end|>\n'
    for _ in range(rnd.randint(1, 3)):
        n, d, p, gen = rnd.choice(tl)
        args = gen()
        out += '<|im_start|>user\n' + rnd.choice(USERS).format(**{**{kk: '' for kk in ['city', 'query', 'path', 'cmd', 'to', 'subject', 'title', 'start', 'sql', 'expression']}, **args}) + '<|im_end|>\n'
        out += '<|im_start|>assistant\n<tool_call>\n' + json.dumps({'name': n, 'arguments': args}) + '\n</tool_call><|im_end|>\n'
        res = {'ok': True, 'result': rnd.choice([rnd.randint(1, 999), 'open: %d' % rnd.randint(0, 80), {'temp': rnd.randint(-3, 31), 'sky': rnd.choice(['bewolkt', 'zonnig', 'regen', 'clear'])}, [rnd.randint(1, 50) for _ in range(rnd.randint(2, 6))], 'exit 0'])}
        out += '<|im_start|>user\n<tool_response>\n' + json.dumps(res) + '\n</tool_response><|im_end|>\n'
        out += '<|im_start|>assistant\n' + rnd.choice(FINAL) + json.dumps(res['result'], ensure_ascii=False) + '<|im_end|>\n'
    return out

CHAT_NL = [('Hoi!', 'Hoi! Leuk dat je er bent. Waar kan ik je mee helpen?'), ('Goedemorgen, kun je me iets uitleggen over DNS?', 'Natuurlijk. DNS vertaalt namen zoals voorbeeld.nl naar IP-adressen, zodat je browser weet met welke server hij moet praten.'),
           ('Kun je een korte samenvatting geven van wat een API is?', 'Een API is een afgesproken manier waarop programma\'s met elkaar praten: jij stuurt een verzoek in een vast formaat en krijgt een antwoord terug.'), ('Bedankt voor je hulp!', 'Graag gedaan! Laat het maar weten als je nog iets nodig hebt.'),
           ('Schrijf een zin over de zee.', 'De golven rolden rustig het strand op, terwijl de meeuwen hoog boven het water hun rondjes draaiden.'), ('Hello, who are you?', 'Hello! I am an AI assistant. I can answer questions, help with writing and code, and call tools when they are available.'),
           ('Explain what a mutex is.', 'A mutex is a lock that lets only one thread at a time enter a critical section, so shared data is never modified by two threads at once.'), ('Hoe laat is het ongeveer in Tokio als het hier 12:00 is?', 'Tokio loopt ongeveer acht uur voor op Nederland in de zomer en zeven uur in de winter, dus dan is het daar ongeveer 20:00 of 19:00.')]
def chat_nl():
    out = ''
    for _ in range(rnd.randint(2, 4)):
        u, r = rnd.choice(CHAT_NL)
        out += '<|im_start|>user\n' + u + '<|im_end|>\n<|im_start|>assistant\n' + r + '<|im_end|>\n'
    return out

def json_blob():
    kind = rnd.randint(0, 2)
    if kind == 0:
        recs = [{'id': rnd.randint(1000, 99999), 'naam': rnd.choice(['Jansen BV', 'De Vries Bouw', 'Smit Logistiek', 'Acme Ltd', 'Bakker & Zn']), 'plaats': rnd.choice(['Utrecht', 'Breda', 'Zwolle', 'Leiden', 'Delft']), 'omzet': round(rnd.uniform(1e3, 9e5), 2), 'actief': rnd.random() > .3, 'tags': rnd.sample(['mkb', 'retail', 'bouw', 'it', 'zorg', 'export'], 2)} for _ in range(rnd.randint(6, 14))]
        return json.dumps({'klanten': recs}, indent=rnd.choice([None, 1, 2]), ensure_ascii=False)
    if kind == 1:
        return '\n'.join(json.dumps({'ts': '2026-%02d-%02dT%02d:%02d:%02dZ' % (rnd.randint(1, 12), rnd.randint(1, 28), rnd.randint(0, 23), rnd.randint(0, 59), rnd.randint(0, 59)), 'level': rnd.choice(['info', 'warn', 'error']), 'svc': rnd.choice(['api', 'worker', 'mailer']), 'msg': rnd.choice(['request handled', 'retrying upstream', 'queue length high', 'verbinding verbroken', 'cache miss']), 'ms': rnd.randint(1, 900)}) for _ in range(rnd.randint(12, 24)))
    return json.dumps({'openapi': '3.0.1', 'paths': {'/v1/' + rnd.choice(['orders', 'users', 'items']) + '/{id}': {'get': {'parameters': [{'name': 'id', 'in': 'path', 'required': True, 'schema': {'type': 'integer'}}], 'responses': {'200': {'description': 'OK'}, '404': {'description': 'Not found'}}}}}}, indent=2)

def json_pool(n_chars):
    parts = []; n = 0
    while n < n_chars:
        p = rnd.choice([tool_chat, tool_chat, chat_nl, json_blob])()
        parts.append(p); n += len(p)
    return parts

# ---------------- windows ----------------
def windows(texts, L, H):
    ids = []
    for t in texts:
        if len(t) < 200: continue
        ids += tok.encode(t).ids
    need = a.seqs * L + H
    wins = [ids[i:i + L] for i in range(0, len(ids) - L + 1, L)]
    return ids, wins

def build(name, texts):
    L = a.seqlen
    if name == 'json':
        # synthetic: every sequence is a fresh draw, held-out is a separate draw (same seed stream, later)
        seqs = []
        for _ in range(a.seqs + 1):
            ids = []
            while len(ids) < L:
                ids += tok.encode(rnd.choice(json_pool(1500))).ids
            seqs.append(ids[:L])
        return seqs[:a.seqs], seqs[a.seqs][:a.heldlen]
    ids, wins = windows(texts, L, a.heldlen)
    print(f'{name}: pool {len(ids)} tokens, {len(wins)} windows')
    if len(wins) < a.seqs + 1:
        print(f'  WARNING pool of {name} too small for {a.seqs + 1} distinct windows; windows reused')
        while len(wins) < a.seqs + 1: wins.append(wins[len(wins) % max(1, len(wins))])
    held = wins[-1][:a.heldlen]
    rest = wins[:-1]; rnd.shuffle(rest)
    return rest[:a.seqs], held

n = 0
for name, texts in [('nl', nl), ('en', en), ('code', code), ('json', None)]:
    seqs, held = build(name, texts)
    for s in seqs:
        open(f'{a.out}/cal_{n:02d}_{name}.ids', 'w').write(','.join(map(str, s))); n += 1
    open(f'{a.out}/held_{name}.ids', 'w').write(','.join(map(str, held)))
print('sequences', n, 'x', a.seqlen, '=', n * a.seqlen, 'tokens; held-out 4 x', a.heldlen)
