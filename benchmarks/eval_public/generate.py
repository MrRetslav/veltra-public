#!/usr/bin/env python3
"""Deterministic generator of the 160-task public eval set (tasks.jsonl): 40 math / logic, 40 Python with unit tests, 40 strict JSON / tool calls,
40 long-context extraction (about 1500-3000 tokens). All text is generated or written here; no third-party benchmark items. Fixed seed.

  python3 benchmarks/eval_public/generate.py          # rewrites tasks.jsonl next to this file
"""
import json
import os
import random

from code_tasks import CODE_TASKS

SEED = 20261127
HERE = os.path.dirname(os.path.abspath(__file__))
PEOPLE = ["Noor", "Sem", "Yara", "Daan", "Mila", "Levi", "Sofie", "Bram", "Zoë", "Finn", "Lotte", "Thijs", "Evi", "Mats", "Roos", "Jules"]


# ---------------------------------------------------------------- math and multi-step logic (exact integer answers)
def m_shop(r):
    a, b = r.randint(3, 12), r.randint(4, 15)
    pa, pb = r.randint(2, 9), r.randint(3, 14)
    disc = r.choice([10, 20, 25, 50])
    total_a = a * pa
    while (total_a * disc) % 100:
        total_a, a = total_a + pa, a + 1
    ans = total_a - total_a * disc // 100 + b * pb
    return ("A shop sells notebooks for %d euros each and pens for %d euros each. %s buys %d notebooks and %d pens. Only the notebooks get a %d%% "
            "discount. How many euros does %s pay in total?" % (pa, pb, r.choice(PEOPLE), a, b, disc, "this customer")), ans


def m_boxes(r):
    per, boxes, extra, bag = r.randint(12, 36), r.randint(7, 25), r.randint(1, 11), r.randint(5, 9)
    total = per * boxes + extra
    return ("A warehouse has %d full boxes with %d bolts each, plus %d loose bolts. All bolts are repacked into bags of %d. How many bolts are left "
            "over after filling as many bags as possible?" % (boxes, per, extra, bag)), total % bag


def m_rate(r):
    m1, t1, parts = r.randint(3, 8), r.randint(2, 6), r.randint(4, 9)
    per = parts * 10
    m2, t2 = r.randint(5, 14), r.randint(3, 9)
    ans = per * m2 * t2
    return ("%d identical machines make %d parts in %d hours. How many parts do %d of these machines make in %d hours?" % (m1, per * m1 * t1, t1, m2, t2)), ans


def m_time(r):
    h, m = r.randint(6, 11), r.choice([5, 10, 25, 40, 50])
    legs = [r.randint(20, 95) for _ in range(3)]
    waits = [r.randint(6, 18) for _ in range(2)]
    total = h * 60 + m + sum(legs) + sum(waits)
    return ("A trip starts at %02d:%02d. It has three legs of %d, %d and %d minutes, with waits of %d minutes after the first leg and %d minutes "
            "after the second. At what time does it end? Answer as the number of minutes after midnight." % (h, m, *legs, *waits)), total


def m_ages(r):
    child = r.randint(4, 15)
    diff = r.randint(22, 34)
    years = r.randint(3, 12)
    ans = (child + years) + (child + diff + years)
    return ("A parent is %d years older than their child. The child is %d years old now. What will the sum of their two ages be in %d years?"
            % (diff, child, years)), ans


def m_perimeter(r):
    w = r.randint(6, 30)
    k = r.randint(2, 4)
    gate = r.randint(1, 4)
    price = r.randint(7, 19)
    per = 2 * (w + k * w) - gate
    return ("A rectangular field is %d times as long as it is wide. Its width is %d metres. A fence goes all around it except for one %d metre gate. "
            "Fencing costs %d euros per metre. What does the fence cost in euros?" % (k, w, gate, price)), per * price


def m_stock(r):
    start = r.randint(300, 900)
    ops, v, text = [], start, []
    for _ in range(5):
        kind = r.choice(["sell", "receive", "half"])
        if kind == "half" and v % 2 == 0:
            v //= 2
            text.append("then half of the stock is moved to another site")
        elif kind == "sell":
            x = r.randint(20, max(21, v // 3))
            v -= x
            text.append("then %d units are sold" % x)
        else:
            x = r.randint(30, 250)
            v += x
            text.append("then %d units arrive" % x)
    return ("A stock starts at %d units, %s. How many units are in stock at the end?" % (start, ", ".join(text))), v


def m_digits(r):
    n = r.randint(120, 480)
    d = r.choice([3, 4, 6, 7, 9])
    k = r.choice([5, 8, 11])
    ans = sum(1 for i in range(1, n + 1) if i % d == 0 and i % k != 0)
    return ("How many whole numbers from 1 to %d are divisible by %d but not by %d?" % (n, d, k)), ans


def m_handshake(r):
    n = r.randint(9, 24)
    pairs = r.randint(2, 4)
    return ("%d people meet. Everyone shakes hands exactly once with everyone else, except that %d specific pairs do not shake hands. How many "
            "handshakes take place?" % (n, pairs)), n * (n - 1) // 2 - pairs


def m_order(r):
    names = r.sample(PEOPLE, 5)
    scores = r.sample(range(40, 99), 5)
    gaps = scores[:]
    lines = ["%s scored %d points." % (names[0], scores[0])]
    for i in range(1, 5):
        d = scores[i] - scores[i - 1]
        lines.append("%s scored %d points %s than %s." % (names[i], abs(d), "more" if d > 0 else "fewer", names[i - 1]))
    rank = r.randint(2, 4)
    ans = sorted(gaps, reverse=True)[rank - 1]
    return (" ".join(lines) + " What is the score of the person in place %d when ranked from highest to lowest?" % rank), ans


MATH = [m_shop, m_boxes, m_rate, m_time, m_ages, m_perimeter, m_stock, m_digits, m_handshake, m_order]


def math_tasks(r):
    out = []
    for i in range(40):
        q, ans = MATH[i % len(MATH)](r)
        out.append({"id": "math%02d" % (i + 1), "cat": "math", "max_tokens": 400,
                    "messages": [{"role": "user", "content": q + "\nThink step by step briefly, then end with a line of the form 'Answer: <number>'."}],
                    "check": {"type": "number", "value": ans}})
    return out


def code_tasks():
    assert len(CODE_TASKS) == 40
    return [{"id": "code%02d" % (i + 1), "cat": "code", "max_tokens": 450,
             "messages": [{"role": "user", "content": "Write a Python function with the signature `%s`. %s\nReply with only the code in one "
                                                      "```python block. Do not include tests or example calls." % (sig, spec)}],
             "check": {"type": "python", "tests": tests}} for i, (sig, spec, tests) in enumerate(CODE_TASKS)]


# ---------------------------------------------------------------- strict JSON extraction and tool calls
CITIES = ["Utrecht", "Zwolle", "Delft", "Leiden", "Breda", "Arnhem", "Assen", "Venlo", "Hoorn", "Emmen"]
ITEMS = ["hex bolts", "cable ties", "wall plugs", "pipe clamps", "door hinges", "led panels", "drain grates", "roof anchors"]
PRIOS = ["low", "normal", "high", "urgent"]


def json_extract(r, i):
    kind = i % 4
    name, other = r.sample(PEOPLE, 2)
    city, city2 = r.sample(CITIES, 2)
    item, item2 = r.sample(ITEMS, 2)
    qty, old = r.sample(range(3, 180), 2)
    day, day2 = r.sample(range(2, 28), 2)
    month, mnum = r.choice([("January", 1), ("March", 3), ("April", 4), ("June", 6), ("September", 9), ("November", 11)])
    if kind == 0:
        text = ("Hi, this is %s from %s. We first thought about %d %s, but please make that %d. Deliver on %d %s 2027 to our %s depot, not to %s. "
                "Thanks!" % (name, city, old, item, qty, day, month, city2, city))
        schema = '{"customer": string, "item": string, "quantity": integer, "delivery_date": "YYYY-MM-DD", "delivery_city": string}'
        want = {"customer": name, "item": item, "quantity": qty, "delivery_date": "2027-%02d-%02d" % (mnum, day), "delivery_city": city2}
    elif kind == 1:
        prio = r.choice(PRIOS)
        room = "%s-%d" % (r.choice("ABCD"), r.randint(100, 420))
        text = ("Ticket from %s: the heating in room %s has been off since this morning. %s had the same problem in room E-%d last week, but that one "
                "is fixed. Please mark this as %s priority. Call back on extension %d." % (name, room, other, r.randint(100, 420), prio, 2000 + qty))
        schema = '{"reporter": string, "room": string, "priority": "low"|"normal"|"high"|"urgent", "extension": integer}'
        want = {"reporter": name, "room": room, "priority": prio, "extension": 2000 + qty}
    elif kind == 2:
        price = r.randint(3, 40)
        text = ("Invoice note: %d x %s at %d euros each, and %d x %s at %d euros each. No VAT on this note." % (qty, item, price, old, item2, price + 2))
        schema = '{"lines": [{"item": string, "quantity": integer, "unit_price": integer}], "total": integer}'
        want = {"lines": [{"item": item, "quantity": qty, "unit_price": price}, {"item": item2, "quantity": old, "unit_price": price + 2}],
                "total": qty * price + old * (price + 2)}
    else:
        att = r.sample(PEOPLE, 3)
        mins = r.choice([25, 30, 45, 50, 90])
        hh = r.randint(8, 17)
        text = ("Can we move the review? %s, %s and %s should join; %s cannot make it. New slot: %d %s 2027 at %02d:30, %d minutes."
                % (att[0], att[1], att[2], other if other not in att else "the intern", day2, month, hh, mins))
        schema = '{"attendees": [string], "date": "YYYY-MM-DD", "start": "HH:MM", "duration_minutes": integer}'
        want = {"attendees": att, "date": "2027-%02d-%02d" % (mnum, day2), "start": "%02d:30" % hh, "duration_minutes": mins}
    content = ("Extract the information from the message below. Reply with exactly one JSON object with this schema and nothing else:\n%s\n\n"
               "Message:\n%s" % (schema, text))
    return content, want


TOOLS_DOC = ("You can call exactly one of these tools:\n- create_ticket(title: string, priority: \"low\"|\"normal\"|\"high\"|\"urgent\", room: string)\n"
             "- order_stock(item: string, quantity: integer, depot: string)\n- schedule_visit(person: string, date: \"YYYY-MM-DD\", hour: integer)\n"
             "- convert_units(value: number, from_unit: string, to_unit: string)\n"
             "Reply with exactly one JSON object {\"tool\": <name>, \"args\": {...}} using exactly the argument names above, and nothing else.")


def tool_call(r, i):
    kind = i % 4
    name = r.choice(PEOPLE)
    city = r.choice(CITIES)
    item = r.choice(ITEMS)
    n = r.randint(4, 90)
    day = r.randint(2, 28)
    if kind == 0:
        prio = r.choice(PRIOS)
        room = "%s-%d" % (r.choice("ABCD"), r.randint(100, 420))
        title = r.choice(["broken window", "printer offline", "water leak", "light flickering"])
        return ("Open a ticket titled \"%s\" for room %s with %s priority." % (title, room, prio),
                {"tool": "create_ticket", "args": {"title": title, "priority": prio, "room": room}})
    if kind == 1:
        return ("We are almost out. Get %d %s delivered to the %s depot; last time we ordered %d, which was too few." % (n, item, city, n - 3),
                {"tool": "order_stock", "args": {"item": item, "quantity": n, "depot": city}})
    if kind == 2:
        hour = r.randint(8, 17)
        return ("Plan a visit for %s on %d May 2027 at %d o'clock (24-hour clock)." % (name, day, hour),
                {"tool": "schedule_visit", "args": {"person": name, "date": "2027-05-%02d" % day, "hour": hour}})
    a, b = r.choice([("miles", "kilometers"), ("celsius", "fahrenheit"), ("kilograms", "pounds"), ("liters", "gallons")])
    return ("Convert %d %s to %s." % (n, a, b), {"tool": "convert_units", "args": {"value": n, "from_unit": a, "to_unit": b}})


def json_tasks(r):
    out = []
    for i in range(20):
        content, want = json_extract(r, i)
        out.append({"id": "json%02d" % (i + 1), "cat": "json", "max_tokens": 220, "messages": [{"role": "user", "content": content}],
                    "check": {"type": "json", "value": want}})
    for i in range(20):
        user, want = tool_call(r, i)
        out.append({"id": "json%02d" % (i + 21), "cat": "json", "max_tokens": 120,
                    "messages": [{"role": "system", "content": TOOLS_DOC}, {"role": "user", "content": user}], "check": {"type": "json", "value": want}})
    return out


# ---------------------------------------------------------------- long context
PROJECTS = ["Aurora", "Basalt", "Cobalt", "Delta", "Ember", "Fjord", "Granite", "Harbor", "Indigo", "Juniper", "Kestrel", "Lagoon", "Meridian",
            "Nimbus", "Onyx", "Pioneer", "Quartz", "Ridge", "Summit", "Tundra", "Umber", "Vertex", "Willow", "Xenon", "Yarrow", "Zephyr", "Amber", "Birch",
            "Cedar", "Dune", "Estuary", "Falcon", "Glacier", "Heron", "Ivory", "Jasper", "Kelp", "Larch", "Marble", "Nettle", "Orchid", "Pebble"]
FILLER = ["The weekly meeting ran long and the notes were shared afterwards.", "Nobody objected to the revised planning for the next sprint.",
          "The coffee machine on the second floor was serviced again.", "A short demo showed the new dashboard to the support team.",
          "The supplier confirmed that the earlier shipment had arrived.", "Several people asked for clearer meeting minutes.",
          "The archive was moved to a new folder structure without incidents.", "Training sessions will continue after the holiday period.",
          "The building access system was tested during the weekend.", "Feedback from the pilot users was mostly about small layout issues."]


def long_task(r, i):
    projects = r.sample(PROJECTS, 36)
    leads = [r.choice(PEOPLE) + " " + r.choice(["de Vries", "Jansen", "Bakker", "Visser", "Smit", "Mulder", "de Boer", "Bos"]) for _ in projects]
    codes = r.sample(range(10000, 99999), len(projects))
    budgets = r.sample(range(12, 940), len(projects))
    paras = []
    for p, lead, code, budget in zip(projects, leads, codes, budgets):
        sents = ["Project %s is led by %s." % (p, lead), "Its release code is %s-%d." % (p[:3].upper(), code),
                 "The approved budget of project %s is %d thousand euros." % (p, budget)]
        sents += r.sample(FILLER, r.randint(2, 4))
        r.shuffle(sents)
        paras.append(" ".join(sents))
    size = [23, 29, 36][i % 3]   # roughly 1600, 2000 and 2500 tokens
    paras = paras[:size]
    idx = r.randrange(size)
    kind = i % 4
    p, lead, code, budget = projects[idx], leads[idx], codes[idx], budgets[idx]
    if kind == 0:
        q, want = "What is the release code of project %s?" % p, "%s-%d" % (p[:3].upper(), code)
    elif kind == 1:
        q, want = "Who leads project %s?" % p, lead
    elif kind == 2:
        q, want = "The project whose release code is %s-%d has what approved budget, in thousands of euros? Answer with the number." % (p[:3].upper(), code), str(budget)
    else:
        j = (idx + 1 + r.randrange(size - 1)) % size
        a, b = budgets[idx], budgets[j]
        q = "What is the sum of the approved budgets of projects %s and %s, in thousands of euros? Answer with the number." % (p, projects[j])
        want = str(a + b)
    doc = "\n\n".join(paras)
    return {"id": "long%02d" % (i + 1), "cat": "long", "max_tokens": 60,
            "messages": [{"role": "user", "content": "Read the status report below and answer the question at the end. Answer with only the value.\n\n"
                                                     "%s\n\nQuestion: %s" % (doc, q)}],
            "check": {"type": "value", "value": want}}


def main():
    r = random.Random(SEED)
    tasks = math_tasks(r) + code_tasks() + json_tasks(r) + [long_task(r, i) for i in range(40)]
    assert len(tasks) == 160 and len({t["id"] for t in tasks}) == 160
    with open(os.path.join(HERE, "tasks.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print("wrote %d tasks" % len(tasks))


if __name__ == "__main__":
    main()
