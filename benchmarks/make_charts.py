#!/usr/bin/env python3
"""Draws the README charts from the committed measurement files only (matplotlib). No number is typed into this script: speeds, scores, memory and
sizes come from battle_3way.json, the output-head figures from apex_feasibility.json, and the same-session cross-check from the per-round raw
files (<results>/raw/ik2/speed/, skipped when a run of your own has none). It also writes improvements.json: every percentage and every
paired p-value shown in the charts.

  python3 benchmarks/make_charts.py --results benchmarks/results --out docs/assets
"""
import argparse
import json
import math
import os
import statistics
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def load(*parts):
    path = os.path.join(a.results, *parts)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


battle = load("battle_3way.json")
if battle is None:
    raise SystemExit("no battle_3way.json in %s (run benchmarks/assemble.py first)" % a.results)
E = {e["id"]: e for e in battle["entries"]}
HW = battle["hardware"]

# one fixed colour per engine in every chart (checked for colour-vision deficiency on a white surface; every bar also carries its value)
ORDER = ["veltra", "veltra-pm", "llamacpp", "ikllama", "ikllama-rtr"]
NAME = {"veltra": "Veltra default", "veltra-pm": "Veltra --pareto-max", "llamacpp": "llama.cpp", "ikllama": "ik_llama.cpp",
        "ikllama-rtr": "ik_llama.cpp -rtr"}
COLOR = {"veltra": "#257de0", "veltra-pm": "#f47230", "llamacpp": "#00b894", "ikllama": "#f2a600", "ikllama-rtr": "#ee7fac"}
INK, INK2, GRID, AXIS = "#191919", "#555555", "#e7e7e7", "#9a9a9a"
CTX = ["ctx0", "ctx500", "ctx2000"]
# an opaque white surface keeps every chart readable on GitHub's light and dark themes
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "svg.fonttype": "none", "svg.hashsalt": "veltra"})

speed_ids = [i for i in ORDER if E.get(i) and E[i].get("speed")]
quality = {i: E[i]["quality"] for i in ORDER if E.get(i) and E[i].get("quality")}


def decode(i, c):
    return statistics.median(E[i]["speed"][c]["decode_all"])


def pp(i, c):
    return E[i]["speed"][c]["pp_tok_s"]


def pct(x, y):
    return round(100.0 * (x / y - 1.0), 1)


def sign_test(x, y):
    """Exact two-sided sign test on the tasks that exactly one of the two runs passed."""
    fx, fy = set(quality[x]["failed"]), set(quality[y]["failed"])
    only_x, only_y = len(fy - fx), len(fx - fy)
    n = only_x + only_y
    p = min(1.0, 2.0 * sum(math.comb(n, k) for k in range(min(only_x, only_y) + 1)) / 2.0 ** n) if n else 1.0
    return {"only_first_passed": only_x, "only_second_passed": only_y, "p": round(p, 4)}


def fmt_p(p):
    return "%.3f" % p if p < 0.01 else "%.2f" % p


def ctx_labels():
    ref = E[speed_ids[0]]["speed"]
    return ["%d-token prompt" % round(ref[c]["prompt_tokens"]) for c in CTX]


def same_session():
    """Default against ik_llama.cpp from the rounds that were interleaved with each other (their own Veltra default run)."""
    out = {}
    for eng in ("veltra", "ikllama", "ikllama_rtr"):
        rounds = [load("raw", "ik2", "speed", "speed_%s_r%d.json" % (eng, r)) for r in (1, 2)]
        if any(r is None for r in rounds):
            return None
        out[eng] = {c: {"decode_tok_s": round(statistics.median([x for r in rounds for x in r["contexts"][c]["decode_all"]]), 2),
                        "pp_tok_s": round(statistics.median([r["contexts"][c]["pp_tok_s"] for r in rounds]), 2)} for c in CTX}
    return out


SAME = same_session()
loads = [x for i in speed_ids for x in E[i].get("speed_load", [])]
reps = len(E[speed_ids[0]]["speed"]["ctx0"]["decode_all"])
RUN_NOTE = ("%s. %d repetitions per engine and prompt length, 128 new tokens, greedy; bars are medians." % (HW, reps))
FILE_NOTE = ("Model files differ: llama.cpp and ik_llama.cpp ran a GGUF with higher-precision attention; Veltra ran its own file (same "
             "expert weights, mostly 3-bit q/o attention, int8 output head and embeddings). Part of the speed difference comes from the file.")


def score_note():
    if not all(i in quality for i in ("veltra", "veltra-pm", "llamacpp", "ikllama")):
        return ""
    s = {i: quality[i]["passed"] for i in quality}
    return ("The GGUF engines answered more of the %d tasks correctly: %d (llama.cpp) and %d (ik_llama.cpp) against %d (Veltra default) and "
            "%d (--pareto-max). --pareto-max changes answers." % (quality["veltra"]["total"], s["llamacpp"], s["ikllama"], s["veltra"],
                                                                 s["veltra-pm"]))


def caption(fig, notes, width=150):
    """Wraps the notes below the plot; returns the height in inches that they need."""
    lines = []
    for n in notes:
        if n:
            lines += textwrap.wrap(n, width)
    fig.text(0.015, 0.012, "\n".join(lines), ha="left", va="bottom", fontsize=8.5, color=INK2, linespacing=1.45)
    return 0.155 * len(lines) + 0.18


def frame(fig, title, subtitle, notes, left=0.07, right=0.985, legend=None, ncol=5, width=150, **adjust):
    w, h = fig.get_size_inches()
    cap = caption(fig, notes, width)
    fig.text(0.015, 1 - 0.12 / h, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.015, 1 - 0.45 / h, subtitle, ha="left", va="top", fontsize=10, color=INK2)
    top = 1.05
    if legend:
        fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.008, 1 - 0.68 / h), ncol=ncol, frameon=False, fontsize=9,
                   handlelength=1.2, columnspacing=1.4)
        top = 1.35
    fig.subplots_adjust(left=left, right=right, top=1 - top / h, bottom=(cap + 0.45) / h, **adjust)


def save(fig, name):
    for ext in ("png", "svg"):
        meta = {"Date": None} if ext == "svg" else {"Software": None}
        fig.savefig(os.path.join(a.out, "%s.%s" % (name, ext)), dpi=170, metadata=meta)
    plt.close(fig)
    print("wrote", name)


def engine_legend(ids):
    return [Patch(color=COLOR[i], label=NAME[i] + (" (changes answers)" if i == "veltra-pm" else "")) for i in ids]


def grouped_bars(ax, ids, value, spread=None):
    w = 0.84 / len(ids)
    top = 0.0
    for n, i in enumerate(ids):
        for j, c in enumerate(CTX):
            x, v = j - 0.42 + w * (n + 0.5), value(i, c)
            ax.bar(x, v, w - 0.018, color=COLOR[i])
            hi = v
            if spread:
                lo, hi = spread(i, c)
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5, capthick=0.9)
            top = max(top, hi)
            ax.annotate("%.1f" % v if v < 40 else "%.0f" % v, (x, hi), textcoords="offset points", xytext=(0, 2), ha="center",
                        va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(CTX)))
    ax.set_xticklabels(ctx_labels())
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, top * 1.12)
    ax.grid(axis="y")


# 1) decode speed
fig, ax = plt.subplots(figsize=(10, 6.0))
grouped_bars(ax, speed_ids, decode, spread=lambda i, c: (min(E[i]["speed"][c]["decode_all"]), max(E[i]["speed"][c]["decode_all"])))
ax.set_ylabel("tokens per second")
cross = ("The Veltra and llama.cpp bars come from one benchmark session, the two ik_llama.cpp bars from a second one on the same machine; "
         "see the percentage chart for the same-session comparison." if SAME else "")
frame(fig, "Decode speed: Qwen3-30B-A3B on CPU", "Tokens per second at three prompt lengths; higher is better. Whiskers: slowest and fastest repetition.",
      [RUN_NOTE, cross, FILE_NOTE, score_note()], legend=engine_legend(speed_ids))
save(fig, "decode_speed")

# 2) prompt processing
fig, ax = plt.subplots(figsize=(10, 5.9))
grouped_bars(ax, speed_ids, pp)
ax.set_ylabel("tokens per second")
frame(fig, "Prompt processing speed", "Tokens of prompt read per second; higher is better. Median of the per-round medians (two rounds).",
      [RUN_NOTE, cross, FILE_NOTE, score_note()],
      legend=engine_legend(speed_ids))
save(fig, "prompt_processing")

# 3) improvement in percent
improvements = {"source": "computed by benchmarks/make_charts.py from battle_3way.json and raw/ik2/speed/; percent = (Veltra / other - 1) * 100",
                "published_medians": {}, "same_session_default": None, "paired_sign_tests": {}}
bases = [b for b in ("llamacpp", "ikllama", "ikllama-rtr") if b in speed_ids]
modes = [m for m in ("veltra", "veltra-pm") if m in speed_ids]
for m in modes:
    for b in bases:
        improvements["published_medians"]["%s_vs_%s" % (m, b)] = {
            "decode_pct": {c: pct(decode(m, c), decode(b, c)) for c in CTX}, "pp_pct": {c: pct(pp(m, c), pp(b, c)) for c in CTX}}
if SAME:
    improvements["same_session_default"] = {
        "medians": SAME,
        **{"veltra_vs_%s" % key.replace("_", "-"): {
            "decode_pct": {c: pct(SAME["veltra"][c]["decode_tok_s"], SAME[key][c]["decode_tok_s"]) for c in CTX},
            "pp_pct": {c: pct(SAME["veltra"][c]["pp_tok_s"], SAME[key][c]["pp_tok_s"]) for c in CTX}} for key in ("ikllama", "ikllama_rtr")}}
for x, y in (("veltra", "llamacpp"), ("veltra", "ikllama"), ("veltra-pm", "llamacpp"), ("veltra-pm", "ikllama"), ("veltra-pm", "veltra")):
    if x in quality and y in quality:
        improvements["paired_sign_tests"]["%s_vs_%s" % (x, y)] = sign_test(x, y)
with open(os.path.join(a.results, "improvements.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(improvements, f, indent=2)
    f.write("\n")

cols = [b for b in ("llamacpp", "ikllama-rtr") if b in speed_ids]
if cols and modes:
    fig, axes = plt.subplots(2, len(cols), figsize=(11, 8.9), sharey="row", squeeze=False)
    for r, (key, rowname) in enumerate((("decode_pct", "decode"), ("pp_pct", "prompt processing"))):
        vals = [improvements["published_medians"]["%s_vs_%s" % (m, b)][key][c] for m in modes for b in cols for c in CTX]
        if SAME:
            vals += list(improvements["same_session_default"]["veltra_vs_ikllama-rtr"][key].values())
        for k, b in enumerate(cols):
            ax = axes[r][k]
            for n, m in enumerate(modes):
                for j, c in enumerate(CTX):
                    v = improvements["published_medians"]["%s_vs_%s" % (m, b)][key][c]
                    x = j + (-0.2 if n == 0 else 0.2)
                    ax.bar(x, v, 0.37, color=COLOR[m])
                    y_text = max(v, 0)
                    if SAME and m == "veltra" and b == "ikllama-rtr":
                        s = improvements["same_session_default"]["veltra_vs_ikllama-rtr"][key][c]
                        ax.plot([x], [s], marker="D", markersize=6.5, color=INK, markeredgecolor="white", markeredgewidth=1.2, zorder=5)
                        y_text = max(y_text, s)
                        ax.annotate("(%+.1f)" % s, (x, y_text), textcoords="offset points", xytext=(0, 15), ha="center", va="bottom",
                                    fontsize=8.5, color=INK2)
                    ax.annotate("%+.1f%%" % v, (x, y_text), textcoords="offset points", xytext=(0, 4), ha="center", va="bottom",
                                fontsize=9, color=INK)
            ax.axhline(0, color=INK, linewidth=0.9)
            ax.set_ylim(min(0.0, min(vals) - 4.0 if min(vals) < 0 else 0.0), max(vals) * 1.22)
            ax.set_xticks(range(len(CTX)))
            ax.set_xticklabels(ctx_labels(), fontsize=9)
            ax.tick_params(axis="x", length=0)
            ax.grid(axis="y")
            ax.set_title("%s: Veltra vs %s" % (rowname.capitalize(), NAME[b]), fontsize=10.5, color=INK, loc="left")
            if k == 0:
                ax.set_ylabel("difference in percent")
    leg = [Patch(color=COLOR[m], label=NAME[m] + (" (changes answers)" if m == "veltra-pm" else "")) for m in modes]
    if SAME:
        leg.append(Line2D([], [], marker="D", linestyle="", color=INK, markersize=6.5, label="default, same-session pair (value in brackets)"))
    pnote = ""
    if all(k in improvements["paired_sign_tests"] for k in ("veltra-pm_vs_llamacpp", "veltra-pm_vs_ikllama")):
        t = improvements["paired_sign_tests"]
        pnote = ("--pareto-max passes fewer tasks than the GGUF engines and the difference is statistically significant (exact sign test, "
                 "p = %s against llama.cpp, p = %s against ik_llama.cpp without -rtr; -rtr was not scored)."
                 % (fmt_p(t["veltra-pm_vs_llamacpp"]["p"]), fmt_p(t["veltra-pm_vs_ikllama"]["p"])))
    xnote = ("Against -rtr the bars compare medians from two different sessions on the same shared machine. For the default mode the diamonds "
             "use the Veltra run that was interleaved with -rtr; --pareto-max has no same-session -rtr run. Differences of a few percent are "
             "within run-to-run noise." if SAME else "Differences of a few percent can be run-to-run noise.")
    frame(fig, "Speed difference in percent", "(Veltra / other engine - 1) x 100, from the median speeds. 0% = equal; below 0 = Veltra slower.",
          [RUN_NOTE, xnote, FILE_NOTE, score_note(), pnote], left=0.075, legend=leg, ncol=3, width=165, hspace=0.34, wspace=0.06)
    save(fig, "improvement_percent")

# 4) quality right next to speed
qids = [i for i in speed_ids]
if quality:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    ys = range(len(qids))
    for y, i in zip(ys, qids):
        v = decode(i, "ctx500")
        a1.barh(y, v, 0.64, color=COLOR[i])
        a1.text(v, y, "  %.1f" % v, va="center", fontsize=9.5)
        q = quality.get(i)
        if q:
            a2.barh(y, q["passed"], 0.64, color=COLOR[i])
            a2.text(q["passed"], y, "  %d (%.1f%%)" % (q["passed"], q["pct"]), va="center", fontsize=9.5)
        else:
            a2.text(2, y, "quality not run", va="center", fontsize=9, color=INK2, style="italic")
    a1.set_yticks(list(ys))
    a1.set_yticklabels([NAME[i] for i in qids], fontsize=10, color=INK)
    a1.invert_yaxis()
    total = next(iter(quality.values()))["total"]
    a1.set_xlim(0, max(decode(i, "ctx500") for i in qids) * 1.17)
    a2.set_xlim(0, total * 1.2)
    a2.set_xticks(range(0, total + 1, 40))
    for ax, title, unit in ((a1, "Speed", "decode tokens per second, %s" % ctx_labels()[1]),
                            (a2, "Quality", "tasks answered correctly, out of %d" % total)):
        ax.set_title(title, fontsize=11.5, loc="left", fontweight="bold")
        ax.set_xlabel(unit)
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
    t = improvements["paired_sign_tests"]
    stats = ""
    if len([k for k in t if k.endswith(("llamacpp", "ikllama"))]) == 4:
        stats = ("Exact sign test on the same tasks: default against llama.cpp p = %s, against ik_llama.cpp p = %s (not distinguishable from "
                 "chance). --pareto-max against llama.cpp p = %s, against ik_llama.cpp p = %s (significant). Single comparisons, not corrected."
                 % (fmt_p(t["veltra_vs_llamacpp"]["p"]), fmt_p(t["veltra_vs_ikllama"]["p"]), fmt_p(t["veltra-pm_vs_llamacpp"]["p"]),
                    fmt_p(t["veltra-pm_vs_ikllama"]["p"])))
    frame(fig, "Read speed and quality together", "--pareto-max is faster because it does less work per token, and it changes answers.",
          [RUN_NOTE, "Quality: the public 160-task set in benchmarks/eval_public/, greedy, thinking off. A small set that compares settings, not a "
           "general intelligence benchmark.", stats, FILE_NOTE], left=0.145, width=165, wspace=0.08)
    save(fig, "quality_vs_speed")

# 5) memory and footprint
mem = [i for i in ORDER if E.get(i) and E[i].get("peak_rss_gb") and E[i].get("file_bytes")]
if mem:
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11.5, 5.7), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    for y, i in enumerate(mem):
        v, long_run = E[i]["peak_rss_gb"], E[i].get("eval_peak_rss_gb")
        a1.barh(y - 0.19, v, 0.36, color=COLOR[i])
        a1.text(v, y - 0.19, " %.1f" % v, va="center", fontsize=9)
        if long_run:
            a1.barh(y + 0.19, long_run, 0.36, color="white", edgecolor=COLOR[i], hatch="////", linewidth=1.0)
            a1.text(long_run, y + 0.19, " %.1f" % long_run, va="center", fontsize=9)
        else:
            a1.text(0.3, y + 0.19, "160-task run not done", va="center", fontsize=8, color=INK2, style="italic")
        for ax, val, f in ((a2, (E[i].get("binary_bytes") or 0) / 1e6, " %.1f"), (a3, E[i]["file_bytes"] / 1e9, " %.2f")):
            ax.barh(y, val, 0.62, color=COLOR[i])
            ax.text(val, y, f % val, va="center", fontsize=9)
    for ax, title, unit, key in ((a1, "Peak RAM", "GiB (lower is better)",
                                  lambda i: max(E[i]["peak_rss_gb"], E[i].get("eval_peak_rss_gb") or 0)),
                                 (a2, "Server binary + own libraries", "MB (1 MB = 1,000,000 bytes)",
                                  lambda i: (E[i].get("binary_bytes") or 0) / 1e6),
                                 (a3, "Model file used", "GB (1 GB = 10^9 bytes)", lambda i: E[i]["file_bytes"] / 1e9)):
        ax.set_yticks(range(len(mem)))
        ax.set_yticklabels([NAME[i] for i in mem] if ax is a1 else [], fontsize=9.5, color=INK)
        ax.set_ylim(len(mem) - 0.45, -0.55)
        ax.set_xlim(0, max(key(i) for i in mem) * 1.2)
        ax.set_title(title, fontsize=10.5, loc="left", fontweight="bold")
        ax.set_xlabel(unit, fontsize=9)
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
    leg = [Patch(facecolor="#777777", label="RAM: solid = peak during the speed run (same short requests for every engine)"),
           Patch(facecolor="white", edgecolor="#777777", hatch="////", label="RAM: hatched = peak during the 160-task run")]
    frame(fig, "Memory and size", "Peak resident memory of the server process (includes the mapped model pages), program size and model file.",
          [RUN_NOTE.split(". %d repetitions" % reps)[0] + ".",
           "In the 160-task run llama.cpp and ik_llama.cpp grew because llama-server keeps a prompt cache in memory by default (limit 8 GiB); "
           "--cache-ram 0 turns it off and was not measured. The -rtr rows use the same binary and file as ik_llama.cpp.", FILE_NOTE],
          left=0.135, legend=leg, ncol=2, width=175, wspace=0.1)
    save(fig, "memory_and_footprint")

# 6) one technique, measured on its own: screening the output head
feas = load("apex_feasibility.json")
if feas and feas.get("C_lm_head"):
    cfg = feas["C_lm_head"]["configs"]
    keys = [("unscreened", "full int8 head"), ("screen4", "4-bit screen, then exact"), ("screen2", "2-bit screen, then exact")]
    shade = ["#9a9a9a", "#84b9f3", "#257de0"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for y, ((k, label), col) in enumerate(zip(keys, shade)):
        a1.barh(y, cfg[k]["head_ms_per_token"], 0.62, color=col)
        a1.text(cfg[k]["head_ms_per_token"], y, "  %.2f ms (%.1f%% of the token)" % (cfg[k]["head_ms_per_token"], cfg[k]["head_share_pct"]),
                va="center", fontsize=9)
        a2.barh(y, cfg[k]["decode_tok_s"], 0.62, color=col)
        a2.text(cfg[k]["decode_tok_s"], y, "  %.2f" % cfg[k]["decode_tok_s"], va="center", fontsize=9)
    a1.set_yticks(range(len(keys)))
    a1.set_yticklabels([label for _, label in keys], color=INK)
    a1.invert_yaxis()
    a1.set_xlim(0, max(cfg[k]["head_ms_per_token"] for k, _ in keys) * 2.0)
    a2.set_xlim(0, max(cfg[k]["decode_tok_s"] for k, _ in keys) * 1.17)
    for ax, title, unit in ((a1, "Time in the output head", "milliseconds per token (lower is better)"),
                            (a2, "Decode speed", "tokens per second (higher is better)")):
        ax.set_title(title, fontsize=10.5, loc="left", fontweight="bold")
        ax.set_xlabel(unit, fontsize=9)
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
    frame(fig, "Screened output head", "All experts, only the output-head setting changes. A separate probe: do not compare with the other charts.",
          ["Qwen3-30B-A3B, AMD Ryzen 9 3900, CPU only, 8 threads, shared machine (load %.1f), median of %d repetitions of %d tokens. Source: "
           "benchmarks/results/apex_feasibility.json." % (feas["C_lm_head"]["load"], feas["C_lm_head"]["reps"],
                                                          cfg["unscreened"]["generated_tokens"]),
           "Screening uses a probabilistic error margin: there is no proof that it picks the same token as the full head for every input."],
          left=0.2, wspace=0.08, width=150)
    save(fig, "output_head")
