"""Charts for the InferTutor Arena report. Reads the result JSONs; writes vector PDFs."""

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

RESULTS = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

# Validated categorical slots (dataviz reference palette, light mode) + neutral + accent.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, RED = "#9a9994", "#e34948"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "figure.facecolor": SURFACE,
})


def score(d):
    r = d["results"]
    return (r["aggregate_stream_chunks_per_s"] * d["config"]["users"] * (1 - r["error_rate"])
            / ((r["ttft_p95_ms"] / 1000) * (r["itl_p95_ms"] / 1000) * d["config"]["total_gpus"]))


runs = {}
for f in glob.glob(str(RESULTS / "*.json")):
    d = json.load(open(f))
    runs[d["config"]["label"]] = d


def r(label, key):
    return runs[label]["results"][key]


def group(label):
    s = runs[label]["config"]["serving"]
    if label in ("m-final", "m-compiled-api2"):
        return "Compiled + 2 API servers"
    if not s["fast_boot"]:
        return "Compiled, 1 API server"
    if s["max_seqs"] == 32:
        return "Default (32 seats)"
    return "Eager, more seats"


COLORS = {"Compiled + 2 API servers": BLUE, "Eager, more seats": ORANGE,
          "Compiled, 1 API server": AQUA, "Default (32 seats)": GRAY}


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def human(v):
    return f"{v / 1e9:.2f}B" if v >= 1e9 else f"{v / 1e6:.0f}M" if v >= 1e6 else f"{v:,.0f}"


# 1. All 21 runs, sorted by score (log scale), coloured by configuration family.
labels = sorted(runs, key=lambda l: score(runs[l]))
fig, ax = plt.subplots(figsize=(6.4, 4.3))
ys = range(len(labels))
for y, l in zip(ys, labels):
    ax.barh(y, score(runs[l]), color=COLORS[group(l)], height=0.72, edgecolor=SURFACE, linewidth=1)
    ax.text(score(runs[l]) * 1.12, y, human(score(runs[l])), va="center", fontsize=7.5, color=INK)
ax.set_yticks(list(ys), [f"{l}  ({runs[l]['config']['users']}u)" for l in labels], fontsize=7.5)
ax.set_xscale("log")
ax.set_xlim(1e6, 3e10)
ax.set_xlabel("Score (log scale)")
ax.grid(axis="y", visible=False)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: human(v)))
ax.legend(handles=[Patch(color=c, label=n) for n, c in COLORS.items()], loc="lower right", fontsize=7.5)
save(fig, "fig_scores.pdf")

# 2. User sweep: 32 seats vs more seats. Throughput and TTFT p95 as two panels (no dual axis).
default = ["m-base-50u", "m-base-100u", "m-base-150u"]
seats = ["m-s128-100u", "m-s128-150u", "m-s256-200u", "m-s256-300u", "m-s512-375u",
         "m-s512-450u", "m-s512-600u"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.5))
for series, color, name in ((default, GRAY, "32 seats (default)"),
                            (seats, ORANGE, "128-512 seats (eager)")):
    users = [runs[l]["config"]["users"] for l in series]
    a1.plot(users, [r(l, "aggregate_stream_chunks_per_s") for l in series], color=color, lw=2,
            marker="o", ms=5, mec=SURFACE, mew=1.2, label=name)
    a2.plot(users, [r(l, "ttft_p95_ms") for l in series], color=color, lw=2,
            marker="o", ms=5, mec=SURFACE, mew=1.2, label=name)
a1.set_title("Throughput (chunks/s) - higher is better", fontsize=8.5, color=INK, loc="left")
a2.set_title("TTFT p95 (ms) - lower is better", fontsize=8.5, color=INK, loc="left")
for a in (a1, a2):
    a.set_xlabel("Concurrent users")
    a.set_xlim(0, 640)
a1.annotate("flat: requests wait\nfor a free seat", xy=(150, 1757), xytext=(230, 3300),
            fontsize=7, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
a1.annotate("ceiling ~10.2k", xy=(600, 10266), xytext=(390, 6700),
            fontsize=7, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
a2.annotate("5.7 s queue", xy=(150, 5706), xytext=(240, 5000), fontsize=7, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
a2.annotate("knee: 300-450 users", xy=(450, 918), xytext=(250, 2700), fontsize=7, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
fig.tight_layout(w_pad=2, rect=(0, 0, 1, 0.9))
fig.legend(*a1.get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=7.5)
save(fig, "fig_sweep.pdf")

# 3. Where the CPU time went (top during the load phase). Small multiples, one per stage.
stages = [
    ("Eager, 1 API server", [("EngineCore", 100.0), ("API server", 81.0), ("Load tester", 82.7)]),
    ("Compiled, 1 API server", [("EngineCore", 108.0), ("API server", 100.3), ("Load tester", 87.0)]),
    ("Compiled, 2 API servers", [("EngineCore", 109.0), ("API servers", 82.1), ("Load tester", 99.3)]),
]
walls = {0: "EngineCore", 1: "API server", 2: "Load tester"}
fig, axes = plt.subplots(1, 3, figsize=(6.4, 1.9), sharey=True)
for i, (ax, (title, procs)) in enumerate(zip(axes, stages)):
    names = [p for p, _ in procs]
    vals = [v for _, v in procs]
    cols = [RED if n.startswith(walls[i]) else GRAY for n in names]
    ax.barh(range(3), vals, color=cols, height=0.62)
    for y, (n, v) in enumerate(procs):
        txt = "82% + 72%" if n == "API servers" else f"{v:.0f}%"
        ax.text(v + 4, y, txt, va="center", fontsize=7, color=INK)
    ax.axvline(100, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    ax.set_yticks(range(3), names)
    ax.set_xlim(0, 150)
    ax.set_ylim(2.5, -0.85)
    ax.set_xticks([0, 50, 100])
    ax.grid(axis="y", visible=False)
axes[0].text(102, -0.55, "1 core", fontsize=6.5, color=INK2)
axes[1].set_xlabel("% CPU from top (100% = one full core)")
save(fig, "fig_cpu.pdf")

# 4. The compiled-mode trade-off: TTFT and ITL as two panels.
cfgs = [("Eager\n(more seats)", "m-s256-300u-r2", ORANGE),
        ("Compiled\n1 API server", "m-compiled", AQUA),
        ("Compiled\n2 API servers", "m-final", BLUE)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.2))
for ax, key, title in ((a1, "ttft_p95_ms", "TTFT p95 (ms) - lower is better"),
                       (a2, "itl_p95_ms", "ITL p95 (ms) - lower is better")):
    vals = [r(l, key) for _, l, _ in cfgs]
    ax.bar(range(3), vals, color=[c for *_, c in cfgs], width=0.6, edgecolor=SURFACE, linewidth=1)
    for x, v in enumerate(vals):
        ax.text(x, v + max(vals) * 0.03, f"{v:.0f}" if v > 50 else f"{v:.1f}", ha="center",
                fontsize=7.5, color=INK)
    ax.set_xticks(range(3), [n for n, *_ in cfgs], fontsize=7.5)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="x", visible=False)
fig.tight_layout(w_pad=2)
save(fig, "fig_tradeoff.pdf")

# 5. Knob toggles vs the noise of one repeated config (all eager, 256 seats, 300 users).
tog = [("Baseline, run 1", "m-s256-300u"), ("Baseline, run 2", "m-s256-300u-r2"),
       ("Prefix cache OFF", "m-nopc"), ("Prefix cache OFF, run 2", "m-nopc-r2"),
       ("Chunked prefill OFF", "m-nocp"), ("Batch tokens 2048", "m-s32-b2048"),
       ("Batch tokens 8192", "m-b8192")]
lo, hi = score(runs["m-s256-300u"]), score(runs["m-s256-300u-r2"])
fig, ax = plt.subplots(figsize=(6.4, 2.2))
ax.axvspan(lo / 1e6, hi / 1e6, color=ORANGE, alpha=0.12, lw=0)
ax.text((lo + hi) / 2e6, -0.95, "same config, two runs: +34%", ha="center", fontsize=7, color=INK2)
for y, (n, l) in enumerate(tog):
    v = score(runs[l]) / 1e6
    ax.barh(y, v, color=ORANGE if l.startswith("m-s256") else GRAY, height=0.62)
    ax.text(v + 8, y, f"{v:.0f}M", va="center", fontsize=7.5, color=INK)
ax.set_yticks(range(len(tog)), [n for n, _ in tog])
ax.invert_yaxis()
ax.set_ylim(len(tog) - 0.5, -1.4)
ax.set_xlim(0, 900)
ax.set_xlabel("Score (millions)")
ax.grid(axis="y", visible=False)
save(fig, "fig_noise.pdf")

print("ok", sorted(p.name for p in OUT.glob("*.pdf")))
