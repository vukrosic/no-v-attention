"""draws the README charts from the numbers in RESULTS.md"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURF, INK, INK2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 12, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "figure.facecolor": SURF, "axes.facecolor": SURF})


def clean(ax):
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


# 1. the main result: four seeds per setup, realistic task suite, width 64
rows = [("Normal attention", [1.073, 1.100, 1.073, 1.089], INK2),
        ("+ gate on the value embedding", [1.039, 1.039, 1.029, 1.069], INK2),
        ("4 slots averaged evenly (control)", [1.055, 1.059, 1.038, 1.027], INK2),
        ("4 context-chosen slots,\nno value matrix", [1.019, 0.994, 0.978, 0.992], BLUE)]
fig, ax = plt.subplots(figsize=(9, 3.8))
for i, (name, v, c) in enumerate(rows[::-1]):
    m = sum(v) / len(v)
    ax.barh(i, m - 0.95, left=0.95, height=0.55, color=c if c == BLUE else "#b9b8b2")
    ax.scatter(v, [i] * 4, s=22, color=INK, zorder=3)
    ax.text(max(v) + 0.004, i, f"mean {m:.3f}", color=INK, fontsize=11, va="center")
ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]]); ax.set_xlim(0.95, 1.14)
ax.set_xlabel("test loss, lower is better (dots: 4 seeds)")
ax.set_title("Value embedding with context-chosen rows vs the value matrix", loc="right", fontsize=14, color=INK)
clean(ax); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("main_result.png", dpi=160); plt.close(fig)

# 2. what the second-layer picker chooses, per task (share of positions whose top slot is each one)
tasks = [("copy a repeated part", [0.00, 0.00, 1.00, 0.00]), ("repeat the word 2 back", [0.00, 0.00, 0.99, 0.00]),
         ("key → value lookup", [0.00, 0.01, 0.98, 0.02]), ("sort words", [0.00, 0.12, 0.88, 0.00]),
         ("common fact", [0.00, 0.99, 0.01, 0.00]), ("look up, then common fact", [0.00, 1.00, 0.00, 0.00]),
         ("rare fact", [0.00, 0.00, 0.00, 1.00]), ("guess the next word of text", [0.45, 0.22, 0.04, 0.29]),
         ("count zeros", [0.88, 0.11, 0.00, 0.00])]
cm = LinearSegmentedColormap.from_list("b", [SURF, "#cfe0f6", BLUE, "#0f3f7a"])
fig, ax = plt.subplots(figsize=(7.6, 4.6))
M = [t[1] for t in tasks]; ax.imshow(M, cmap=cm, vmin=0, vmax=1, aspect="auto")
for i, r in enumerate(M):
    for j, v in enumerate(r):
        if v >= 0.1: ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=10, color=SURF if v > 0.6 else INK)
ax.set_yticks(range(len(tasks)), [t[0] for t in tasks])
ax.set_xticks(range(4), ["slot 1", "slot 2", "slot 3", "slot 4"]); ax.xaxis.tick_top()
for y in (3.5, 6.5): ax.axhline(y, color=SURF, lw=3)
ax.set_title("Second layer: which slot each task uses", loc="left", fontsize=14, color=INK, pad=28)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0); fig.tight_layout(); fig.savefig("slots.png", dpi=160); plt.close(fig)

# 3. loss change in different training situations (mean over 13 tasks; 2-3 seeds)
sit = ["default\n(2 layers, 256 words, 64 facts)", "4 layers", "1024 words", "256 facts, some asked\nmuch more often", "3x longer training", "all three: 1024 words,\n256 uneven facts, 3x training", "4x wider (width 256)"]
nov = [-0.105, -0.096, -0.202, -0.334, -0.031, -0.075, 0.039]
kv = [-0.103, -0.113, -0.157, -0.296, -0.038, -0.090, -0.006]
fig, ax = plt.subplots(figsize=(8.2, 5.6)); y = list(range(len(sit)))[::-1]; h = 0.36
ax.barh([v + h / 2 for v in y], nov, height=h - 0.04, color=BLUE, label="slots, no value matrix")
ax.barh([v - h / 2 for v in y], kv, height=h - 0.04, color=ORANGE, label="slots in later layers, value matrix kept")
for v, a_, b_ in zip(y, nov, kv):
    for yy, val in ((v + h / 2, a_), (v - h / 2, b_)):
        ax.text(val + (0.006 if val > 0 else -0.006), yy, f"{val:+.2f}", va="center", ha="left" if val > 0 else "right", fontsize=9.5, color=INK2)
ax.axvline(0, color=INK2, lw=1); ax.set_yticks(y, sit, fontsize=10.5); ax.set_xlim(-0.40, 0.09)
ax.set_xlabel("loss change vs normal attention (below 0 = better)")
ax.set_title("Where the slots help", loc="left", fontsize=14, color=INK)
ax.legend(frameon=False, loc="lower left", fontsize=10)
clean(ax); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("situations.png", dpi=160); plt.close(fig)

# 4. fact-loss change vs number of facts to memorize, at two widths (2 seeds each)
nf = [16, 32, 64, 128, 256]
gain = {"width 32": ([-0.06, -0.03, -0.26, -0.66, -1.10], ORANGE), "width 64": ([0.01, -0.04, -0.22, -0.80, -0.89], BLUE)}
fig, ax = plt.subplots(figsize=(6.8, 3.8))
for name, (g, c) in gain.items():
    ax.plot(range(5), g, color=c, lw=2, marker="o", ms=8, mec=SURF, mew=2, label=name)
    ax.text(4.12, g[-1], name, va="center", fontsize=11, color=INK)
ax.axhline(0, color=INK2, lw=1); ax.set_xticks(range(5), nf); ax.set_xlim(-0.3, 4.9)
ax.set_xlabel("number of facts the model must memorize"); ax.set_ylabel("fact loss change\n(below 0 = better)")
ax.set_title("More facts to memorize, bigger gain from the slots", loc="left", fontsize=14, color=INK)
ax.legend(frameon=False, loc="lower left", fontsize=10)
clean(ax); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("facts.png", dpi=160); plt.close(fig)

# 5. which layers: left = one plain value-embedding table in chosen layers only (value matrix everywhere), 4 seeds;
#    right = split: value-embedding slots with no value matrix in chosen layers, value matrix and no table in the rest, 6 seeds. 4-layer model, width 64
place = [("table in every layer", 0.0), ("layer 2 only", -0.004), ("layers 1–3", 0.012), ("layers 2–4", 0.018),
         ("layer 4 only", 0.098), ("layer 3 only", 0.148), ("no table", 0.281), ("layer 1 only", 0.377)]
split = [("layers 1–2 slots, 3–4 matrix", -0.058), ("layers 2–3 slots, 1 & 4 matrix", -0.037), ("layer 4 slots, 1–3 matrix", 0.060),
         ("layers 3–4 slots, 1–2 matrix", 0.111), ("layers 1 & 4 slots, 2–3 matrix", 0.358), ("layer 1 slots, 2–4 matrix", 0.916)]
fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.6), gridspec_kw={"width_ratios": [1, 1.05]})
for ax, rows, title, lim in ((axs[0], place, "Which layers get a value-embedding table", (-0.06, 0.45)),
                             (axs[1], split, "Some layers slots only, others value matrix only", (-0.12, 1.05))):
    y = list(range(len(rows)))[::-1]
    ax.barh(y, [v for _, v in rows], height=0.6, color=[BLUE if v < 0.03 else "#b9b8b2" for _, v in rows])
    for yy, (_, v) in zip(y, rows):
        ax.text(max(v, 0) + 0.02 * lim[1], yy, f"{v:+.2f}", va="center", ha="left", fontsize=10, color=INK2)
    ax.axvline(0, color=INK2, lw=1); ax.set_yticks(y, [n for n, _ in rows], fontsize=10.5); ax.set_xlim(*lim)
    ax.set_title(title, loc="left", fontsize=13, color=INK); clean(ax); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
axs[0].set_xlabel("loss change vs a table in every layer"); axs[1].set_xlabel("loss change vs normal attention")
fig.tight_layout(); fig.savefig("layers.png", dpi=160); plt.close(fig)
