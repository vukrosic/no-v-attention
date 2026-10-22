"""draws the three README charts from the numbers in RESULTS.md"""
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

# 3. gain vs model width (mean loss change, context-picked slots without value matrix minus normal attention)
w = ["64\n(4 seeds)", "128\n(4 seeds)", "256\n(2 seeds)"]; g = [-0.088, -0.021, 0.036]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.bar(range(3), g, width=0.55, color=[BLUE, BLUE, "#b9b8b2"])
for i, v in enumerate(g): ax.text(i, v + (0.006 if v > 0 else -0.012), f"{v:+.3f}", ha="center", fontsize=11, color=INK)
ax.axhline(0, color=INK2, lw=1); ax.set_xticks(range(3), w); ax.set_ylim(-0.11, 0.06)
ax.set_xlabel("model width"); ax.set_ylabel("loss change (below 0 = better)")
ax.set_title("The gain shrinks as the model gets wider", loc="left", fontsize=14, color=INK)
clean(ax); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("width.png", dpi=160); plt.close(fig)
