"""Side-by-side feature importance panels from saved detection models.

usage: python scripts/detection/plot_importance.py OUT.png "Title" LABEL=logs_subdir [LABEL=logs_subdir ...]
logs_subdir is relative to logs/detection (and models/detection), e.g. rf_adaptive_balanced5track/9_features.
"""
import json
import pickle
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out, title, specs = Path(sys.argv[1]), sys.argv[2], [a.split("=", 1) for a in sys.argv[3:]]
BAR, INT, TEXT, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
box = dict(facecolor=SURF, edgecolor="none", pad=1.5)

panels = []
for label, sub in specs:
    log = json.loads([l for l in open(f"logs/detection/{sub}/training_log.txt") if l.strip()][-1])
    model = pickle.load(open(next(Path("models/detection", sub).glob("*.pkl")), "rb"))["model"]
    kind = "Gini-csökkenés" if "Forest" in type(model).__name__ else "átlagos gain"
    panels.append((label, dict(zip(log["feature_names"], model.feature_importances_)), log, kind))

order = sorted(panels[0][1], key=panels[0][1].get)
for _, imp, _, _ in panels[1:]:
    order = [f for f in sorted(imp, key=imp.get) if f not in order] + order
xmax = max(max(imp.values()) for _, imp, _, _ in panels) * 1.25

fig_width = max(9.0, 5.2 * len(panels) + 1.5)
fig, axes = plt.subplots(1, len(panels), figsize=(fig_width, 0.42 * len(order) + 2.6),
                         sharey=True, facecolor=SURF, squeeze=False)
for ax, (label, imp, log, kind) in zip(axes[0], panels):
    ax.set_facecolor(SURF)
    for i, f in enumerate(order):
        if f not in imp:
            ax.text(xmax * 0.012, i, "nincs a készletben", va="center", fontsize=9, color=MUTED, style="italic")
            continue
        v = imp[f]
        ax.barh(i, v, height=0.62, color=INT if f.startswith("intensity") else BAR, zorder=2)
        ax.text(v + xmax * 0.012, i, f"{v*100:.1f}%", va="center", fontsize=9, color=MUTED, zorder=3, bbox=box)
    intensity_total = sum(v for f, v in imp.items() if f.startswith("intensity"))
    ax.set_title(label, fontsize=13, color=TEXT, loc="left", pad=36)
    ax.text(0, 1.06, f"{kind} · F1 {log['f1_score']:.4f} · macro F1 {log['f1_macro']:.3f}", transform=ax.transAxes, fontsize=9, color=MUTED)
    if intensity_total:
        ax.text(0, 1.015, f"intensity összesen: {intensity_total*100:.1f}%", transform=ax.transAxes, fontsize=9, color=MUTED)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.grid(axis="x", color=GRID, lw=1, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
axes[0][0].set_yticks(range(len(order)), order, color=TEXT)
fig.suptitle(textwrap.fill(title, width=int(fig_width * 7)), x=0.01, ha="left", fontsize=14, color=TEXT)
fig.text(0.01, 0.005, textwrap.fill("Minden modellben 100%-ra normált. Narancs: intensity alapú feature. "
         "Az RF (Gini) és az XGB (gain) értékei nem összemérhetők, csak a sorrend.", width=int(fig_width * 13)), fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.06, 1, 0.90))
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, facecolor=SURF)
print(out)
