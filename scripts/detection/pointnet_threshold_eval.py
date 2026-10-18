"""Validation-chosen decision threshold + test metrics for saved detection PointNets.

usage: python scripts/detection/pointnet_threshold_eval.py OUT.png "Title" LABEL=output_name [LABEL=output_name ...]
Runs sharing a LABEL (e.g. several seeds) are aggregated. The threshold maximising macro F1
on each run's own validation tracks is applied to the test tracks; the test set is never
used for choosing it. Per-run results go to logs/detection/<output_name>/threshold_tuning.json.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/train/detection"))
from train_detection import MultiTrackDatasetBuilder, fixed_track_split, SPLIT_PRESETS  # noqa: E402
from pointnet_model import PointNetClassifier, PointCloudDataset  # noqa: E402

out, title, specs = Path(sys.argv[1]), sys.argv[2], [a.split("=", 1) for a in sys.argv[3:]]
device = "cuda" if torch.cuda.is_available() else "cpu"
cache = {}


def load_data(dataset):
    if dataset not in cache:
        builder = MultiTrackDatasetBuilder(REPO / dataset, split_group_level="track")
        cache[dataset] = builder.build_raw_dataset()
    return cache[dataset]


def test_metrics(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {"threshold": float(threshold), "f1": float(f1_score(y_true, pred)),
            "f1_macro": float(f1_score(y_true, pred, average="macro")),
            "noncone_recall": float(recall_score(y_true, pred, pos_label=0)),
            "noncone_precision": float(precision_score(y_true, pred, pos_label=0, zero_division=0)),
            "false_cones": int(cm[0, 1]), "missed_cones": int(cm[1, 0])}


thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
by_label = defaultdict(list)
for label, name in specs:
    log = json.loads([l for l in open(REPO / "logs/detection" / name / "training_log.txt") if l.strip()][-1])
    clouds, y, groups = load_data("Dataset/Processed_FrameNorm")
    _, test_idx = fixed_track_split(groups, y, SPLIT_PRESETS["balanced5track"])
    val_idx = np.where(np.isin(groups, log["val_tracks"]))[0]
    ckpt = torch.load(REPO / "models/detection" / name / "pointnet_detection.pth", map_location=device)
    use_intensity = ckpt.get("use_intensity", True)
    model = PointNetClassifier(num_classes=ckpt["num_classes"], in_dim=ckpt.get("in_dim", 4), arch=ckpt.get("arch", "modified")).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])

    def probabilities(idx):
        ds = PointCloudDataset([clouds[i] for i in idx], y[idx], ckpt["num_points"], augment=False, use_intensity=use_intensity)
        X = torch.stack([ds[i][0] for i in range(len(ds))])
        with torch.no_grad():
            return torch.cat([torch.softmax(model(X[i:i + 512].to(device)), 1)[:, 1].cpu()
                              for i in range(0, len(X), 512)]).numpy()

    p_val, p_test = probabilities(val_idx), probabilities(test_idx)
    val_macro = np.array([f1_score(y[val_idx], (p_val >= t).astype(int), average="macro") for t in thresholds])
    best = thresholds[np.flatnonzero(val_macro == val_macro.max())]
    chosen = float(best[np.argmin(np.abs(best - 0.5))])
    result = {"val_tracks": log["val_tracks"], "chosen_threshold": chosen,
              "test_at_0.5": test_metrics(y[test_idx], p_test, 0.5),
              "test_at_chosen": test_metrics(y[test_idx], p_test, chosen)}
    (REPO / "logs/detection" / name / "threshold_tuning.json").write_text(json.dumps(result, indent=1))
    by_label[label].append(result)
    r0, rc = result["test_at_0.5"], result["test_at_chosen"]
    print(f"{name:32s} @0.50 F1 {r0['f1']:.4f} macro {r0['f1_macro']:.4f} | @{chosen:.2f} F1 {rc['f1']:.4f} "
          f"macro {rc['f1_macro']:.4f} ncR {rc['noncone_recall']:.3f} FP {rc['false_cones']} FN {rc['missed_cones']}")

print()
for label, runs in by_label.items():
    for key in ("test_at_0.5", "test_at_chosen"):
        f1 = np.array([r[key]["f1"] for r in runs]); macro = np.array([r[key]["f1_macro"] for r in runs])
        fp = np.array([r[key]["false_cones"] for r in runs]); fn = np.array([r[key]["missed_cones"] for r in runs])
        print(f"{label:12s} {key:15s} F1 {f1.mean():.4f} ± {f1.std():.4f} | macro {macro.mean():.4f} ± {macro.std():.4f} "
              f"| FP {fp.mean():.1f} FN {fn.mean():.1f} (n={len(runs)})")

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
labels = list(by_label)
fig, axes = plt.subplots(1, 2, figsize=(12, 0.9 * len(labels) + 2.6), facecolor=SURF)
for ax, (metric, name) in zip(axes, [("f1", "Sima F1"), ("f1_macro", "Macro F1")]):
    ax.set_facecolor(SURF)
    for row, label in enumerate(labels):
        vals = np.array([r["test_at_chosen"][metric] for r in by_label[label]])
        ax.scatter(vals, np.full(len(vals), row), s=64, color=COLORS[row % len(COLORS)], edgecolor=SURF, linewidth=2, zorder=3)
        ax.plot([vals.mean()] * 2, [row - 0.25, row + 0.25], color=TEXT, lw=2, zorder=4)
        ax.text(vals.max(), row + 0.3, f"átlag {vals.mean():.3f} ± {vals.std():.3f}", fontsize=9, color=MUTED, ha="right")
    ax.set_yticks(range(len(labels)), labels, color=TEXT)
    ax.set_ylim(-0.6, len(labels) - 0.2)
    ax.set_title(f"{name} (teszt, validáción hangolt küszöb)", loc="left", fontsize=12, color=TEXT)
    ax.grid(axis="x", color=GRID, lw=1, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=TEXT)
fig.text(0.01, 0.01, "Pontok: egy-egy seed. Fekete vonás: átlag. Frame-normált adat, balanced5track split.", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.06, 1, 0.92))
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, facecolor=SURF)
print(out)
