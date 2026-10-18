"""Random search over PointNet *training* hyperparameters with the paper architecture fixed.

Stage 1 (search): N configurations x S seeds, trained on the training tracks minus the
validation tracks, scored by validation macro F1 at the validation-tuned threshold. The data
is loaded once. Stage 2 (final): the paper's default recipe, the best searched configuration
and this repo's modified PointNet are each trained with 5 seeds through run_pointnet_pipeline,
which saves models/logs under pointnet_search/final_<name>_seed<k>.

usage: python scripts/detection/pointnet_search.py [--configs 20] [--seeds 3]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/train/detection"))
from train_detection import MultiTrackDatasetBuilder, fixed_track_split, SPLIT_PRESETS  # noqa: E402
from pointnet_model import PointCloudDataset, _balanced_group_split, train_pointnet, run_pointnet_pipeline  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--configs", type=int, default=20)
parser.add_argument("--seeds", type=int, default=3)
parser.add_argument("--final-seeds", type=int, default=5)
parser.add_argument("--dataset", default="Dataset/Processed_FrameNorm")
args = parser.parse_args()

# Paper recipe (Qi et al. 2017): Adam 1e-3, lr halved every 20 epochs, dropout keep 0.7,
# cross-entropy, 0.001 feature-transform regulariser. num_points/patience are ours.
PAPER = {"learning_rate": 1e-3, "lr_step": 20, "dropout": 0.3, "weight_decay": 0.0, "num_points": 32,
         "patience": 15, "loss": "ce"}
SPACE = {"learning_rate": [3e-4, 1e-3, 3e-3], "lr_step": [0, 20], "dropout": [0.3, 0.5],
         "weight_decay": [0.0, 1e-4, 1e-3], "num_points": [16, 32, 64], "patience": [15, 30], "loss": ["ce", "focal"]}
REG_WEIGHT = 0.001
out_dir = REPO / "logs/detection/pointnet_search"
out_dir.mkdir(parents=True, exist_ok=True)
progress = out_dir / "search_progress.txt"

rng = random.Random(0)
configs, seen = [dict(PAPER)], {tuple(sorted(PAPER.items()))}
while len(configs) < args.configs:
    cfg = {k: rng.choice(v) for k, v in SPACE.items()}
    if tuple(sorted(cfg.items())) not in seen:
        seen.add(tuple(sorted(cfg.items())))
        configs.append(cfg)

builder = MultiTrackDatasetBuilder(REPO / args.dataset, split_group_level="track")
clouds, y, groups = builder.build_raw_dataset()
train_idx, test_idx = fixed_track_split(groups, y, SPLIT_PRESETS["balanced5track"])
fit_pos, val_pos = _balanced_group_split(y[train_idx], groups[train_idx])
fit_idx, val_idx = np.asarray(train_idx)[fit_pos], np.asarray(train_idx)[val_pos]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
print(f"fit {len(fit_idx)} | val {len(val_idx)} ({int((y[val_idx] == 0).sum())} non-cones) tracks {sorted(set(groups[val_idx]))}", flush=True)


def val_scores(model, val_dataset):
    model.eval()
    X = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
    with torch.no_grad():
        prob = torch.cat([torch.softmax(model(X[i:i + 512].to(device)), 1)[:, 1].cpu() for i in range(0, len(X), 512)]).numpy()
    yv = val_dataset.labels
    macro = [f1_score(yv, (prob >= t).astype(int), average="macro") for t in thresholds]
    return f1_score(yv, (prob >= 0.5).astype(int), average="macro"), max(macro), float(thresholds[int(np.argmax(macro))])


results = []
for ci, cfg in enumerate(configs):
    runs = []
    for seed in range(args.seeds):
        t0 = time.time()
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        train_ds = PointCloudDataset([clouds[i] for i in fit_idx], y[fit_idx], cfg["num_points"], augment=True, seed=seed)
        val_ds = PointCloudDataset([clouds[i] for i in val_idx], y[val_idx], cfg["num_points"], augment=False, seed=seed)
        model, best_epoch, best_val_loss, epochs_trained = train_pointnet(
            train_ds, val_ds, num_classes=2, in_dim=4, device=device, arch="paper", dropout=cfg["dropout"],
            learning_rate=cfg["learning_rate"], lr_step=cfg["lr_step"], weight_decay=cfg["weight_decay"],
            loss=cfg["loss"], reg_weight=REG_WEIGHT, patience=cfg["patience"], verbose=False)
        macro05, macro_tuned, thr = val_scores(model, val_ds)
        runs.append({"seed": seed, "val_macro_0.5": macro05, "val_macro_tuned": macro_tuned, "threshold": thr,
                     "best_epoch": best_epoch + 1, "epochs": epochs_trained, "seconds": round(time.time() - t0, 1)})
    score = float(np.mean([r["val_macro_tuned"] for r in runs]))
    results.append({"config": cfg, "val_macro_tuned_mean": score,
                    "val_macro_tuned_std": float(np.std([r["val_macro_tuned"] for r in runs])), "runs": runs})
    line = (f"[{ci + 1:2d}/{len(configs)}] {'PAPER ' if ci == 0 else ''}{cfg} -> val macro (tuned) "
            f"{score:.4f} ± {results[-1]['val_macro_tuned_std']:.4f} | best epochs {[r['best_epoch'] for r in runs]}")
    print(line, flush=True)
    with open(progress, "a") as f:
        f.write(line + "\n")
    (out_dir / "search_results.json").write_text(json.dumps(results, indent=1))

best = max(results[1:], key=lambda r: r["val_macro_tuned_mean"])
print(f"\nPaper default: {results[0]['val_macro_tuned_mean']:.4f} | best searched: {best['val_macro_tuned_mean']:.4f} {best['config']}", flush=True)
(out_dir / "best_config.json").write_text(json.dumps({"paper_default": results[0], "best": best}, indent=1))

finals = {
    "paper_default": dict(arch="paper", reg_weight=REG_WEIGHT, **PAPER),
    "paper_tuned": dict(arch="paper", reg_weight=REG_WEIGHT, **best["config"]),
    "modified": dict(arch="modified", reg_weight=0.0, learning_rate=1e-3, lr_step=0, dropout=0.3, weight_decay=0.0,
                     num_points=32, patience=15, loss="focal"),
}
for name, cfg in finals.items():
    for seed in range(args.final_seeds):
        run_pointnet_pipeline(clouds, y, groups, REPO, train_idx=train_idx, test_idx=test_idx, random_state=seed,
                              output_name=f"pointnet_search/final_{name}_seed{seed}", use_intensity=True, **cfg)
        print(f"final {name} seed {seed} done", flush=True)
