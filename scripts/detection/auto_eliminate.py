"""Automatic backward feature elimination around src/train/detection/train_detection.py.

Each stage runs the full GridSearch, plots the importances, then drops the (up to) two
lowest-importance features below --threshold. Stops when no feature is below it.

usage: python scripts/detection/auto_eliminate.py --model rf [--start f1,f2,...]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXTENDED = ["height", "width", "depth", "aspect_ratio", "num_points", "density", "compactness",
            "intensity_mean", "intensity_std", "linearity", "planarity", "volume", "distancefromlidar"]

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["rf", "xgb"], required=True)
parser.add_argument("--start", help="comma-separated starting feature set (default: all 13)")
parser.add_argument("--dataset", default="Dataset/Processed_FrameNorm")
parser.add_argument("--test-tracks", default="balanced5track")
parser.add_argument("--output-root", help="default: <model>_adaptive_balanced5track")
parser.add_argument("--threshold", type=float, default=0.015)
parser.add_argument("--hard-limit", type=float, default=0.03, help="never drop a feature at or above this")
parser.add_argument("--min-features", type=int, default=5)
parser.add_argument("--log-dir", default=str(REPO / "logs/detection/auto_eliminate"))
args = parser.parse_args()

root = args.output_root or f"{args.model}_adaptive_balanced5track"
feats = args.start.split(",") if args.start else list(EXTENDED)
log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
summary = log_dir / f"{root}_summary.txt"
env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore")

while True:
    n, t0 = len(feats), time.time()
    stdout_path = log_dir / f"{root}_{n}_features.out"
    cmd = [sys.executable, "src/train/detection/train_detection.py", "--dataset", args.dataset,
           "--model", args.model, "--features", ",".join(feats), "--test-tracks", args.test_tracks,
           "--output-name", root]
    with open(stdout_path, "w") as f:
        if subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode:
            print(f"{args.model} {n}: FAILED, see {stdout_path}", flush=True)
            sys.exit(1)

    sub = f"{root}/{n}_features"
    log = json.loads([l for l in open(REPO / "logs/detection" / sub / "training_log.txt") if l.strip()][-1])
    label = f"{args.model.upper()} {n}"
    subprocess.run([sys.executable, "scripts/detection/plot_importance.py",
                    f"figures/detection/{sub}/importance_summary.png",
                    f"Feature importance, {label} ({root})", f"{label}={sub}"],
                   cwd=REPO, env=env, stdout=subprocess.DEVNULL)

    imp = log["feature_importances"]
    ranked = sorted(imp, key=imp.get)
    drop = [f for f in ranked if imp[f] < args.threshold][:2]
    drop = drop[:max(0, len(feats) - args.min_features)]
    assert all(imp[f] < args.hard_limit for f in drop)
    line = (f"{label:7s} F1 {log['f1_score']:.4f} macro {log['f1_macro']:.4f} "
            f"ncR {log['noncone_recall']:.4f} ncP {log['noncone_precision']:.4f} cm {log['confusion_matrix']} "
            f"gs {log['gridsearch_cv_f1_macro']:.4f} {time.time() - t0:.0f}s | "
            f"lowest {[(f, round(imp[f] * 100, 2)) for f in ranked[:4]]} | drop {drop}")
    print(line, flush=True)
    with open(summary, "a") as f:
        f.write(line + "\n")
    if not drop:
        break
    feats = [f for f in feats if f not in drop]
