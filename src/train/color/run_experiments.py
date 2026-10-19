#!/usr/bin/env python3
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

MODELS = ["rf", "xgb", "pointnet", "pointnet2"]

# Raw intensity value that maps to normalized 100 (used by the fixed_* stripe thresholds)
INTENSITY_MAX = 255.0

# Optional filter: python run_all_experiments.py fixed_  -> only run IDs containing "fixed_"
ONLY = sys.argv[1] if len(sys.argv) > 1 else None

COLOR_DS = "Dataset/Processed/Color"
RECLASS_DS = "Dataset/Processed/Color/Ground Filter Reclassification"

# The first three configs use the whole Color folder (reclassification tracks included,
# because they are nested inside it). The *_noreclass configs exclude those tracks.
CONFIGURATIONS = [
    # Percentile/GMM stripe removal (per-cluster, adaptive)
    {"name": "pct_no_stripes",   "dataset": COLOR_DS, "stripe_mode": "no_stripes",   "exclude_reclass": False},
    {"name": "pct_only_stripes", "dataset": COLOR_DS, "stripe_mode": "only_stripes", "exclude_reclass": False},
    # Fixed-threshold stripe removal (white >= 70, black <= 35 normalized)
    {"name": "fixed_no_stripes",   "dataset": COLOR_DS, "stripe_mode": "fixed_no_stripes",   "exclude_reclass": False},
    {"name": "fixed_only_stripes", "dataset": COLOR_DS, "stripe_mode": "fixed_only_stripes", "exclude_reclass": False},
    # Full dataset and reclassification-only
    {"name": "processed_color",  "dataset": COLOR_DS,   "stripe_mode": None, "exclude_reclass": False},
    {"name": "reclassification", "dataset": RECLASS_DS, "stripe_mode": None, "exclude_reclass": False},
    # Same experiments with the reclassification tracks excluded
    {"name": "pct_no_stripes_noreclass",   "dataset": COLOR_DS, "stripe_mode": "no_stripes",   "exclude_reclass": True},
    {"name": "pct_only_stripes_noreclass", "dataset": COLOR_DS, "stripe_mode": "only_stripes", "exclude_reclass": True},
    {"name": "fixed_no_stripes_noreclass",   "dataset": COLOR_DS, "stripe_mode": "fixed_no_stripes",   "exclude_reclass": True},
    {"name": "fixed_only_stripes_noreclass", "dataset": COLOR_DS, "stripe_mode": "fixed_only_stripes", "exclude_reclass": True},
    {"name": "processed_color_noreclass", "dataset": COLOR_DS, "stripe_mode": None, "exclude_reclass": True},
    # Reclassification dataset with stripe filters applied
    {"name": "reclass_pct_no_stripes",     "dataset": RECLASS_DS, "stripe_mode": "no_stripes",       "exclude_reclass": False},
    {"name": "reclass_pct_only_stripes",   "dataset": RECLASS_DS, "stripe_mode": "only_stripes",     "exclude_reclass": False},
    {"name": "reclass_fixed_no_stripes",   "dataset": RECLASS_DS, "stripe_mode": "fixed_no_stripes", "exclude_reclass": False},
    {"name": "reclass_fixed_only_stripes", "dataset": RECLASS_DS, "stripe_mode": "fixed_only_stripes","exclude_reclass": False},
]

CLASSES_MODES = [
    {"name": "multiclass", "flag": "--multiclass"},
    {"name": "binary", "flag": None},
]

total_runs = len(MODELS) * len(CONFIGURATIONS) * len(CLASSES_MODES)
print(f"🚀 Launching experiment suite: {total_runs} runs "
      f"({len(CONFIGURATIONS)} configs x {len(MODELS)} models x {len(CLASSES_MODES)} class modes)\n")

failed, skipped = [], []
current_run = 1
suite_start = time.perf_counter()

for class_mode in CLASSES_MODES:
    for config in CONFIGURATIONS:
        for model in MODELS:
            run_id = f"{model}_{class_mode['name']}_{config['name']}"
            ok_marker = LOG_DIR / f"{run_id}.ok"
            if ONLY and ONLY not in run_id:
                current_run += 1
                continue
            tag = f"[{current_run}/{total_runs}] {run_id}"
            current_run += 1

            # Resume support: skip runs that already finished successfully
            if ok_marker.exists():
                print(f"⏭️  {tag} already done, skipping")
                skipped.append(run_id)
                continue

            print(f"▶️  {tag}")
            cmd = [
                sys.executable, str(REPO_ROOT / "train_color.py"),
                "--dataset", config["dataset"],
                "--model", model,
                "--run-id", run_id,
                "--intensity-max", str(INTENSITY_MAX),
            ]
            if class_mode["flag"]:
                cmd.append(class_mode["flag"])
            if config["stripe_mode"]:
                cmd.extend(["--stripe-mode", config["stripe_mode"]])
            if config["exclude_reclass"]:
                cmd.append("--exclude-reclass")

            log_path = LOG_DIR / f"{run_id}.log"
            with open(log_path, "w") as log:
                log.write("CMD: " + " ".join(cmd) + "\n\n")
                log.flush()
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)

            if result.returncode == 0:
                ok_marker.touch()
                print(f"✓ Completed {run_id} (log: {log_path.name})\n")
            else:
                failed.append(run_id)
                print(f"❌ Failed {run_id} (exit {result.returncode}), see {log_path}\n")

elapsed = time.perf_counter() - suite_start
print(f"🏁 Suite finished in {elapsed/60:.1f} min | "
      f"{total_runs - len(failed) - len(skipped)} succeeded, {len(skipped)} skipped, {len(failed)} failed")
if failed:
    print("Failed runs:")
    for r in failed:
        print(f"  - {r}")
    sys.exit(1)