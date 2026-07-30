#!/usr/bin/env python3
"""
Cross-track (leave-one-track-out) evaluation of the cone detector.

WHY THIS EXISTS
---------------
train.py evaluates with a random *per-cluster* split (`train_test_split` with
shuffle). Many clusters come from the same or adjacent scan frames, so a random
split scatters near-duplicate clusters across both train and test. The model
then "sees" almost-identical clusters at training time, which leaks information
and inflates the reported F1 — it does not tell us how well the detector
generalises to a track it has never observed.

A reviewer flagged exactly this. This script answers the honest question:

    Train on Skidpad + Acceleration, test on the completely held-out
    Autocross track. What is the real F1?

It reuses train.py's feature extraction and Random-Forest tuning verbatim, so
the only thing that changes versus train.py is the train/test split strategy.

USAGE
-----
    cluster_labeler_env/bin/python cross_track_eval.py            # full gridsearch (matches train.py)
    cluster_labeler_env/bin/python cross_track_eval.py --quick    # fast default RF, no gridsearch
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from tqdm import tqdm

# Reuse the *exact* feature pipeline + tuned model from train.py so this is an
# apples-to-apples comparison — only the split strategy differs.
from train import load_pcd_binary, extract_features, RandomForestConeDetector

sns.set(style='whitegrid', context='talk')

# ---------------------------------------------------------------------------
# Track grouping. Track names are relative paths under Processed/Detection,
# e.g. "Skidpad", "Acceleration", "Autocross/AutoX_7", "Autocross/AutoX_7_2".
# A track belongs to the group given by its first path component.
# ---------------------------------------------------------------------------
TRAIN_GROUPS = {'Skidpad', 'Acceleration'}
TEST_GROUPS = {'Autocross'}


def group_of(track_name):
    """Top-level track group, e.g. 'Autocross/AutoX_7' -> 'Autocross'."""
    return track_name.split('/')[0]


def discover_tracks(base_path):
    """Find every folder holding a labeled_clusters.json under base_path."""
    tracks = {}
    for labels_path in sorted(base_path.rglob('labeled_clusters.json')):
        folder = labels_path.parent
        name = str(folder.relative_to(base_path)).strip('/')
        with open(labels_path) as f:
            tracks[name] = {'path': folder, 'labels': json.load(f)}
        print(f'  ✓ {name}: {len(tracks[name]["labels"])} labels')
    if not tracks:
        raise FileNotFoundError(f'No labeled_clusters.json found under {base_path}')
    return tracks


def build_features(tracks, wanted_track_names, desc):
    """Extract features/labels for the given subset of track names."""
    X, y = [], []
    total = sum(len(tracks[n]['labels']) for n in wanted_track_names)
    pbar = tqdm(total=total, desc=desc)
    per_track = {}

    for name in wanted_track_names:
        data = tracks[name]
        n_before = len(X)
        cones = 0
        for filename, label in data['labels'].items():
            pbar.update(1)
            pcd_path = data['path'] / filename
            if not pcd_path.exists():
                continue
            cluster = load_pcd_binary(str(pcd_path))
            if cluster is None or len(cluster) < 3:
                continue
            X.append(extract_features(cluster))
            is_cone = 1 if label['is_cone'] else 0
            y.append(is_cone)
            cones += is_cone
        n = len(X) - n_before
        per_track[name] = {'samples': n, 'cones': cones, 'non_cones': n - cones}

    pbar.close()
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int64),
            per_track)


def print_split_summary(title, per_track, y):
    print(f'\n{title}')
    for name, s in per_track.items():
        print(f'  {name:<24} {s["samples"]:>5} samples '
              f'| cones {s["cones"]:>4} | non-cones {s["non_cones"]:>4}')
    n_cone = int(y.sum())
    print(f'  {"TOTAL":<24} {len(y):>5} samples '
          f'| cones {n_cone:>4} ({100*n_cone/max(len(y),1):.1f}%) '
          f'| non-cones {len(y)-n_cone:>4}')


def save_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-cone', 'Cone'],
                yticklabels=['Non-cone', 'Cone'])
    plt.title('Cross-track Confusion Matrix\n(train: Skidpad+Accel, test: Autocross)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'\n✓ Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Cross-track cone-detector evaluation')
    script_dir = Path(__file__).parent
    parser.add_argument('--dataset', default=str(script_dir / 'Dataset'),
                        help='Path to Dataset folder')
    parser.add_argument('--quick', action='store_true',
                        help='Skip gridsearch; use a sensible default RandomForest')
    args = parser.parse_args()

    base_path = Path(args.dataset).expanduser() / 'Processed' / 'Detection'

    print('🔎 Discovering tracks...')
    tracks = discover_tracks(base_path)

    train_names = [n for n in tracks if group_of(n) in TRAIN_GROUPS]
    test_names = [n for n in tracks if group_of(n) in TEST_GROUPS]
    if not train_names:
        raise RuntimeError(f'No training tracks matched {TRAIN_GROUPS}')
    if not test_names:
        raise RuntimeError(f'No testing tracks matched {TEST_GROUPS}')

    print(f'\n🚂 Train groups {sorted(TRAIN_GROUPS)} -> tracks {train_names}')
    print(f'🧪 Test  groups {sorted(TEST_GROUPS)} -> tracks {test_names}')

    X_train, y_train, train_per_track = build_features(tracks, train_names, 'Train features')
    X_test, y_test, test_per_track = build_features(tracks, test_names, 'Test  features')

    print_split_summary('📊 TRAIN SPLIT (Skidpad + Acceleration)', train_per_track, y_train)
    print_split_summary('📊 TEST SPLIT (Autocross, held out)', test_per_track, y_test)

    # Scale on TRAIN ONLY — no leakage from the held-out track.
    detector = RandomForestConeDetector()
    X_train_scaled = detector.scaler.fit_transform(X_train)
    X_test_scaled = detector.scaler.transform(X_test)

    if args.quick:
        print('\n⚡ Quick mode: default RandomForest (no gridsearch)')
        detector.model = RandomForestClassifier(
            n_estimators=150, max_depth=None, random_state=42, n_jobs=-1)
        detector.model.fit(X_train_scaled, y_train)
        detector.best_params = detector.model.get_params()
    else:
        print('\n🔧 Tuning on TRAIN ONLY (same gridsearch as train.py)...')
        detector.gridsearch(X_train_scaled, y_train)

    # Evaluate on the completely unseen Autocross track.
    y_pred = detector.model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print('\n' + '=' * 70)
    print('🎯 CROSS-TRACK RESULTS  (train: Skidpad+Accel  →  test: Autocross)')
    print('=' * 70)
    print(f'  Accuracy:  {acc:.2%}')
    print(f'  Precision: {prec:.2%}')
    print(f'  Recall:    {rec:.2%}')
    print(f'  F1 Score:  {f1:.2%}   <-- headline number')
    print('=' * 70)
    print('\nClassification report (test = Autocross):')
    print(classification_report(y_test, y_pred,
                                target_names=['Non-cone', 'Cone'],
                                digits=4,
                                zero_division=0))

    save_confusion_matrix(
        y_test, y_pred,
        script_dir / 'figures' / 'detection' / 'cross_track_confusion_matrix.png')

    # Append a machine-readable record next to the normal training logs.
    log_file = script_dir / 'logs' / 'detection' / 'cross_track_eval.txt'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'a') as f:
        json.dump({
            'split': 'cross-track',
            'train_groups': sorted(TRAIN_GROUPS),
            'test_groups': sorted(TEST_GROUPS),
            'train_tracks': train_names,
            'test_tracks': test_names,
            'train_samples': int(len(y_train)),
            'test_samples': int(len(y_test)),
            'gridsearch': not args.quick,
            'best_params': detector.best_params,
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
        }, f)
        f.write('\n\n')
    print(f'✓ Logged: {log_file}')


if __name__ == '__main__':
    main()
