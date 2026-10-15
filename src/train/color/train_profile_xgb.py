#!/usr/bin/env python3
"""Two-stage colour classifiers on ring-profile features with XGBoost.

  profile_xgb       stage 1: yellow vs non-yellow, 20 ring-profile features
  yref_profile_xgb  stage 2: orange vs blue (non-yellow clusters only), the same 20 features plus
                    4 yellow-referenced intensity features (cluster level minus the scan's yellow cones)

Evaluation: leave-one-recording-out (every recording held out once, predictions pooled) is the
headline number; grouped 5-fold x 3 CV (groups = recording + 50-frame block) is logged alongside.
The final model is trained on all clusters and saved to models/color/<name>/ (pickle + XGBoost JSON);
one JSON line per run is appended to logs/color/<name>/training_log.txt.

Usage:
  cluster_labeler_env/bin/python src/train/color/train_profile_xgb.py --model all
"""
import argparse
import json
import pickle
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from blue_orange_models import FeatureApproach
from cone_dataset import load_cone_dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILE_GROUPS = ['profile', 'ring', 'bimodal', 'range']
MODELS = {
    'profile_xgb': dict(colours=('blue', 'orange', 'yellow'), positive='yellow', negative='non-yellow',
                        groups=PROFILE_GROUPS),
    'yref_profile_xgb': dict(colours=('blue', 'orange'), positive='orange', negative='blue',
                             groups=PROFILE_GROUPS + ['yref'], recommended_max_distance_m=4.0),
}
DISTANCE_BANDS = [(0, 4, '0-4m'), (4, 7, '4-7m'), (7, 10, '7-10m'), (10, np.inf, '10m+')]
THRESHOLD = 0.5


def metrics(y, p) -> dict:
    y, pred = np.asarray(y).astype(int), (np.asarray(p) >= THRESHOLD).astype(int)
    both = len(np.unique(y)) == 2
    prec, rec, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {
        'n': int(len(y)), 'n_negative': int(support[0]), 'n_positive': int(support[1]),
        'macro_f1': float(f1_score(y, pred, labels=[0, 1], average='macro', zero_division=0)) if both else None,
        'roc_auc': float(roc_auc_score(y, p)) if both else None,
        'precision_positive': float(prec[1]), 'recall_positive': float(rec[1]), 'f1_positive': float(f1[1]),
        'precision_negative': float(prec[0]), 'recall_negative': float(rec[0]), 'f1_negative': float(f1[0]),
        'confusion_rows_true_negative_positive': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def fit_predict(groups, seed, reps, y, train, test):
    model = FeatureApproach(groups, 'xgb', seed=seed)
    model.fit([reps[i] for i in train], y[train])
    return model.predict_proba([reps[i] for i in test])


def git_state() -> dict:
    def run(*args):
        return subprocess.run(['git', *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return {'commit': run('rev-parse', 'HEAD'), 'dirty': bool(run('status', '--porcelain', '--untracked-files=no'))}


def train(name: str, spec: dict, dataset_path: Path, seed: int):
    t0 = time.time()
    print(f'\n▶ {name}: {spec["positive"]} vs {spec["negative"]}')
    ds = load_cone_dataset(dataset_path, colors=spec['colours'])
    meta = ds.meta
    y = (meta['color'] == spec['positive']).to_numpy().astype(int)
    rec, dist = meta['recording'].to_numpy(), meta['distance'].to_numpy()
    blocks = (meta['recording'] + '#' + (meta['frame'] // 50).astype(str)).to_numpy()
    idx = np.arange(len(y))

    probe = FeatureApproach(spec['groups'], 'xgb', seed=seed)
    reps = [probe.featurize(p, ds.context(i)) for i, p in enumerate(ds.points)]
    print(f'  {len(y)} clusters ({y.sum()} {spec["positive"]}), {len(set(rec))} recordings, '
          f'{len(probe.feature_names)} features')

    # Leave-one-recording-out, pooled
    p_loro = np.empty(len(y))
    for held_out in sorted(set(rec)):
        test = idx[rec == held_out]
        p_loro[test] = fit_predict(spec['groups'], seed, reps, y, idx[rec != held_out], test)
    loro = metrics(y, p_loro)
    loro['by_distance'] = {band: metrics(y[(dist >= lo) & (dist < hi)], p_loro[(dist >= lo) & (dist < hi)])
                           for lo, hi, band in DISTANCE_BANDS}
    loro['per_recording'] = {r: metrics(y[rec == r], p_loro[rec == r]) for r in sorted(set(rec))}
    print(f'  leave-one-recording-out macro-F1 {loro["macro_f1"]:.3f}  AUC {loro["roc_auc"]:.3f}  '
          f'(0-4 m: {loro["by_distance"]["0-4m"]["macro_f1"]:.3f})')

    # Grouped 5-fold x 3 CV
    cv_f1, cv_auc = [], []
    for r in range(3):
        p_cv = np.empty(len(y))
        for train_idx, test_idx in StratifiedGroupKFold(5, shuffle=True, random_state=seed + r).split(idx, y, blocks):
            p_cv[test_idx] = fit_predict(spec['groups'], seed + r, reps, y, train_idx, test_idx)
        m = metrics(y, p_cv)
        cv_f1.append(m['macro_f1'])
        cv_auc.append(m['roc_auc'])
    print(f'  grouped CV macro-F1 {np.mean(cv_f1):.3f} ± {np.std(cv_f1):.3f}')

    # Final model on all data
    final = FeatureApproach(spec['groups'], 'xgb', seed=seed)
    final.fit(reps, y)
    final.prepare_for_latency()
    times = []
    for i in np.random.default_rng(seed).choice(idx, size=min(200, len(idx)), replace=False):
        start = time.perf_counter()
        final.predict_proba([final.featurize(ds.points[i], ds.context(i))])
        times.append((time.perf_counter() - start) * 1e3)

    model_dir = REPO_ROOT / 'models' / 'color' / name
    model_dir.mkdir(parents=True, exist_ok=True)
    pkl_path, json_path = model_dir / f'color_classifier_{name}.pkl', model_dir / f'color_classifier_{name}.json'
    with open(pkl_path, 'wb') as f:
        pickle.dump({'model': final.estimator_, 'feature_names': final.feature_names,
                     'feature_groups': spec['groups'], 'height_mode': final.height_mode,
                     'positive_class': spec['positive'], 'negative_class': spec['negative'],
                     'threshold': THRESHOLD, 'params': final.estimator_.get_params()}, f)
    final.estimator_.get_booster().save_model(str(json_path))

    log = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'model_type': name, 'classifier': 'xgboost', 'binary_mode': True,
        'positive_class': spec['positive'], 'negative_class': spec['negative'], 'threshold': THRESHOLD,
        'git': git_state(), 'seed': seed,
        'dataset_size': int(len(y)), 'train_size': int(len(y)),
        'class_counts': {spec['negative']: int((y == 0).sum()), spec['positive']: int(y.sum())},
        'recordings': {r: {spec['negative']: int(((rec == r) & (y == 0)).sum()),
                           spec['positive']: int(((rec == r) & (y == 1)).sum())} for r in sorted(set(rec))},
        'feature_names': final.feature_names,
        'params': {k: v for k, v in final.estimator_.get_params().items()
                   if k in ('n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree',
                            'tree_method', 'scale_pos_weight')},
        'f1_score': loro['macro_f1'],
        'leave_one_recording_out': loro,
        'grouped_cv_5x3': {'macro_f1_mean': float(np.mean(cv_f1)), 'macro_f1_std': float(np.std(cv_f1)),
                           'roc_auc_mean': float(np.mean(cv_auc)),
                           'groups': 'recording + 50-frame block'},
        'latency_ms': {'median': float(np.median(times)), 'p95': float(np.percentile(times, 95)),
                       'note': 'single cluster, feature extraction + predict, one thread'},
        'model_files': [str(pkl_path.relative_to(REPO_ROOT)), str(json_path.relative_to(REPO_ROOT))],
    }
    if name == 'yref_profile_xgb':
        log['yellow_reference'] = ('median log intensity of the labelled yellow cones in the same scan; '
                                   'at inference it must come from profile_xgb predictions')
        log['recommended_max_distance_m'] = spec['recommended_max_distance_m']

    log_dir = REPO_ROOT / 'logs' / 'color' / name
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / 'training_log.txt', 'a') as f:
        f.write(json.dumps(log) + '\n')
    print(f'  saved {pkl_path.relative_to(REPO_ROOT)}, {json_path.relative_to(REPO_ROOT)}; '
          f'log logs/color/{name}/training_log.txt  [{time.time() - t0:.0f}s]')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=[*MODELS, 'all'], default='all')
    parser.add_argument('--dataset', type=Path, default=REPO_ROOT / 'Dataset')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    for name in (MODELS if args.model == 'all' else [args.model]):
        train(name, MODELS[name], args.dataset, args.seed)


if __name__ == '__main__':
    main()
