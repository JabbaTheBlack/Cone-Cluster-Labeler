#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')


import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import struct
import pickle
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm




sns.set(style='whitegrid', context='talk')



# ============================================================================
# UTILS
# ============================================================================



class Timer:
    def __init__(self, name):
        self.name = name
        self.start = None



    def __enter__(self):
        self.start = time.perf_counter()
        print(f'⏱️  {self.name} started...')
        return self



    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        print(f'⏱️  {self.name} finished in {elapsed:.2f} s')




def format_seconds(seconds: float) -> str:
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours >= 1:
        return f'{int(hours)}h {int(minutes)}m {sec:.1f}s'
    if minutes >= 1:
        return f'{int(minutes)}m {sec:.1f}s'
    return f'{sec:.2f}s'




# ============================================================================
# FEATURE EXTRACTION
# ============================================================================



def load_pcd_binary(filepath):
    """Load binary PCD file."""
    with open(filepath, 'rb') as f:
        while True:
            line = f.readline().decode('utf-8').strip()
            if line.startswith('DATA'):
                break
        points = []
        while True:
            data = f.read(16)
            if not data:
                break
            try:
                x, y, z, intensity = struct.unpack('ffff', data)
                points.append([x, y, z, intensity])
            except Exception:
                break
    return np.array(points, dtype=np.float32) if points else None




def extract_features(cluster_points):
    xyz = cluster_points[:, :3]
    intensity_raw = cluster_points[:, 3]



    # Range compensation: raw intensity falls off with distance, so scale it back up
    distance_sq = xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2
    intensity = intensity_raw * distance_sq



    sigma_x = float(xyz[:, 0].std())
    sigma_y = float(xyz[:, 1].std())



    height = float(xyz[:, 2].max() - xyz[:, 2].min())
    width = max(sigma_x, sigma_y)
    depth = min(sigma_x, sigma_y)
    aspect_ratio = height / (width + 1e-6)



    num_points = len(xyz)
    volume = (height + 1e-6) * (width + 1e-6) * (depth + 1e-6)
    density = num_points / volume



    center = xyz.mean(axis=0)
    distancefromlidar = np.linalg.norm(center)



    intensity_std = float(intensity.std())
    intensity_mean = float(intensity.mean())



    return np.array([
        height, width, depth, aspect_ratio,
        density, intensity_std, intensity_mean, volume, distancefromlidar
    ], dtype=np.float32)




# ============================================================================
# DATASET BUILDER
# ============================================================================



class MultiTrackDatasetBuilder:
    """Combines labeled clusters from multiple track folders into one dataset."""



    def __init__(self, base_dataset_path, split_group_level='parent'):
        """
        Args:
            base_dataset_path: Path to *the folder you want to search*.
                               We recurse from here for labeled_clusters.json.
        """
        self.base_path = Path(base_dataset_path).expanduser()
        self.split_group_level = split_group_level



        self.tracks = {}
        self.X = []
        self.y = []
        self.groups = []
        self.track_stats = {}



        self._discover_tracks()



    def _discover_tracks(self):
        """Find all track folders with labeled_clusters.json files recursively."""
        self.tracks = {}



        for labels_path in self.base_path.rglob('labeled_clusters.json'):
            track_folder = labels_path.parent
            track_name = str(track_folder.relative_to(self.base_path)).strip('/')



            with open(labels_path) as f:
                labels = json.load(f)



            self.tracks[track_name] = {
                'path': track_folder,
                'labels': labels,
                'label_count': len(labels)
            }
            print(f'✓ Found {track_name}: {len(labels)} labels')



        if not self.tracks:
            raise FileNotFoundError(f'No labeled_clusters.json found in {self.base_path}')



    def _make_group_id(self, track_name: str, filename: str) -> str:
        rel = Path(filename)



        if self.split_group_level == 'track':
            return track_name



        if self.split_group_level == 'parent':
            parent_rel = rel.parent
            if str(parent_rel) == '.':
                return f'{track_name}::__root__'
            return f'{track_name}::{parent_rel.as_posix()}'



        raise ValueError(f'Unknown split_group_level: {self.split_group_level}')



    def build(self):
        """Extract features from all labeled clusters across all tracks."""
        print(f'\n📊 Building dataset from {len(self.tracks)} tracks...\n')



        total_labels = sum(t['label_count'] for t in self.tracks.values())
        pbar = tqdm(total=total_labels, desc='Processing clusters')



        for track_name, track_data in self.tracks.items():
            track_path = track_data['path']
            labels = track_data['labels']



            X_track, y_track = [], []
            missing = 0



            for filename, label in labels.items():
                pcd_path = track_path / filename



                if not pcd_path.exists():
                    missing += 1
                    pbar.update(1)
                    continue



                cluster = load_pcd_binary(str(pcd_path))
                if cluster is None or len(cluster) < 3:
                    pbar.update(1)
                    continue



                features = extract_features(cluster)
                target = 1 if label['is_cone'] else 0
                group_id = self._make_group_id(track_name, filename)



                X_track.append(features)
                y_track.append(target)



                self.X.append(features)
                self.y.append(target)
                self.groups.append(group_id)



                pbar.update(1)



            X_track = np.array(X_track, dtype=np.float32)
            y_track = np.array(y_track, dtype=np.int64)



            self.track_stats[track_name] = {
                'samples': len(X_track),
                'cones': int(np.sum(y_track)),
                'non_cones': int(len(y_track) - np.sum(y_track)),
                'missing': missing
            }



        pbar.close()



        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)
        self.groups = np.array(self.groups)



        self._print_stats()



        return self.X, self.y, self.groups



    def _print_stats(self):
        print('\n' + '=' * 70)
        print('📈 DATASET STATISTICS')
        print('=' * 70)



        for track_name, stats in self.track_stats.items():
            print(f'\n{track_name}:')
            print(f'  Samples: {stats["samples"]}')
            print(f'  Cones: {stats["cones"]} ({100 * stats["cones"] / (stats["samples"] + 1e-6):.1f}%)')
            print(f'  Non-cones: {stats["non_cones"]} ({100 * stats["non_cones"] / (stats["samples"] + 1e-6):.1f}%)')
            if stats["missing"] > 0:
                print(f'  ⚠️  Missing PCDs: {stats["missing"]}')



        unique_groups = len(np.unique(self.groups))
        print(f'\n{"-" * 70}')
        print(f'✓ COMBINED DATASET: {len(self.X)} total samples')
        print(f'  Cones: {sum(self.y)} ({100 * sum(self.y) / len(self.y):.1f}%)')
        print(f'  Non-cones: {len(self.y) - sum(self.y)} ({100 * (len(self.y) - sum(self.y)) / len(self.y):.1f}%)')
        print(f'  Unique split groups: {unique_groups}')
        print('=' * 70 + '\n')




# ============================================================================
# GROUPED SPLIT
# ============================================================================



def grouped_train_test_split_best_effort(
    X, y, groups,
    test_size=0.2,
    n_trials=200,
    random_state=42
):
    """
    Group-aware split:
    - no group appears in both train and test
    - searches all test-group combinations
    - keeps the split closest to requested sample-level test_size
    - uses class balance as a tie-breaker
    """

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError('Need at least 2 groups for grouped train/test split.')

    total_samples = len(X)
    group_to_indices = {g: np.where(groups == g)[0] for g in unique_groups}

    best = None
    best_score = None

    for r in range(1, len(unique_groups)):
        for test_group_combo in combinations(unique_groups, r):
            test_group_set = set(test_group_combo)
            train_group_set = set(unique_groups) - test_group_set

            if not train_group_set or not test_group_set:
                continue

            test_idx = np.concatenate([group_to_indices[g] for g in test_group_combo])
            train_idx = np.concatenate([group_to_indices[g] for g in train_group_set])

            actual_test_ratio = len(test_idx) / total_samples
            ratio_error = abs(actual_test_ratio - test_size)

            train_pos = y[train_idx].mean() if len(train_idx) else 0.0
            test_pos = y[test_idx].mean() if len(test_idx) else 0.0
            class_balance_error = abs(train_pos - test_pos)

            score = (ratio_error, class_balance_error)

            if best is None or score < best_score:
                best = (train_idx, test_idx, actual_test_ratio, class_balance_error, train_group_set, test_group_set)
                best_score = score

    if best is None:
        raise RuntimeError('Failed to create a valid grouped split.')

    train_idx, test_idx, actual_test_ratio, class_balance_error, train_groups, test_groups = best

    print('\n[Grouped Train/Test Split]')
    print(f'  Requested test ratio: {test_size:.2%}')
    print(f'  Actual test ratio:    {actual_test_ratio:.2%}')
    print(f'  Ratio error:          {abs(actual_test_ratio - test_size):.2%}')
    print(f'  Train samples:        {len(train_idx)}')
    print(f'  Test samples:         {len(test_idx)}')
    print(f'  Train groups:         {len(train_groups)}')
    print(f'  Test groups:          {len(test_groups)}')
    print(f'  Class balance delta:  {class_balance_error:.4f}')

    return train_idx, test_idx


# ============================================================================
# RANDOM FOREST
# ============================================================================



class RandomForestConeDetector:
    def __init__(self, split_test_size=0.2, split_trials=200, random_state=42):
        self.scaler = StandardScaler()
        self.model = None
        self.best_params = None
        self.random_state = random_state
        self.split_test_size = split_test_size
        self.split_trials = split_trials
        self.feature_names = [
            'height', 'width', 'depth', 'aspect_ratio',
            'density', 'intensity_std', 'intensity_mean', 'volume', 'distance_from_lidar'
        ]



    def cross_validate(self, X_scaled, y, cv_folds=5):
        from sklearn.model_selection import cross_validate
        print(f'\n🔍 {cv_folds}-Fold Cross-Validation (F1 scoring)...')



        rf_temp = RandomForestClassifier(**self.best_params, random_state=self.random_state, n_jobs=-1)
        cv_results = cross_validate(
            rf_temp, X_scaled, y, cv=cv_folds,
            scoring=['accuracy', 'precision', 'recall', 'f1'],
            return_train_score=True
        )



        print(f'  CV F1:     {cv_results["test_f1"].mean():.4f} ± {cv_results["test_f1"].std():.4f}')
        print(f'  CV Acc:    {cv_results["test_accuracy"].mean():.4f} ± {cv_results["test_accuracy"].std():.4f}')
        print(f'  CV Prec:   {cv_results["test_precision"].mean():.4f} ± {cv_results["test_precision"].std():.4f}')
        print(f'  CV Recall: {cv_results["test_recall"].mean():.4f} ± {cv_results["test_recall"].std():.4f}')
        print(f'  Train F1:  {cv_results["train_f1"].mean():.4f} (overfitting check)')



        return cv_results



    def gridsearch(self, X_train, y_train):
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)



        print('🔍 Phase 1: Coarse search (100-400 estimators)...')
        coarse_grid = {
            'n_estimators': [10, 50, 100, 150],
            'max_depth': [5, 10, 15, 25, None],
            'min_samples_split': [2, 5, 15],
            'min_samples_leaf': [1, 2, 6],
            'max_features': ['sqrt', 'log2']
        }



        with Timer('GridSearch Phase 1'):
            coarse_search = GridSearchCV(rf, coarse_grid, cv=5, scoring='f1', n_jobs=-1)
            coarse_search.fit(X_train, y_train)



        best_coarse = coarse_search.best_params_
        print(f'  Coarse best F1: {coarse_search.best_score_:.4f} → {best_coarse}')



        print('🔍 Phase 2: Medium refinement...')
        n_est_start = max(10, best_coarse['n_estimators'] - 30)
        n_est_end = min(450, best_coarse['n_estimators'] + 31)
        med_grid = {
            'n_estimators': list(range(n_est_start, n_est_end, 10)),
            'max_depth': [best_coarse['max_depth']] if best_coarse['max_depth'] is not None else [None, 15, 25],
            'min_samples_split': [best_coarse['min_samples_split']],
            'min_samples_leaf': [best_coarse['min_samples_leaf']],
            'max_features': ['sqrt', 'log2']
        }



        with Timer('GridSearch Phase 2'):
            med_search = GridSearchCV(rf, med_grid, cv=5, scoring='f1', n_jobs=-1)
            med_search.fit(X_train, y_train)



        best_med = med_search.best_params_
        print(f'  Medium best F1: {med_search.best_score_:.4f} → {best_med}')



        print('🔍 Phase 3: Fine tuning...')
        n_est_fine_start = max(10, best_med['n_estimators'] - 15)
        n_est_fine_end = min(450, best_med['n_estimators'] + 16)
        fine_grid = {
            'n_estimators': list(range(n_est_fine_start, n_est_fine_end, 5)),
            'max_depth': [None, 10, 15, 20, 25, 30] if best_med['max_depth'] is None else
                         list(range(max(5, best_med['max_depth'] - 5), min(31, best_med['max_depth'] + 6))),
            'min_samples_split': sorted(set([
                max(2, best_med['min_samples_split'] - 3),
                best_med['min_samples_split'],
                min(20, best_med['min_samples_split'] + 4)
            ])),
            'min_samples_leaf': sorted(set([
                max(1, best_med['min_samples_leaf'] - 2),
                best_med['min_samples_leaf'],
                min(10, best_med['min_samples_leaf'] + 3)
            ])),
            'max_features': ['sqrt', 'log2']
        }



        with Timer('GridSearch Phase 3'):
            fine_search = GridSearchCV(rf, fine_grid, cv=5, scoring='f1', n_jobs=-1)
            fine_search.fit(X_train, y_train)



        print(f'\n✓ Progressive GridSearch Complete!')
        print(f'  Final Best F1: {fine_search.best_score_:.4f}')
        print(f'  Final Best Params: {fine_search.best_params_}')



        self.best_params = fine_search.best_params_
        self.model = fine_search.best_estimator_
        return self.model



    def train(self, X, y, groups, use_gridsearch=True):
        """Train model with grouped split."""
        split_start = time.perf_counter()
        train_idx, test_idx = grouped_train_test_split_best_effort(
            X, y, groups,
            test_size=self.split_test_size,
            n_trials=self.split_trials,
            random_state=self.random_state
        )
        split_elapsed = time.perf_counter() - split_start



        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]



        print(f'\n[Random Forest Training]')
        print(f'  Train: {len(X_train)} | Test: {len(X_test)}')
        print(f'  Split search time: {format_seconds(split_elapsed)}')



        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)



        if use_gridsearch:
            print("Starting gridsearch...")
            self.gridsearch(X_train_scaled, y_train)
        else:
            self.best_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt'
            }
            self.model = RandomForestClassifier(**self.best_params, random_state=self.random_state, n_jobs=-1)
            with Timer('Model fit'):
                self.model.fit(X_train_scaled, y_train)



        X_full_scaled = self.scaler.transform(X)



        print("Cross validating...")
        with Timer('Cross-validation'):
            cv_results = self.cross_validate(X_full_scaled, y)



        self.plot_feature_correlation(X)



        eval_start = time.perf_counter()
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)



        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1_score = f1_score(y_test, y_test_pred, zero_division=0)
        eval_elapsed = time.perf_counter() - eval_start



        print(f'\n✓ Evaluation Results:')
        print(f'  Train Acc:  {train_accuracy:.2%}')
        print(f'  Test Acc:   {test_acc:.2%}')
        print(f'  Precision:  {test_precision:.2%}')
        print(f'  Recall:     {test_recall:.2%}')
        print(f'  F1 Score:   {test_f1_score:.2%}')
        print(f'  Eval time:  {format_seconds(eval_elapsed)}')



        logSaver = LogSaver(log_dir='logs')
        logSaver.save(
            X_train_scaled, X_test_scaled, y_train, y_test,
            train_accuracy, test_acc, test_precision, test_recall, test_f1_score,
            self.feature_names, self.model.feature_importances_, self.best_params, cv_results,
            split_test_size=self.split_test_size,
            split_trials=self.split_trials,
            split_time_seconds=split_elapsed
        )



        self.visualize_confusion_matrix(y_test, y_test_pred)
        self.visualize_feature_importances()



    def plot_feature_correlation(self, X):
        feat_df = pd.DataFrame(X, columns=self.feature_names)
        corr = feat_df.corr()



        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()



        script_dir = Path(__file__).parent
        out_dir = script_dir / 'figures' / 'detection'
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✓ Saved: figures/detection/feature_correlation.png')



    def visualize_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-cone', 'Cone'],
            yticklabels=['Non-cone', 'Cone']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()



        script_dir = Path(__file__).parent
        out_dir = script_dir / 'figures' / 'detection'
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✓ Saved: figures/detection/confusion_matrix.png')



    def visualize_feature_importances(self):
        importances = self.model.feature_importances_
        feat_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)



        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=feat_importance_df,
            x='importance',
            y='feature',
            hue='feature',
            palette='viridis',
            legend=False
        )
        plt.title('Feature Importances')
        plt.xlabel('Importance Score')
        plt.tight_layout()



        script_dir = Path(__file__).parent
        out_dir = script_dir / 'figures' / 'detection'
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'feature_importances.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✓ Saved: figures/detection/feature_importances.png')



    def save(self, path='cone_detector_rf.pkl'):
        data = {
            'scaler': self.scaler,
            'model': self.model,
            'best_params': self.best_params
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f'\n✓ Saved to {path}')



    def save_cpp_ready(self, path='cone_detector.bin'):
        import struct



        scaler_mean = self.scaler.mean_.astype(np.float32)
        scaler_std = self.scaler.scale_.astype(np.float32)
        n_features = len(scaler_mean)



        print(f'Saving {n_features}-feature model to {path}')



        with open(path, 'wb') as f:
            f.write(struct.pack('i', n_features))
            f.write(scaler_mean.tobytes())
            f.write(scaler_std.tobytes())
            f.write(struct.pack('i', self.model.n_estimators))



            for tree_obj in self.model.estimators_:
                tree = tree_obj.tree_
                f.write(struct.pack('i', tree.node_count))
                for i in range(tree.node_count):
                    f.write(struct.pack('i', int(tree.feature[i])))
                    f.write(struct.pack('f', float(tree.threshold[i])))
                    f.write(struct.pack('i', int(tree.children_left[i])))
                    f.write(struct.pack('i', int(tree.children_right[i])))
                    f.write(struct.pack('f', float(tree.value[i][0][0])))
                    f.write(struct.pack('f', float(tree.value[i][0][1])))



        print(f'✓ C++ Ready ({n_features} feats): {path}')



# ============================================================================
# LOG SAVER
# ============================================================================



class LogSaver:
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir).expanduser()
        (self.log_dir / 'detection').mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / 'detection' / 'training_log.txt'



    def save(self, X_train, X_test, y_train, y_test, train_acc, test_acc, precision, recall, f1,
             features, importances, best_params=None, cv_results=None,
             split_test_size=None, split_trials=None, split_time_seconds=None):



        results = {
            'dataset_size': len(X_train) + len(X_test),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'cones_total': int(np.sum(y_train) + np.sum(y_test)),
            'non_cones_total': int(len(y_train) + len(y_test) - np.sum(y_train) - np.sum(y_test)),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'best_params': best_params,
            'cv_f1_mean': float(cv_results['test_f1'].mean()) if cv_results is not None else None,
            'cv_f1_std': float(cv_results['test_f1'].std()) if cv_results is not None else None,
            'feature_importances': dict(zip(features, importances.tolist()))
        }



        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write('\n')
            f.write(json.dumps(results))
            f.write('\n')



        print(f'Training log appended to: {self.log_file}')
        return self.log_file



# ============================================================================
# MAIN
# ============================================================================



def main():
    import argparse



    parser = argparse.ArgumentParser(description='Random Forest Cone Detector')
    script_dir = Path(__file__).parent



    default_dataset = script_dir / 'Dataset' / 'Processed'
    default_output = script_dir / 'models' / 'detection' / 'cone_detector_rf.pkl'
    default_output_bin = script_dir / 'models' / 'detection' / 'cone_detector.bin'



    parser.add_argument('--dataset', default=str(default_dataset),
                        help='Path to Dataset folder')
    parser.add_argument('--output', default=str(default_output))
    parser.add_argument('--output-bin', default=str(default_output_bin))



    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Requested test split ratio at sample level, approximated via group-aware search')
    parser.add_argument('--split-trials', type=int, default=200,
                        help='How many grouped split candidates to try before picking the closest one')
    parser.add_argument('--split-group-level', choices=['parent', 'track'], default='parent',
                        help='Grouping granularity: parent folder or full track')
    parser.add_argument('--no-gridsearch', action='store_true',
                        help='Skip progressive grid search and fit a default RF directly')
    parser.add_argument('--random-state', type=int, default=42)



    args = parser.parse_args()



    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_bin).parent.mkdir(parents=True, exist_ok=True)



    total_start = time.perf_counter()



    builder = MultiTrackDatasetBuilder(
        args.dataset,
        split_group_level=args.split_group_level
    )



    with Timer('Dataset build'):
        X, y, groups = builder.build()



    if len(X) < 50:
        print('⚠️  Too few samples!')
        return



    model = RandomForestConeDetector(
        split_test_size=args.test_size,
        split_trials=args.split_trials,
        random_state=args.random_state
    )



    with Timer('Full training pipeline'):
        model.train(X, y, groups, use_gridsearch=not args.no_gridsearch)



    model.save(args.output)
    model.save_cpp_ready(args.output_bin)



    total_elapsed = time.perf_counter() - total_start
    print(f'\n🚀 Model trained and saved!')
    print(f'⏱️  Total wall time: {format_seconds(total_elapsed)}')



if __name__ == '__main__':
    main()