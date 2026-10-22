#!/usr/bin/env python3
import json
import time
import pickle
import struct
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from math import prod
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, cross_validate, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from features import FEATURE_NAMES


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        print(f"  [{self.name}] Completed in {self.elapsed:.2f}s")


class TrainingLogger:
    def __init__(self, repo_root: Path, model_subfolder: str):
        self.log_dir = Path(repo_root) / "logs" / "detection" / model_subfolder
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training_log.txt"

    def log_metrics(self, metrics: dict):
        with open(self.log_file, "a") as f:
            f.write("\n" + json.dumps(metrics))
        print(f'  Saved: {self.log_file.resolve()}')


class RandomForestConeDetector:
    def __init__(self, repo_root, random_state=42, feature_names=None, output_name="random_forest"):
        self.repo_root = Path(repo_root)
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.best_params = None
        self.best_cv_score = None
        self.feature_names = list(feature_names) if feature_names is not None else FEATURE_NAMES
        self.output_name = output_name

    def cross_validate(self, X_scaled, y, groups, cv_folds=5):
        effective_folds = min(cv_folds, len(np.unique(groups)))
        print(f'\n🔍 {effective_folds}-Fold Cross-Validation (Binary F1 scoring)...')
        rf_temp = RandomForestClassifier(
            **self.best_params,
            random_state=self.random_state,
            n_jobs=1
        )
        
        cv_results = cross_validate(
            rf_temp, X_scaled, y, groups=groups,
            cv=StratifiedGroupKFold(n_splits=effective_folds, shuffle=True, random_state=self.random_state),
            scoring=['accuracy', 'precision', 'recall', 'f1'],
            return_train_score=True, n_jobs=-1
        )

        print(f'  CV F1:          {cv_results["test_f1"].mean():.4f} ± {cv_results["test_f1"].std():.4f}')
        print(f'  CV Accuracy:    {cv_results["test_accuracy"].mean():.4f} ± {cv_results["test_accuracy"].std():.4f}')
        print(f'  CV Precision:   {cv_results["test_precision"].mean():.4f} ± {cv_results["test_precision"].std():.4f}')
        print(f'  CV Recall:      {cv_results["test_recall"].mean():.4f} ± {cv_results["test_recall"].std():.4f}')
        print(f'  Train F1:       {cv_results["train_f1"].mean():.4f} (overfitting check)')

        return cv_results

    def gridsearch(self, X_train, y_train, groups_train=None):
        rf = RandomForestClassifier(
            random_state=self.random_state,
            class_weight='balanced_subsample',
            n_jobs=1
        )

        global_best_score = -1.0
        global_best_params = None
        global_best_estimator = None
        global_best_phase = None

        if groups_train is not None:
            n_unique_groups = len(np.unique(groups_train))
            n_splits = min(5, n_unique_groups)
            cv_splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        else:
            cv_splitter = 5

        def evaluate_phase(search, phase_name, grid):
            nonlocal global_best_score, global_best_params, global_best_estimator, global_best_phase

            n_combinations = prod(len(values) for values in grid.values())
            total_fits = n_combinations * (cv_splitter.get_n_splits() if hasattr(cv_splitter, 'get_n_splits') else cv_splitter)

            print(
                f'\n🔍 {phase_name}... '
                f'({n_combinations} candidates, {total_fits} total fits)'
            )

            with Timer(f'GridSearch {phase_name}'):
                if groups_train is not None:
                    search.fit(X_train, y_train, groups=groups_train)
                else:
                    search.fit(X_train, y_train)

            best_idx = search.best_index_
            mean_cv_score = float(search.cv_results_['mean_test_score'][best_idx])
            mean_train_score = float(search.cv_results_['mean_train_score'][best_idx])
            overfit_gap = mean_train_score - mean_cv_score

            print(
                f'   {phase_name} Best F1 (CV): {mean_cv_score:.4f} '
                f'| Train F1: {mean_train_score:.4f} '
                f'| Overfit Gap: {overfit_gap:.4f}'
            )

            if mean_cv_score > global_best_score:
                global_best_score = mean_cv_score
                global_best_params = search.best_params_.copy()
                global_best_estimator = search.best_estimator_
                global_best_phase = phase_name
                print(f'   ✓ New global best from {phase_name}: {global_best_score:.4f}')
            else:
                print(f'   ↳ Global best remains: {global_best_score:.4f} from {global_best_phase}')

            return search

        # ==========================================
        # Phase 1: Baseline Coarse Search
        # ==========================================
        coarse_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [8, 14, 20],
            'min_samples_split': [8, 16],
            'min_samples_leaf': [3, 6],
            'max_features': ['sqrt'],
            'class_weight': ['balanced_subsample']
        }

        coarse_search = GridSearchCV(
            estimator=rf,
            param_grid=coarse_grid,
            cv=cv_splitter,
            scoring='f1',
            n_jobs=-1,
            refit=True,
            return_train_score=True,
            verbose=1
        )

        evaluate_phase(coarse_search, 'Phase 1: Coarse search', coarse_grid)
        best_coarse = coarse_search.best_params_

        # ==========================================
        # Phase 2: Medium Refinement
        # ==========================================
        c_nest = best_coarse['n_estimators']
        c_depth = best_coarse['max_depth']
        c_split = best_coarse['min_samples_split']
        c_leaf = best_coarse['min_samples_leaf']

        medium_grid = {
            'n_estimators': sorted({max(50, c_nest - 50), c_nest, c_nest + 50}),
            'max_depth': sorted({max(4, c_depth - 2), c_depth, min(25, c_depth + 2)}),
            'min_samples_split': sorted({max(4, c_split - 4), c_split, c_split + 4}),
            'min_samples_leaf': sorted({max(2, c_leaf - 2), c_leaf, c_leaf + 2}),
            'max_features': ['sqrt'],
            'class_weight': ['balanced_subsample']
        }

        medium_search = GridSearchCV(
            estimator=rf,
            param_grid=medium_grid,
            cv=cv_splitter,
            scoring='f1',
            n_jobs=-1,
            refit=True,
            return_train_score=True,
            verbose=1
        )

        evaluate_phase(medium_search, 'Phase 2: Medium refinement', medium_grid)
        best_medium = medium_search.best_params_

        # ==========================================
        # Phase 3: Fine Tuning
        # ==========================================
        m_nest = best_medium['n_estimators']
        m_depth = best_medium['max_depth']
        m_split = best_medium['min_samples_split']
        m_leaf = best_medium['min_samples_leaf']

        fine_grid = {
            'n_estimators': sorted({max(30, m_nest - 20), m_nest, m_nest + 20}),
            'max_depth': [m_depth],
            'min_samples_split': sorted({max(2, m_split - 2), m_split, m_split + 2}),
            'min_samples_leaf': sorted({max(1, m_leaf - 1), m_leaf, m_leaf + 1}),
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced_subsample']
        }

        fine_search = GridSearchCV(
            estimator=rf,
            param_grid=fine_grid,
            cv=cv_splitter,
            scoring='f1',
            n_jobs=-1,
            refit=True,
            return_train_score=True,
            verbose=1
        )

        evaluate_phase(fine_search, 'Phase 3: Fine tuning', fine_grid)

        if global_best_estimator is None:
            raise RuntimeError('Progressive grid search did not produce a valid RF estimator.')

        self.best_params = global_best_params
        self.model = global_best_estimator
        self.best_cv_score = global_best_score

        print('\n✓ Progressive Random Forest GridSearch Complete!')
        print(f'   Winning phase:   {global_best_phase}')
        print(f'   Best F1:         {global_best_score:.4f}')
        print(f'   Best parameters: {self.best_params}')

        return self.model

    def train(self, X, y, groups, use_gridsearch=True, train_idx=None, test_idx=None):
        if train_idx is None or test_idx is None:
            train_idx, test_idx = next(GroupShuffleSplit(
                n_splits=1, test_size=0.2, random_state=self.random_state
            ).split(X, y, groups))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]

        if not set(groups_train).isdisjoint(set(groups_test)):
            raise RuntimeError("Run leakage detected between train and test groups.")

        print(f'\n[Random Forest Detection Training]')
        print(f'  Train: {len(X_train)} (Cones: {int(y_train.sum())}, Non-cones: {int(len(y_train) - y_train.sum())})')
        print(f'  Test:  {len(X_test)} (Cones: {int(y_test.sum())}, Non-cones: {int(len(y_test) - y_test.sum())})')

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if use_gridsearch:
            self.gridsearch(X_train_scaled, y_train, groups_train)
        else:
            self.best_params = {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt'
            }
            self.model = RandomForestClassifier(
                **self.best_params, random_state=self.random_state,
                class_weight='balanced_subsample', n_jobs=-1
            )
            with Timer('Model fit'):
                self.model.fit(X_train_scaled, y_train)

        with Timer('Cross-validation'):
            cv_results = self.cross_validate(X_train_scaled, y_train, groups_train)

        self.plot_feature_correlation(X)

        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1_score = f1_score(y_test, y_test_pred, zero_division=0)

        print(f'\n✓ Evaluation Results:')
        print(f'  Train F1: {train_f1:.2%}')
        print(f'  Test Acc:  {test_acc:.2%}')
        print(f'  Precision: {test_precision:.2%}')
        print(f'  Recall:    {test_recall:.2%}')
        print(f'  F1 Score:  {test_f1_score:.2%}')

        self.visualize_confusion_matrix(y_test, y_test_pred)
        self.visualize_feature_importances()

        log_metrics = {
            "dataset_size": len(X),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_f1": float(train_f1),
            "test_accuracy": float(test_acc),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "f1_score": float(test_f1_score),
            "best_params": self.best_params,
            "cv_f1_mean": float(cv_results["test_f1"].mean()),
            "cv_f1_std": float(cv_results["test_f1"].std()),
            "feature_importances": {f: float(i) for f, i in zip(self.feature_names, self.model.feature_importances_)},
            "feature_names": self.feature_names,
            "gridsearch_cv_f1": getattr(self, "best_cv_score", None),
            "f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
            "noncone_precision": float(precision_score(y_test, y_test_pred, pos_label=0, zero_division=0)),
            "noncone_recall": float(recall_score(y_test, y_test_pred, pos_label=0, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_test_pred, labels=[0, 1]).tolist(),
        }

        logger = TrainingLogger(self.repo_root, self.output_name)
        logger.log_metrics(log_metrics)
        return log_metrics

    def plot_feature_correlation(self, X):
        feat_df = pd.DataFrame(X, columns=self.feature_names)
        corr = feat_df.corr()

        n = len(self.feature_names)
        plt.figure(figsize=(max(8, 0.75 * n), max(6, 0.6 * n)))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix - Random Forest', fontsize=14)
        plt.tight_layout()

        out_dir = self.repo_root / 'figures' / 'detection' / self.output_name
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'rf_feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {(out_dir / "rf_feature_correlation.png").resolve()}')

    def visualize_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Cone', 'Cone'], yticklabels=['Non-Cone', 'Cone'])
        plt.title('Confusion Matrix - Random Forest')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()

        out_dir = self.repo_root / 'figures' / 'detection' / self.output_name
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {(out_dir / "rf_confusion_matrix.png").resolve()}  (matrix total = {cm.sum()})')

    def visualize_feature_importances(self):
        importances = self.model.feature_importances_
        df = pd.DataFrame({'feature': self.feature_names, 'importance': importances}).sort_values('importance', ascending=True)

        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x='importance', y='feature', hue='feature', palette='viridis', legend=False)
        plt.title('Feature Importances - Random Forest')
        plt.tight_layout()

        out_dir = self.repo_root / 'figures' / 'detection' / self.output_name
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'rf_feature_importances.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {(out_dir / "rf_feature_importances.png").resolve()}')

    def save(self, pkl_path, bin_path):
        Path(pkl_path).parent.mkdir(parents=True, exist_ok=True)
        with open(pkl_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'model': self.model, 'best_params': self.best_params}, f)

        Path(bin_path).parent.mkdir(parents=True, exist_ok=True)
        scaler_mean = self.scaler.mean_.astype(np.float32)
        scaler_std = self.scaler.scale_.astype(np.float32)

        with open(bin_path, 'wb') as f:
            f.write(struct.pack('ii', len(scaler_mean), len(self.model.classes_)))
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
                    vals = tree.value[i][0].astype(np.float32)
                    probs = vals / (np.sum(vals) + 1e-6)
                    f.write(probs.tobytes())


def run_rf_pipeline(
    X,
    y,
    groups,
    repo_root,
    use_gridsearch=True,
    train_idx=None,
    test_idx=None,
    feature_names=None,
    output_name="random_forest",
):
    detector = RandomForestConeDetector(repo_root, feature_names=feature_names, output_name=output_name)
    detector.train(
        X,
        y,
        groups,
        use_gridsearch=use_gridsearch,
        train_idx=train_idx,
        test_idx=test_idx,
    )
    detector.save(
        repo_root / 'models' / 'detection' / output_name / 'cone_detector_rf.pkl',
        repo_root / 'models' / 'detection' / output_name / 'cone_detector_rf.bin'
    )