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

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
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

class RandomForestConeDetector:
    def __init__(self, repo_root, random_state=42):
        self.repo_root = Path(repo_root)
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.best_params = None
        self.feature_names = FEATURE_NAMES

    def cross_validate(self, X_scaled, y, cv_folds=5):
        print(f'\n🔍 {cv_folds}-Fold Cross-Validation (F1 scoring)...')
        rf_temp = RandomForestClassifier(
            **self.best_params,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        cv_results = cross_validate(
            rf_temp, X_scaled, y, cv=cv_folds,
            scoring=['accuracy', 'precision', 'recall', 'f1'],
            return_train_score=True, n_jobs=-1
        )

        print(f'  CV F1:        {cv_results["test_f1"].mean():.4f} ± {cv_results["test_f1"].std():.4f}')
        print(f'  CV Accuracy:  {cv_results["test_accuracy"].mean():.4f} ± {cv_results["test_accuracy"].std():.4f}')
        print(f'  CV Precision: {cv_results["test_precision"].mean():.4f} ± {cv_results["test_precision"].std():.4f}')
        print(f'  CV Recall:    {cv_results["test_recall"].mean():.4f} ± {cv_results["test_recall"].std():.4f}')
        print(f'  Train F1:     {cv_results["train_f1"].mean():.4f} (overfitting check)')

        return cv_results

    def gridsearch(self, X_train, y_train):
        rf = RandomForestClassifier(random_state=self.random_state, class_weight='balanced')

        coarse_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [10, 15, 20, 25, 30, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6, 8, 12],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        print('\n🔍 Phase 1: Coarse search...')
        with Timer('GridSearch Phase 1'):
            coarse_search = GridSearchCV(rf, coarse_grid, cv=5, scoring='f1', n_jobs=-1)#, verbose=2)
            coarse_search.fit(X_train, y_train)

        best_coarse = coarse_search.best_params_
        print(f'  Coarse best F1: {coarse_search.best_score_:.4f} → {best_coarse}')

        c_nest = best_coarse['n_estimators']
        c_depth = best_coarse['max_depth']
        c_split = best_coarse['min_samples_split']
        c_leaf = best_coarse['min_samples_leaf']

        med_grid = {
            'n_estimators': list(range(max(50, c_nest - 60), c_nest + 61, 20)),
            'max_depth': [None, 25, 35] if c_depth is None else sorted(set([max(5, c_depth - 5), c_depth, c_depth + 5])),
            'min_samples_split': sorted(set([max(2, c_split - 2), c_split, c_split + 2])),
            'min_samples_leaf': sorted(set([max(1, c_leaf - 1), c_leaf, c_leaf + 1])),
            'max_features': [best_coarse['max_features']],
            'class_weight': ['balanced', 'balanced_subsample']
        }

        print('\n🔍 Phase 2: Medium refinement...')
        with Timer('GridSearch Phase 2'):
            med_search = GridSearchCV(rf, med_grid, cv=5, scoring='f1', n_jobs=-1)#,  verbose=2)
            med_search.fit(X_train, y_train)

        best_med = med_search.best_params_
        print(f'  Medium best F1: {med_search.best_score_:.4f} → {best_med}')

        m_nest = best_med['n_estimators']
        m_depth = best_med['max_depth']
        m_split = best_med['min_samples_split']
        m_leaf = best_med['min_samples_leaf']

        fine_grid = {
            'n_estimators': list(range(max(20, m_nest - 15), m_nest + 16, 5)),
            'max_depth': [None] if m_depth is None else sorted(set([max(3, m_depth - 2), m_depth, m_depth + 2])),
            'min_samples_split': sorted(set([max(2, m_split - 1), m_split, m_split + 1])),
            'min_samples_leaf': sorted(set([max(1, m_leaf - 1), m_leaf, m_leaf + 1])),
            'max_features': [best_med['max_features']],
            'class_weight': ['balanced', 'balanced_subsample']
        }

        print('\n🔍 Phase 3: Fine tuning...')
        with Timer('GridSearch Phase 3'):
            fine_search = GridSearchCV(rf, fine_grid, cv=5, scoring='f1', n_jobs=-1)#, verbose=2)
            fine_search.fit(X_train, y_train)

        print(f'\n✓ Progressive GridSearch Complete!')
        print(f'  Final Best F1: {fine_search.best_score_:.4f}')
        print(f'  Final Best Params: {fine_search.best_params_}')

        self.best_params = fine_search.best_params_
        self.model = fine_search.best_estimator_
        return self.model

    def train(self, X, y, use_gridsearch=True):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        print(f'\n[Random Forest Detection Training]')
        print(f'  Train: {len(X_train)} (Cones: {int(y_train.sum())}, Non-cones: {int(len(y_train) - y_train.sum())})')
        print(f'  Test:  {len(X_test)} (Cones: {int(y_test.sum())}, Non-cones: {int(len(y_test) - y_test.sum())})')

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if use_gridsearch:
            self.gridsearch(X_train_scaled, y_train)
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
                class_weight='balanced', n_jobs=-1
            )
            with Timer('Model fit'):
                self.model.fit(X_train_scaled, y_train)

        X_full_scaled = self.scaler.transform(X)
        with Timer('Cross-validation'):
            cv_results = self.cross_validate(X_full_scaled, y)

        self.plot_feature_correlation(X)

        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1_score = f1_score(y_test, y_test_pred, zero_division=0)

        print(f'\n✓ Evaluation Results:')
        print(f'  Train Acc: {train_accuracy:.2%}')
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
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_acc),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "f1_score": float(test_f1_score),
            "best_params": self.best_params,
            "cv_f1_mean": float(cv_results["test_f1"].mean()),
            "cv_f1_std": float(cv_results["test_f1"].std()),
            "feature_importances": {f: float(i) for f, i in zip(self.feature_names, self.model.feature_importances_)}
        }

        logger = TrainingLogger(self.repo_root, "random_forest")
        logger.log_metrics(log_metrics)

    def plot_feature_correlation(self, X):
        feat_df = pd.DataFrame(X, columns=self.feature_names)
        corr = feat_df.corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix - Random Forest', fontsize=14)
        plt.tight_layout()

        out_dir = self.repo_root / 'figures' / 'detection' / 'random_forest'
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'rf_feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Cone', 'Cone'], yticklabels=['Non-Cone', 'Cone'])
        plt.title('Confusion Matrix - Random Forest')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()

        out_dir = self.repo_root / 'figures' / 'detection' / 'random_forest'
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_feature_importances(self):
        importances = self.model.feature_importances_
        df = pd.DataFrame({'feature': self.feature_names, 'importance': importances}).sort_values('importance', ascending=True)

        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x='importance', y='feature', palette='viridis')
        plt.title('Feature Importances - Random Forest')
        plt.tight_layout()

        out_dir = self.repo_root / 'figures' / 'detection' / 'random_forest'
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'rf_feature_importances.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save(self, pkl_path, bin_path):
        Path(pkl_path).parent.mkdir(parents=True, exist_ok=True)
        with open(pkl_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'model': self.model, 'best_params': self.best_params}, f)

        Path(bin_path).parent.mkdir(parents=True, exist_ok=True)
        scaler_mean = self.scaler.mean_.astype(np.float32)
        scaler_std = self.scaler.scale_.astype(np.float32)

        with open(bin_path, 'wb') as f:
            f.write(struct.pack('i', len(scaler_mean)))
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

def run_rf_pipeline(X, y, repo_root, use_gridsearch=True):
    detector = RandomForestConeDetector(repo_root)
    detector.train(X, y, use_gridsearch=use_gridsearch)
    detector.save(
        repo_root / 'models' / 'detection' / 'random_forest' / 'cone_detector_rf.pkl',
        repo_root / 'models' / 'detection' / 'random_forest' / 'cone_detector_rf.bin'
    )