#!/usr/bin/env python3
import json
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
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

class XGBoostConeDetector:
    def __init__(self, repo_root, random_state=42):
        self.repo_root = Path(repo_root)
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.best_params = None
        self.feature_names = FEATURE_NAMES

    def cross_validate(self, X_scaled, y, cv_folds=5):
        print(f'\n🔍 {cv_folds}-Fold Cross-Validation (F1 scoring)...')
        scale_pos_weight = (len(y) - sum(y)) / max(sum(y), 1)

        xgb_temp = xgb.XGBClassifier(
            **self.best_params,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )

        cv_results = cross_validate(
            xgb_temp, X_scaled, y, cv=cv_folds,
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
        scale_pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)

        xgb_clf = xgb.XGBClassifier(
            random_state=self.random_state,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            n_jobs=-1,
            tree_method='hist'
        )

        coarse_grid = {
            'n_estimators': [300, 400, 500],
            'max_depth': [3, 5, 7, 9, 12],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
            'min_child_weight': [1, 3, 5, 8],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0.0, 0.1, 0.3, 0.5],
            'reg_alpha': [0.0, 0.01, 0.1],
            'reg_lambda': [0.5, 1.0, 2.0, 5.0],
        }

        print('\n🔍 Phase 1: Coarse XGBoost search...')
        with Timer('XGBoost GridSearch Phase 1'):
            coarse_search = GridSearchCV(
                xgb_clf,
                coarse_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1,
            )
            coarse_search.fit(X_train, y_train)

        best_coarse = coarse_search.best_params_
        print(f'  Coarse best F1: {coarse_search.best_score_:.4f} → {best_coarse}')

        c_nest = best_coarse['n_estimators']
        c_depth = best_coarse['max_depth']
        c_lr = best_coarse['learning_rate']
        c_child = best_coarse['min_child_weight']
        c_subsample = best_coarse['subsample']
        c_colsample = best_coarse['colsample_bytree']
        c_gamma = best_coarse['gamma']
        c_alpha = best_coarse['reg_alpha']
        c_lambda = best_coarse['reg_lambda']

        med_grid = {
            'n_estimators': list(range(max(50, c_nest - 100), c_nest + 101, 25)),
            'max_depth': sorted(set([
                max(2, c_depth - 2),
                max(2, c_depth - 1),
                c_depth,
                c_depth + 1,
                c_depth + 2,
            ])),
            'learning_rate': sorted(set([
                max(0.005, c_lr / 2),
                max(0.005, c_lr * 0.75),
                c_lr,
                c_lr * 1.25,
                c_lr * 1.5,
            ])),
            'min_child_weight': sorted(set([
                max(1, c_child - 2),
                max(1, c_child - 1),
                c_child,
                c_child + 1,
                c_child + 2,
            ])),
            'subsample': sorted(set([
                max(0.5, c_subsample - 0.1),
                c_subsample,
                min(1.0, c_subsample + 0.1),
            ])),
            'colsample_bytree': sorted(set([
                max(0.5, c_colsample - 0.1),
                c_colsample,
                min(1.0, c_colsample + 0.1),
            ])),
            'gamma': sorted(set([
                max(0.0, c_gamma - 0.1),
                c_gamma,
                c_gamma + 0.1,
            ])),
            'reg_alpha': sorted(set([
                max(0.0, c_alpha / 2),
                c_alpha,
                c_alpha * 2 if c_alpha > 0 else 0.01,
            ])),
            'reg_lambda': sorted(set([
                max(0.01, c_lambda / 2),
                c_lambda,
                c_lambda * 2,
            ])),
        }

        print('\n🔍 Phase 2: Medium XGBoost refinement...')
        with Timer('XGBoost GridSearch Phase 2'):
            med_search = GridSearchCV(
                xgb_clf,
                med_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1,
            )
            med_search.fit(X_train, y_train)

        best_med = med_search.best_params_
        print(f'  Medium best F1: {med_search.best_score_:.4f} → {best_med}')

        m_nest = best_med['n_estimators']
        m_depth = best_med['max_depth']
        m_lr = best_med['learning_rate']
        m_child = best_med['min_child_weight']
        m_subsample = best_med['subsample']
        m_colsample = best_med['colsample_bytree']
        m_gamma = best_med['gamma']
        m_alpha = best_med['reg_alpha']
        m_lambda = best_med['reg_lambda']

        fine_grid = {
            'n_estimators': list(range(max(20, m_nest - 30), m_nest + 31, 10)),
            'max_depth': sorted(set([
                max(2, m_depth - 1),
                m_depth,
                m_depth + 1,
            ])),
            'learning_rate': sorted(set([
                max(0.001, m_lr * 0.8),
                m_lr,
                m_lr * 1.2,
            ])),
            'min_child_weight': sorted(set([
                max(1, m_child - 1),
                m_child,
                m_child + 1,
            ])),
            'subsample': sorted(set([
                max(0.5, m_subsample - 0.05),
                m_subsample,
                min(1.0, m_subsample + 0.05),
            ])),
            'colsample_bytree': sorted(set([
                max(0.5, m_colsample - 0.05),
                m_colsample,
                min(1.0, m_colsample + 0.05),
            ])),
            'gamma': sorted(set([
                max(0.0, m_gamma - 0.05),
                m_gamma,
                m_gamma + 0.05,
            ])),
            'reg_alpha': sorted(set([
                max(0.0, m_alpha * 0.8),
                m_alpha,
                m_alpha * 1.2 if m_alpha > 0 else 0.01,
            ])),
            'reg_lambda': sorted(set([
                max(0.01, m_lambda * 0.8),
                m_lambda,
                m_lambda * 1.2,
            ])),
        }

        print('\n🔍 Phase 3: Fine XGBoost tuning...')
        with Timer('XGBoost GridSearch Phase 3'):
            fine_search = GridSearchCV(
                xgb_clf,
                fine_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1,
            )
            fine_search.fit(X_train, y_train)

        print('\n✓ Progressive XGBoost GridSearch Complete!')
        print(f'  Final Best F1: {fine_search.best_score_:.4f}')
        print(f'  Final Best Params: {fine_search.best_params_}')

        self.best_params = fine_search.best_params_
        self.model = fine_search.best_estimator_
        return self.model

    def train(self, X, y, use_gridsearch=True):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        print(f'\n[XGBoost Detection Training]')
        print(f'  Train: {len(X_train)} (Cones: {int(y_train.sum())}, Non-cones: {int(len(y_train) - y_train.sum())})')
        print(f'  Test:  {len(X_test)} (Cones: {int(y_test.sum())}, Non-cones: {int(len(y_test) - y_test.sum())})')

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if use_gridsearch:
            self.gridsearch(X_train_scaled, y_train)
        else:
            scale_pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
            self.best_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
            self.model = xgb.XGBClassifier(
                **self.best_params,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                eval_metric='logloss',
                n_jobs=-1
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

        logger = TrainingLogger(self.repo_root, "xgboost")
        logger.log_metrics(log_metrics)

    def plot_feature_correlation(self, X):
        feat_df = pd.DataFrame(X, columns=self.feature_names)
        corr = feat_df.corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix - XGBoost', fontsize=14)
        plt.tight_layout()

        out_dir = self.repo_root / 'figures' / 'detection' / 'xgboost'
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'xgb_feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Cone', 'Cone'], yticklabels=['Non-Cone', 'Cone'])
        plt.title('Confusion Matrix - XGBoost')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()

        out_dir = self.repo_root / 'figures' / 'detection' / 'xgboost'
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'xgb_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_feature_importances(self):
        importances = self.model.feature_importances_
        df = pd.DataFrame({'feature': self.feature_names, 'importance': importances}).sort_values('importance', ascending=True)

        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x='importance', y='feature', palette='viridis')
        plt.title('Feature Importances - XGBoost')
        plt.tight_layout()

        out_dir = self.repo_root / 'figures' / 'detection' / 'xgboost'
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'xgb_feature_importances.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save(self, pkl_path, json_path):
        Path(pkl_path).parent.mkdir(parents=True, exist_ok=True)
        with open(pkl_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'model': self.model, 'best_params': self.best_params}, f)

        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(json_path))

def run_xgb_pipeline(X, y, repo_root, use_gridsearch=True):
    detector = XGBoostConeDetector(repo_root)
    detector.train(X, y, use_gridsearch=use_gridsearch)
    detector.save(
        repo_root / 'models' / 'detection' / 'xgboost' / 'cone_detector_xgb.pkl',
        repo_root / 'models' / 'detection' / 'xgboost' / 'cone_detector_xgb.json'
    )