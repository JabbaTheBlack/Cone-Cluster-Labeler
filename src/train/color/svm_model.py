#!/usr/bin/env python3
"""One-feature SVM yellow vs non-yellow cone colour classifier.

Expected input X shape: (n_clusters, 1)
Feature: distance_compensated_mean_intensity
Class mapping: 0=non-yellow, 1=yellow
"""

import json
import pickle
import time
from math import prod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import SVM_FEATURE_NAMES


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"  [{self.name}] Completed in {elapsed:.2f}s")


class TrainingLogger:
    def __init__(self, repo_root: Path, model_subfolder: str):
        self.log_dir = Path(repo_root) / "logs" / "color" / model_subfolder
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training_log.txt"

    def log_metrics(self, metrics: dict):
        with self.log_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics) + "\n")


class SVMColorClassifier:
    """RBF SVM for the one-feature 0=non-yellow, 1=yellow task."""

    def __init__(self, repo_root, random_state=42):
        self.repo_root = Path(repo_root)
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.label_encoder = None
        self.feature_names = SVM_FEATURE_NAMES

    def _pipeline(self):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=self.random_state,
            )),
        ])

    @staticmethod
    def _params_with_prefix(params):
        return {f"svc__{key}": value for key, value in params.items()}

    @staticmethod
    def _strip_prefix(params):
        return {key.removeprefix("svc__"): value for key, value in params.items()}

    def gridsearch(self, X_train, y_train):
        model = self._pipeline()

        coarse_grid = {
            "svc__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "svc__gamma": ["scale", 0.001, 0.01, 0.1, 1.0],
            "svc__kernel": ["rbf"],
        }
        coarse_candidates = prod(len(values) for values in coarse_grid.values())
        print(
            f"\n🔍 Phase 1: Coarse one-feature SVM search... "
            f"({coarse_candidates} candidates, {coarse_candidates * 5} total fits)"
        )

        with Timer("SVM GridSearch Phase 1"):
            coarse_search = GridSearchCV(
                estimator=model,
                param_grid=coarse_grid,
                cv=5,
                scoring="f1_macro",
                n_jobs=-1,
                verbose=1,
            )
            coarse_search.fit(X_train, y_train)

        best_coarse = coarse_search.best_params_
        print(
            f"  Coarse best F1 Macro: {coarse_search.best_score_:.4f} "
            f"→ {self._strip_prefix(best_coarse)}"
        )

        c_value = float(best_coarse["svc__C"])
        c_gamma = best_coarse["svc__gamma"]
        c_values = sorted(set([
            max(1e-4, c_value / 5.0),
            max(1e-4, c_value / 2.0),
            c_value,
            c_value * 2.0,
            c_value * 5.0,
        ]))

        if c_gamma == "scale":
            gamma_values = ["scale", 0.001, 0.01, 0.1]
        else:
            gamma_value = float(c_gamma)
            gamma_values = sorted(set([
                max(1e-6, gamma_value / 5.0),
                max(1e-6, gamma_value / 2.0),
                gamma_value,
                gamma_value * 2.0,
                gamma_value * 5.0,
            ]))

        med_grid = {
            "svc__C": c_values,
            "svc__gamma": gamma_values,
            "svc__kernel": ["rbf"],
        }
        med_candidates = prod(len(values) for values in med_grid.values())
        print(
            f"\n🔍 Phase 2: Medium one-feature SVM refinement... "
            f"({med_candidates} candidates, {med_candidates * 5} total fits)"
        )

        with Timer("SVM GridSearch Phase 2"):
            med_search = GridSearchCV(
                estimator=model,
                param_grid=med_grid,
                cv=5,
                scoring="f1_macro",
                n_jobs=-1,
                verbose=1,
            )
            med_search.fit(X_train, y_train)

        best_med = med_search.best_params_
        print(
            f"  Medium best F1 Macro: {med_search.best_score_:.4f} "
            f"→ {self._strip_prefix(best_med)}"
        )

        m_value = float(best_med["svc__C"])
        m_gamma = best_med["svc__gamma"]
        fine_c_values = sorted(set([
            max(1e-4, m_value * 0.7),
            m_value,
            m_value * 1.3,
        ]))

        if m_gamma == "scale":
            fine_gamma_values = ["scale", 0.001, 0.01]
        else:
            m_gamma_value = float(m_gamma)
            fine_gamma_values = sorted(set([
                max(1e-6, m_gamma_value * 0.7),
                m_gamma_value,
                m_gamma_value * 1.3,
            ]))

        fine_grid = {
            "svc__C": fine_c_values,
            "svc__gamma": fine_gamma_values,
            "svc__kernel": ["rbf"],
        }
        fine_candidates = prod(len(values) for values in fine_grid.values())
        print(
            f"\n🔍 Phase 3: Fine one-feature SVM tuning... "
            f"({fine_candidates} candidates, {fine_candidates * 5} total fits)"
        )

        with Timer("SVM GridSearch Phase 3"):
            fine_search = GridSearchCV(
                estimator=model,
                param_grid=fine_grid,
                cv=5,
                scoring="f1_macro",
                n_jobs=-1,
                verbose=1,
            )
            fine_search.fit(X_train, y_train)

        self.best_params = self._strip_prefix(fine_search.best_params_)
        self.model = fine_search.best_estimator_

        print("\n✓ Progressive one-feature SVM GridSearch Complete!")
        print(f"  Final Best F1 Macro: {fine_search.best_score_:.4f}")
        print(f"  Final Best Params:   {self.best_params}")
        return self.model

    def cross_validate(self, X, y, cv_folds=5):
        print(f"\n🔍 {cv_folds}-Fold Cross-Validation (Macro F1 scoring)...")

        model = self._pipeline()
        model.set_params(**self._params_with_prefix(self.best_params))

        cv_results = cross_validate(
            estimator=model,
            X=X,
            y=y,
            cv=cv_folds,
            scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
            return_train_score=True,
            n_jobs=-1,
        )

        print(f'  CV F1 Macro:  {cv_results["test_f1_macro"].mean():.4f} ± {cv_results["test_f1_macro"].std():.4f}')
        print(f'  CV Accuracy:  {cv_results["test_accuracy"].mean():.4f} ± {cv_results["test_accuracy"].std():.4f}')
        print(f'  CV Precision: {cv_results["test_precision_macro"].mean():.4f} ± {cv_results["test_precision_macro"].std():.4f}')
        print(f'  CV Recall:    {cv_results["test_recall_macro"].mean():.4f} ± {cv_results["test_recall_macro"].std():.4f}')
        print(f'  Train F1:     {cv_results["train_f1_macro"].mean():.4f} (overfitting check)')
        return cv_results

    def train(self, X, y, label_encoder, use_gridsearch=True):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.label_encoder = label_encoder

        expected_feature_count = len(self.feature_names)
        if X.ndim != 2 or X.shape[1] != expected_feature_count:
            raise ValueError(
                f"Expected one-feature X shape (N, {expected_feature_count}), received {X.shape}. "
                "Build the SVM dataset with extract_svm_intensity_feature()."
            )

        if set(np.unique(y)) - {0, 1}:
            raise ValueError("SVM color classifier expects 0=non-yellow and 1=yellow labels.")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y,
        )

        classes = (
            self.label_encoder.classes_
            if self.label_encoder is not None
            else np.array(["non-yellow", "yellow"])
        )
        train_counts = {
            str(class_name): int((y_train == index).sum())
            for index, class_name in enumerate(classes)
        }
        test_counts = {
            str(class_name): int((y_test == index).sum())
            for index, class_name in enumerate(classes)
        }

        print("\n[SVM Yellow/Non-Yellow Training — One Feature]")
        print(f"  Feature: {self.feature_names[0]}")
        print(f"  Train: {len(X_train)} ({' '.join(f'{count} {name}' for name, count in train_counts.items())})")
        print(f"  Test:  {len(X_test)} ({' '.join(f'{count} {name}' for name, count in test_counts.items())})")

        if use_gridsearch:
            self.gridsearch(X_train, y_train)
        else:
            self.best_params = {
                "C": 1.0,
                "gamma": "scale",
                "kernel": "rbf",
            }
            self.model = self._pipeline()
            self.model.set_params(**self._params_with_prefix(self.best_params))
            with Timer("SVM model fit"):
                self.model.fit(X_train, y_train)

        cv_results = self.cross_validate(X, y)

        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)

        print("\n✓ Evaluation Results — Yellow is positive:")
        print(f"  Train Acc:  {train_accuracy:.2%}")
        print(f"  Test Acc:   {test_accuracy:.2%}")
        print(f"  Precision:  {test_precision:.2%}")
        print(f"  Recall:     {test_recall:.2%}")
        print(f"  F1 Score:   {test_f1:.2%}")

        self.visualize_confusion_matrix(y_test, y_test_pred)
        permutation_result = self.permutation_feature_importance(X_test, y_test)

        metrics = {
            "model_type": "svm",
            "binary_mode": True,
            "positive_class": "yellow",
            "negative_class": "non-yellow",
            "feature_names": self.feature_names,
            "dataset_size": len(X),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_class_counts": train_counts,
            "test_class_counts": test_counts,
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "precision_yellow": float(test_precision),
            "recall_yellow": float(test_recall),
            "f1_score_yellow": float(test_f1),
            "best_params": self.best_params,
            "cv_f1_macro_mean": float(cv_results["test_f1_macro"].mean()),
            "cv_f1_macro_std": float(cv_results["test_f1_macro"].std()),
            "permutation_importance": permutation_result,
        }
        TrainingLogger(self.repo_root, "svm").log_metrics(metrics)
        return self.model, metrics

    def visualize_confusion_matrix(self, y_true, y_pred):
        labels = (
            list(self.label_encoder.classes_)
            if self.label_encoder is not None
            else ["non-yellow", "yellow"]
        )
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        output_dir = self.repo_root / "figures" / "color" / "svm"
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(7, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title("Confusion Matrix - One-Feature SVM Yellow/Non-Yellow")
        plt.ylabel("True Color")
        plt.xlabel("Predicted Color")
        plt.tight_layout()
        plt.savefig(output_dir / "svm_confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("✓ Saved: figures/color/svm/svm_confusion_matrix.png")

    def permutation_feature_importance(self, X_test, y_test):
        result = permutation_importance(
            self.model,
            X_test,
            y_test,
            scoring="f1_macro",
            n_repeats=30,
            random_state=self.random_state,
            n_jobs=-1,
        )
        importance = float(result.importances_mean[0])
        importance_std = float(result.importances_std[0])

        output_dir = self.repo_root / "figures" / "color" / "svm"
        output_dir.mkdir(parents=True, exist_ok=True)
        chart_data = pd.DataFrame({
            "feature": self.feature_names,
            "importance": result.importances_mean,
        })

        plt.figure(figsize=(8, 4))
        sns.barplot(data=chart_data, x="importance", y="feature", color="gold")
        plt.title("Permutation Importance - One-Feature SVM")
        plt.xlabel("Macro-F1 Decrease After Permutation")
        plt.tight_layout()
        plt.savefig(output_dir / "svm_feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("✓ Saved: figures/color/svm/svm_feature_importance.png")

        return {
            self.feature_names[0]: importance,
            "std": importance_std,
        }

    def save(self, pkl_path):
        pkl_path = Path(pkl_path)
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with pkl_path.open("wb") as file:
            pickle.dump({
                "model": self.model,
                "best_params": self.best_params,
                "label_encoder": self.label_encoder,
                "binary_mode": True,
                "positive_class": "yellow",
                "negative_class": "non-yellow",
                "feature_names": self.feature_names,
            }, file)


def run_svm_pipeline(X, y, label_encoder, repo_root, use_gridsearch=True):
    detector = SVMColorClassifier(repo_root)
    detector.train(X, y, label_encoder, use_gridsearch=use_gridsearch)
    detector.save(
        Path(repo_root) / "models" / "color" / "svm" / "color_classifier_svm.pkl"
    )