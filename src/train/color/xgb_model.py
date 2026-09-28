import json
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from math import prod
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.utils.class_weight import compute_sample_weight
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
        self.log_dir = Path(repo_root) / "logs" / "color" / model_subfolder
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training_log.txt"


    def log_metrics(self, metrics: dict):
        with open(self.log_file, "a") as f:
            f.write("\n")
            f.write(json.dumps(metrics))


class XGBoostConeDetector:
    def __init__(self, repo_root, random_state=42):
        self.repo_root = Path(repo_root)
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.best_params = None
        self.label_encoder = None
        self.feature_names = FEATURE_NAMES


    def cross_validate(self, X_scaled, y, cv_folds=5):
        print(f'\n🔍 {cv_folds}-Fold Cross-Validation (Macro F1 scoring)...')
        objective = 'binary:logistic' if len(np.unique(y)) == 2 else 'multi:softprob'
        xgb_temp = XGBClassifier(
            **self.best_params,
            objective=objective,
            random_state=self.random_state,
            n_jobs=-1
        )


        sample_weights = compute_sample_weight('balanced', y);
        cv_results = cross_validate(
            xgb_temp, X_scaled, y, cv=cv_folds,
            scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
            return_train_score=True, n_jobs=-1, params={'sample_weight': sample_weights}
        )


        print(f'  CV F1 Macro:  {cv_results["test_f1_macro"].mean():.4f} ± {cv_results["test_f1_macro"].std():.4f}')
        print(f'  CV Accuracy:  {cv_results["test_accuracy"].mean():.4f} ± {cv_results["test_accuracy"].std():.4f}')
        print(f'  CV Precision: {cv_results["test_precision_macro"].mean():.4f} ± {cv_results["test_precision_macro"].std():.4f}')
        print(f'  CV Recall:    {cv_results["test_recall_macro"].mean():.4f} ± {cv_results["test_recall_macro"].std():.4f}')
        print(f'  Train F1:     {cv_results["train_f1_macro"].mean():.4f} (overfitting check)')


        return cv_results


    def gridsearch(self, X_train, y_train):
        is_binary = len(np.unique(y_train)) == 2

        objective = (
            "binary:logistic"
            if is_binary
            else "multi:softprob"
        )

        eval_metric = (
            "logloss"
            if is_binary
            else "mlogloss"
        )

        xgb_base = XGBClassifier(
            objective=objective,
            eval_metric=eval_metric,
            random_state=self.random_state,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )

        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        global_best_score = -np.inf
        global_best_params = None
        global_best_estimator = None
        global_best_phase = None

        def evaluate_phase(search, phase_name, grid):
            nonlocal global_best_score
            nonlocal global_best_params
            nonlocal global_best_estimator
            nonlocal global_best_phase

            candidates = prod(len(values) for values in grid.values())

            print(
                f"\n🔍 {phase_name}... "
                f"({candidates} candidates, {candidates * 5} total fits)"
            )

            with Timer(f"XGBoost {phase_name}"):
                search.fit(
                    X_train,
                    y_train,
                    sample_weight=sample_weights,
                )

            score = float(search.best_score_)

            print(
                f"  {phase_name} best F1 Macro: "
                f"{score:.4f} → {search.best_params_}"
            )

            if score > global_best_score:
                global_best_score = score
                global_best_params = search.best_params_.copy()
                global_best_estimator = search.best_estimator_
                global_best_phase = phase_name

                print(
                    f"  ✓ New global best: "
                    f"{global_best_score:.4f}"
                )
            else:
                print(
                    f"  ↳ Global best remains: "
                    f"{global_best_score:.4f} "
                    f"from {global_best_phase}"
                )

        coarse_grid = {
            "n_estimators": [200, 300, 500, 700],
            "max_depth": [2, 3, 6, 9],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_weight": [1, 3, 5],
            "gamma": [0.0, 0.1],
        }

        coarse_search = GridSearchCV(
            estimator=xgb_base,
            param_grid=coarse_grid,
            cv=5,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
            return_train_score=True,
            verbose=1,
        )

        evaluate_phase(
            coarse_search,
            "Phase 1: Coarse search",
            coarse_grid,
        )

        coarse = coarse_search.best_params_

        medium_grid = {
            "n_estimators": sorted({
                max(50, coarse["n_estimators"] - 100),
                max(50, coarse["n_estimators"] - 50),
                coarse["n_estimators"],
                coarse["n_estimators"] + 50,
                coarse["n_estimators"] + 100,
            }),
            "max_depth": sorted({
                max(1, coarse["max_depth"] - 2),
                max(1, coarse["max_depth"] - 1),
                coarse["max_depth"],
                coarse["max_depth"] + 1,
                coarse["max_depth"] + 2,
            }),
            "learning_rate": sorted({
                max(0.005, coarse["learning_rate"] * 0.5),
                max(0.005, coarse["learning_rate"] * 0.75),
                coarse["learning_rate"],
                coarse["learning_rate"] * 1.25,
                coarse["learning_rate"] * 1.5,
            }),
            "subsample": sorted({
                max(0.5, coarse["subsample"] - 0.1),
                coarse["subsample"],
                min(1.0, coarse["subsample"] + 0.1),
            }),
            "colsample_bytree": sorted({
                max(0.5, coarse["colsample_bytree"] - 0.1),
                coarse["colsample_bytree"],
                min(1.0, coarse["colsample_bytree"] + 0.1),
            }),
            "min_child_weight": sorted({
                max(1, coarse["min_child_weight"] - 1),
                coarse["min_child_weight"],
                coarse["min_child_weight"] + 1,
            }),
            "gamma": sorted({
                max(0.0, coarse["gamma"] - 0.05),
                coarse["gamma"],
                coarse["gamma"] + 0.05,
            }),
        }

        medium_search = GridSearchCV(
            estimator=xgb_base,
            param_grid=medium_grid,
            cv=5,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
            return_train_score=True,
            verbose=1,
        )

        evaluate_phase(
            medium_search,
            "Phase 2: Medium refinement",
            medium_grid,
        )

        medium = medium_search.best_params_

        fine_grid = {
            "n_estimators": sorted({
                max(50, medium["n_estimators"] - 30),
                max(50, medium["n_estimators"] - 15),
                medium["n_estimators"],
                medium["n_estimators"] + 15,
                medium["n_estimators"] + 30,
            }),
            "max_depth": sorted({
                max(1, medium["max_depth"] - 1),
                medium["max_depth"],
                medium["max_depth"] + 1,
            }),
            "learning_rate": sorted({
                max(0.001, medium["learning_rate"] * 0.8),
                medium["learning_rate"],
                medium["learning_rate"] * 1.2,
            }),
            "subsample": sorted({
                max(0.5, medium["subsample"] - 0.05),
                medium["subsample"],
                min(1.0, medium["subsample"] + 0.05),
            }),
            "colsample_bytree": sorted({
                max(0.5, medium["colsample_bytree"] - 0.05),
                medium["colsample_bytree"],
                min(1.0, medium["colsample_bytree"] + 0.05),
            }),
            "min_child_weight": sorted({
                max(1, medium["min_child_weight"] - 1),
                medium["min_child_weight"],
                medium["min_child_weight"] + 1,
            }),
            "gamma": sorted({
                max(0.0, medium["gamma"] - 0.02),
                medium["gamma"],
                medium["gamma"] + 0.02,
            }),
        }

        fine_search = GridSearchCV(
            estimator=xgb_base,
            param_grid=fine_grid,
            cv=5,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
            return_train_score=True,
            verbose=1,
        )

        evaluate_phase(
            fine_search,
            "Phase 3: Fine tuning",
            fine_grid,
        )

        if global_best_estimator is None:
            raise RuntimeError(
                "Progressive XGBoost search produced no model."
            )

        self.best_params = global_best_params
        self.model = global_best_estimator
        self.grid_search_best_score = global_best_score
        self.grid_search_winning_phase = global_best_phase

        print("\n✓ Progressive XGBoost GridSearch Complete!")
        print(f"  Winning phase:   {global_best_phase}")
        print(f"  Best F1 Macro:   {global_best_score:.4f}")
        print(f"  Best parameters: {self.best_params}")

        return self.model


    def train(
        self,
        X,
        y,
        label_encoder,
        use_gridsearch=True,
        train_idx=None,
        test_idx=None,
    ):
        self.label_encoder = label_encoder


        if train_idx is None or test_idx is None:
            raise ValueError(
                "XGBoost requires precomputed leakage-safe "
                "group-disjoint train_idx and test_idx."
            )


        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)


        if np.intersect1d(train_idx, test_idx).size != 0:
            raise RuntimeError("Train/test sample-index overlap detected.")


        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError("Train/test indices must both be non-empty.")


        if train_idx.min() < 0 or test_idx.min() < 0:
            raise ValueError("Train/test indices must be non-negative.")


        if train_idx.max() >= len(X) or test_idx.max() >= len(X):
            raise ValueError("Train/test indices exceed dataset size.")


        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]


        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            raise ValueError(
                "Grouped split must contain both classes in train and test."
            )


        classes = self.label_encoder.classes_
        train_counts = {str(cls): int((y_train == i).sum()) for i, cls in enumerate(classes)}
        test_counts = {str(cls): int((y_test == i).sum()) for i, cls in enumerate(classes)}


        train_str = " ".join([f"{count} {cls}" for cls, count in train_counts.items()])
        test_str = " ".join([f"{count} {cls}" for cls, count in test_counts.items()])


        print(f'\n[XGBoost Training]')
        print(f'  Train: {len(X_train)} ({train_str})')
        print(f'  Test:  {len(X_test)} ({test_str})')


        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)


        if use_gridsearch:
            self.gridsearch(X_train_scaled, y_train)
        else:
            self.best_params = {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
            objective = 'binary:logistic' if len(np.unique(y_train)) == 2 else 'multi:softprob'
            self.model = XGBClassifier(
                **self.best_params,
                objective=objective,
                eval_metric='logloss' if len(np.unique(y_train)) == 2 else 'mlogloss',
                random_state=self.random_state,
                n_jobs=-1
            )
            with Timer('Model fit'):
                sample_weights_train = compute_sample_weight('balanced', y_train)
                self.model.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)


        X_full_scaled = self.scaler.transform(X)


        with Timer('Cross-validation'):
            cv_results = self.cross_validate(X_full_scaled, y)


        self.plot_feature_correlation(X)


        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)


        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
        test_f1_score = f1_score(y_test, y_test_pred, average='macro', zero_division=0)


        print(f'\n✓ Evaluation Results:')
        print(f'  Train Acc:  {train_accuracy:.2%}')
        print(f'  Test Acc:   {test_acc:.2%}')
        print(f'  Precision:  {test_precision:.2%}')
        print(f'  Recall:     {test_recall:.2%}')
        print(f'  F1 Score:   {test_f1_score:.2%}')


        self.visualize_confusion_matrix(y_test, y_test_pred)
        self.visualize_feature_importances()


        feature_importances = {
            feat: float(imp)
            for feat, imp in zip(self.feature_names, self.model.feature_importances_)
        }


        log_metrics = {
            "dataset_size": len(X),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_class_counts": train_counts, 
            "test_class_counts": test_counts, 
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_acc),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "f1_score": float(test_f1_score),
            "best_params": self.best_params,
            "cv_f1_mean": float(cv_results["test_f1_macro"].mean()),
            "cv_f1_std": float(cv_results["test_f1_macro"].std()),
            "feature_importances": feature_importances
        }


        logger = TrainingLogger(self.repo_root, "xgboost")
        logger.log_metrics(log_metrics)


    def plot_feature_correlation(self, X):
        feat_df = pd.DataFrame(X, columns=self.feature_names)
        corr = feat_df.corr()


        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('XGBoost - Feature Correlation Matrix')
        plt.tight_layout()


        out_dir = self.repo_root / 'figures' / 'color' / 'xgboost'
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'xgb_feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✓ Saved: figures/color/xgboost/xgb_feature_correlation.png')


    def visualize_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        labels = list(self.label_encoder.classes_) if self.label_encoder else [str(i) for i in range(cm.shape[0])]


        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            annot_kws={"size": 14}
        )
        plt.title('Confusion Matrix - XGBoost', fontsize=18, pad=12)
        plt.ylabel('True Color', fontsize=14)
        plt.xlabel('Predicted Color', fontsize=14)
        plt.tight_layout()


        out_dir = self.repo_root / 'figures' / 'color' / 'xgboost'
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'xgb_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✓ Saved: figures/color/xgboost/xgb_confusion_matrix.png')


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
        plt.title('Feature Importances - XGBoost')
        plt.xlabel('Importance Score')
        plt.tight_layout()


        out_dir = self.repo_root / 'figures' / 'color' / 'xgboost'
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'xgb_feature_importances.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✓ Saved: figures/color/xgboost/xgb_feature_importances.png')


    def save(self, pkl_path):
        Path(pkl_path).parent.mkdir(parents=True, exist_ok=True)
        with open(pkl_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'model': self.model, 'best_params': self.best_params}, f)


def run_xgb_pipeline(
    X,
    y,
    label_encoder,
    repo_root,
    use_gridsearch=True,
    train_idx=None,
    test_idx=None,
):
    detector = XGBoostConeDetector(repo_root)
    detector.train(
        X,
        y,
        label_encoder,
        use_gridsearch=use_gridsearch,
        train_idx=train_idx,
        test_idx=test_idx,
    )
    detector.save(repo_root / 'models' / 'color' / 'xgboost' / 'color_classifier_xgb.pkl')