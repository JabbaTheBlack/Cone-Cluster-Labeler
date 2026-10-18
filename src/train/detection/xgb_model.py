#!/usr/bin/env python3
import json
import time
import pickle
from math import prod
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV, cross_validate
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
    def __init__(self, repo_root, random_state=42, feature_names=None, output_name="xgboost"):
        self.repo_root = Path(repo_root)
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.best_params = None
        self.feature_names = list(feature_names) if feature_names is not None else FEATURE_NAMES
        # Subfolder under logs/detection and figures/detection
        self.output_name = output_name



    def cross_validate(self, X_scaled, y, groups, cv_folds=5):
        effective_folds = min(cv_folds, len(np.unique(groups)))
        print(f'\n🔍 {effective_folds}-Fold Cross-Validation (F1 scoring)...')
        scale_pos_weight = (len(y) - sum(y)) / max(sum(y), 1)



        xgb_temp = xgb.XGBClassifier(
            **self.best_params,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )



        cv_results = cross_validate(
            xgb_temp, X_scaled, y, groups=groups, cv=GroupKFold(n_splits=effective_folds),
            scoring=['accuracy', 'precision', 'recall', 'f1'],
            return_train_score=True, n_jobs=-1
        )



        print(f'  CV F1:        {cv_results["test_f1"].mean():.4f} ± {cv_results["test_f1"].std():.4f}')
        print(f'  CV Accuracy:  {cv_results["test_accuracy"].mean():.4f} ± {cv_results["test_accuracy"].std():.4f}')
        print(f'  CV Precision: {cv_results["test_precision"].mean():.4f} ± {cv_results["test_precision"].std():.4f}')
        print(f'  CV Recall:    {cv_results["test_recall"].mean():.4f} ± {cv_results["test_recall"].std():.4f}')
        print(f'  Train F1:     {cv_results["train_f1"].mean():.4f} (overfitting check)')



        return cv_results



    def gridsearch(self, X_train, y_train, groups_train):
        scale_pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
        n_folds = min(5, len(np.unique(groups_train)))
        cv = GroupKFold(n_splits=n_folds)



        # n_jobs=1 per model: GridSearchCV already runs the fits in parallel
        xgb_clf = xgb.XGBClassifier(
            random_state=self.random_state,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            n_jobs=1,
            tree_method='hist'
        )



        global_best = {'score': -np.inf, 'params': None, 'estimator': None, 'phase': None}



        def run_phase(phase_name, grid, verbose):
            n_combinations = prod(len(values) for values in grid.values())
            print(f'\n🔍 {phase_name}... ({n_combinations} candidates, {n_combinations * n_folds} total fits)')
            search = GridSearchCV(
                xgb_clf,
                grid,
                cv=cv,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=verbose,
            )
            with Timer(f'XGBoost GridSearch {phase_name}'):
                search.fit(X_train, y_train, groups=groups_train)



            print(f'  {phase_name} best F1 Macro: {search.best_score_:.4f} → {search.best_params_}')
            if search.best_score_ > global_best['score']:
                global_best.update(
                    score=float(search.best_score_),
                    params=search.best_params_.copy(),
                    estimator=search.best_estimator_,
                    phase=phase_name,
                )
                print(f'  ✓ New global best from {phase_name}: {global_best["score"]:.4f}')
            else:
                print(f'  ↳ Global best remains: {global_best["score"]:.4f} from {global_best["phase"]}')
            return search.best_params_



        def around_zero_or_scale(value, zero_steps):
            # Neighbours of a regularisation value; 0 has no multiplicative neighbours
            return zero_steps if value == 0 else sorted({value / 2, value, value * 2})



        # ==========================================
        # Phase 1: Coarse Search
        # ==========================================
        coarse_grid = {
            'n_estimators': [100, 200, 400],
            'max_depth': [2, 3, 4, 6, 8],
            'learning_rate': [0.03, 0.1, 0.3],
            'min_child_weight': [1, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0.0, 0.5],
            'reg_alpha': [0.0, 1.0],
            'reg_lambda': [1.0, 5.0],
        }
        c = run_phase('Phase 1: Coarse search', coarse_grid, verbose=1)



        # ==========================================
        # Phase 2: Medium Refinement
        # ==========================================
        medium_grid = {
            'n_estimators': sorted({max(50, c['n_estimators'] - 100), c['n_estimators'], c['n_estimators'] + 100}),
            'max_depth': sorted({max(2, c['max_depth'] - 1), c['max_depth'], min(10, c['max_depth'] + 1)}),
            'learning_rate': sorted({round(c['learning_rate'] * 0.5, 4), c['learning_rate'], round(min(0.5, c['learning_rate'] * 1.5), 4)}),
            'min_child_weight': sorted({max(1, c['min_child_weight'] - 2), c['min_child_weight'], c['min_child_weight'] + 2}),
            'subsample': [c['subsample']],
            'colsample_bytree': [c['colsample_bytree']],
            'gamma': around_zero_or_scale(c['gamma'], [0.0, 0.1, 0.25]),
            'reg_alpha': around_zero_or_scale(c['reg_alpha'], [0.0, 0.1, 0.5]),
            'reg_lambda': around_zero_or_scale(c['reg_lambda'], [0.0, 0.5, 1.0]),
        }
        m = run_phase('Phase 2: Medium refinement', medium_grid, verbose=1)



        # ==========================================
        # Phase 3: Fine Tuning
        # ==========================================
        fine_grid = {
            'n_estimators': sorted({max(50, m['n_estimators'] - 50), m['n_estimators'], m['n_estimators'] + 50}),
            'max_depth': [m['max_depth']],
            'learning_rate': sorted({round(m['learning_rate'] * 0.8, 4), m['learning_rate'], round(m['learning_rate'] * 1.2, 4)}),
            'min_child_weight': [m['min_child_weight']],
            'subsample': sorted({max(0.6, round(m['subsample'] - 0.1, 2)), m['subsample'], min(1.0, round(m['subsample'] + 0.1, 2))}),
            'colsample_bytree': sorted({max(0.6, round(m['colsample_bytree'] - 0.1, 2)), m['colsample_bytree'], min(1.0, round(m['colsample_bytree'] + 0.1, 2))}),
            'gamma': [m['gamma']],
            'reg_alpha': [m['reg_alpha']],
            'reg_lambda': [m['reg_lambda']],
        }
        run_phase('Phase 3: Fine tuning', fine_grid, verbose=1)



        print('\n✓ Progressive XGBoost GridSearch Complete!')
        print(f'  Winning phase:   {global_best["phase"]}')
        print(f'  Best F1 Macro:   {global_best["score"]:.4f}')
        print(f'  Best parameters: {global_best["params"]}')



        self.best_params = global_best['params']
        self.best_cv_score = global_best['score']
        self.model = global_best['estimator']
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



        print(f'\n[XGBoost Detection Training]')
        print(f'  Train: {len(X_train)} (Cones: {int(y_train.sum())}, Non-cones: {int(len(y_train) - y_train.sum())})')
        print(f'  Test:  {len(X_test)} (Cones: {int(y_test.sum())}, Non-cones: {int(len(y_test) - y_test.sum())})')



        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)



        if use_gridsearch:
            self.gridsearch(X_train_scaled, y_train, groups_train)
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
            cv_results = self.cross_validate(X_full_scaled, y, groups)



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
            "feature_importances": {f: float(i) for f, i in zip(self.feature_names, self.model.feature_importances_)},
            "feature_names": self.feature_names,
            "gridsearch_cv_f1_macro": getattr(self, "best_cv_score", None),
            "f1_macro": float(f1_score(y_test, y_test_pred, average='macro', zero_division=0)),
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
        plt.title('Feature Correlation Matrix - XGBoost', fontsize=14)
        plt.tight_layout()



        out_dir = self.repo_root / 'figures' / 'detection' / self.output_name
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



        out_dir = self.repo_root / 'figures' / 'detection' / self.output_name
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



        out_dir = self.repo_root / 'figures' / 'detection' / self.output_name
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'xgb_feature_importances.png', dpi=300, bbox_inches='tight')
        plt.close()



    def save(self, pkl_path, json_path):
        Path(pkl_path).parent.mkdir(parents=True, exist_ok=True)
        with open(pkl_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'model': self.model, 'best_params': self.best_params}, f)



        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(json_path))



def run_xgb_pipeline(
    X,
    y,
    groups,
    repo_root,
    use_gridsearch=True,
    train_idx=None,
    test_idx=None,
    feature_names=None,
    output_name="xgboost",
):
    detector = XGBoostConeDetector(repo_root, feature_names=feature_names, output_name=output_name)
    detector.train(
        X,
        y,
        groups,
        use_gridsearch=use_gridsearch,
        train_idx=train_idx,
        test_idx=test_idx,
    )
    detector.save(
        repo_root / 'models' / 'detection' / output_name / 'cone_detector_xgb.pkl',
        repo_root / 'models' / 'detection' / output_name / 'cone_detector_xgb.json'
    )