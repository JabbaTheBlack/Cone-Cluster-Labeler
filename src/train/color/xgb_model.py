import json
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
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
        xgb_temp = XGBClassifier(
            **self.best_params,
            objective='multi:softprob',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        cv_results = cross_validate(
            xgb_temp, X_scaled, y, cv=cv_folds,
            scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
            return_train_score=True, n_jobs=-1
        )

        print(f'  CV F1 Macro:  {cv_results["test_f1_macro"].mean():.4f} ± {cv_results["test_f1_macro"].std():.4f}')
        print(f'  CV Accuracy:  {cv_results["test_accuracy"].mean():.4f} ± {cv_results["test_accuracy"].std():.4f}')
        print(f'  CV Precision: {cv_results["test_precision_macro"].mean():.4f} ± {cv_results["test_precision_macro"].std():.4f}')
        print(f'  CV Recall:    {cv_results["test_recall_macro"].mean():.4f} ± {cv_results["test_recall_macro"].std():.4f}')
        print(f'  Train F1:     {cv_results["train_f1_macro"].mean():.4f} (overfitting check)')

        return cv_results

    def gridsearch(self, X_train, y_train):
        xgb_base = XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=self.random_state,
            n_jobs=-1
        )

        # Phase 1: Coarse Search
        print('\n🔍 Phase 1: Coarse search...')
        coarse_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 1.0]
        }

        with Timer('XGBoost GridSearch Phase 1'):
            coarse_search = GridSearchCV(xgb_base, coarse_grid, cv=5, scoring='f1_macro', n_jobs=-1)
            coarse_search.fit(X_train, y_train)

        best_coarse = coarse_search.best_params_
        print(f'  Coarse best F1 Macro: {coarse_search.best_score_:.4f} → {best_coarse}')

        # Phase 2: Medium Refinement
        print('🔍 Phase 2: Medium refinement...')
        n_est_start = max(20, best_coarse['n_estimators'] - 40)
        n_est_end = min(450, best_coarse['n_estimators'] + 41)
        med_grid = {
            'n_estimators': list(range(n_est_start, n_est_end, 20)),
            'max_depth': sorted(list(set([max(2, best_coarse['max_depth'] - 1), best_coarse['max_depth'], best_coarse['max_depth'] + 1]))),
            'learning_rate': sorted(list(set([max(0.01, best_coarse['learning_rate'] - 0.02), best_coarse['learning_rate'], best_coarse['learning_rate'] + 0.02]))),
            'subsample': [best_coarse['subsample']],
            'colsample_bytree': [0.7, 0.8, 1.0]
        }

        with Timer('XGBoost GridSearch Phase 2'):
            med_search = GridSearchCV(xgb_base, med_grid, cv=5, scoring='f1_macro', n_jobs=-1)
            med_search.fit(X_train, y_train)

        best_med = med_search.best_params_
        print(f'  Medium best F1 Macro: {med_search.best_score_:.4f} → {best_med}')

        # Phase 3: Fine Tuning
        print('🔍 Phase 3: Fine tuning...')
        n_est_fine_start = max(10, best_med['n_estimators'] - 15)
        n_est_fine_end = min(450, best_med['n_estimators'] + 16)
        fine_grid = {
            'n_estimators': list(range(n_est_fine_start, n_est_fine_end, 5)),
            'max_depth': [best_med['max_depth']],
            'learning_rate': [best_med['learning_rate']],
            'subsample': sorted(list(set([max(0.5, best_med['subsample'] - 0.1), best_med['subsample'], min(1.0, best_med['subsample'] + 0.1)]))),
            'colsample_bytree': sorted(list(set([max(0.5, best_med['colsample_bytree'] - 0.1), best_med['colsample_bytree'], min(1.0, best_med['colsample_bytree'] + 0.1)])))
        }

        with Timer('XGBoost GridSearch Phase 3'):
            fine_search = GridSearchCV(xgb_base, fine_grid, cv=5, scoring='f1_macro', n_jobs=-1)
            fine_search.fit(X_train, y_train)

        print(f'\n✓ Progressive GridSearch Complete!')
        print(f'  Final Best F1 Macro: {fine_search.best_score_:.4f}')
        print(f'  Final Best Params:   {fine_search.best_params_}')

        self.best_params = fine_search.best_params_
        self.model = fine_search.best_estimator_
        return self.model

    def train(self, X, y, label_encoder, use_gridsearch=True):
        self.label_encoder = label_encoder
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
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
            self.model = XGBClassifier(
                **self.best_params,
                objective='multi:softprob',
                eval_metric='mlogloss',
                random_state=self.random_state,
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

def run_xgb_pipeline(X, y, label_encoder, repo_root, use_gridsearch=True):
    detector = XGBoostConeDetector(repo_root)
    detector.train(X, y, label_encoder, use_gridsearch=use_gridsearch)
    detector.save(repo_root / 'models' / 'color' / 'xgboost' / 'color_classifier_xgb.pkl')