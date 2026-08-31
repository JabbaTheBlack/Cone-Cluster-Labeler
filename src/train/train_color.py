#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import struct
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import time

sns.set(style='whitegrid', context='talk')


def find_project_root(start_path: Path | None = None) -> Path:
    """Walk upward until we find the repository root containing Dataset and src."""
    current = (start_path or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / "Dataset").exists() and (candidate / "src").exists():
            return candidate

    return current


REPO_ROOT = find_project_root()


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
            except:
                break
    return np.array(points, dtype=np.float32) if points else None

def extract_features(points):
    if points is None or len(points) < 3:
        return None

    xyz = points[:, :3]
    intensity = points[:, 3]
    z = xyz[:, 2]

    height = float(z.max() - z.min())
    width = max(xyz[:, 0].std(), xyz[:, 1].std(), 1e-6)
    aspect_ratio = height / width

    distance = np.linalg.norm(xyz.mean(axis=0))
    dist_sq = (distance ** 2) + 1e-6
    avg_i = intensity.mean()
    std_i = intensity.std()

    norm_avg_i = np.log1p(avg_i * dist_sq)
    norm_std_i = np.log1p(std_i * dist_sq)

    # percentile-based range: robust to single noisy returns
    i_lo, i_hi = np.percentile(intensity, [5, 95])
    contrast = (i_hi - i_lo) / (avg_i + 1e-6)
    reflective_point_pct = np.mean(intensity > (avg_i * 1.5))

    # three-band vertical intensity profile (bottom / mid / top)
    z_min, z_max = z.min(), z.max()
    z_range = max(z_max - z_min, 1e-6)
    bot_mask = z < (z_min + 0.33 * z_range)
    top_mask = z > (z_min + 0.67 * z_range)
    mid_mask = ~bot_mask & ~top_mask

    bot_i = intensity[bot_mask].mean() if bot_mask.sum() > 0 else avg_i
    mid_i = intensity[mid_mask].mean() if mid_mask.sum() > 0 else avg_i
    top_i = intensity[top_mask].mean() if top_mask.sum() > 0 else avg_i

    contrast_bot_mid = (mid_i - bot_i) / (avg_i + 1e-6)
    contrast_mid_top = (top_i - mid_i) / (avg_i + 1e-6)
    contrast_bot_top = (top_i - bot_i) / (avg_i + 1e-6)

    z_centered = z - z.mean()
    i_centered = intensity - avg_i
    v_grad = np.sum(z_centered * i_centered) / (np.sum(z_centered ** 2) + 1e-6)

    return np.array([
        norm_avg_i,
        norm_std_i,
        v_grad,
        height,
        aspect_ratio,
        contrast,
        reflective_point_pct,
        contrast_bot_mid,
        contrast_mid_top,
        contrast_bot_top,
    ], dtype=np.float32)

class MultiTrackDatasetBuilder:
    def __init__(self, base_dataset_path):
        self.root_path = Path(base_dataset_path).expanduser().resolve()

        if (self.root_path / "Processed" / "Color").exists():
            self.dataset_root = self.root_path
            self.label_base = self.root_path / "Processed" / "Color"
        elif self.root_path.name == "Color" and self.root_path.is_dir():
            self.dataset_root = self.root_path.parents[1]   # .../Dataset
            self.label_base = self.root_path
        else:
            raise FileNotFoundError(
                f"Could not locate Processed/Color from: {self.root_path}"
            )

        self.tracks = {}
        self._discover_tracks()
    
    def _discover_tracks(self):
        """Find all track folders with labeled_clusters.json files."""
        self.tracks = {}
        # Search recursively for the JSON files
        label_files = list(self.label_base.rglob('labeled_clusters.json'))
        
        for labels_path in label_files:
            track_folder = labels_path.parent
            # Get relative track name (e.g., Skidpad/fireup_...)
            track_name = str(track_folder.relative_to(self.label_base))
            
            with open(labels_path) as f:
                labels = json.load(f)
            
            # This ensures 'label_count' exists for main()
            self.tracks[track_name] = {
                'path': track_folder,
                'labels': labels,
                'label_count': len(labels)
            }
        
        if not self.tracks:
            print(f"❌ Error: No labeled_clusters.json found in {self.label_base}")

    def build_dataset(self):
        VALID_COLORS = {'orange', 'blue', 'yellow'}
        self.label_encoder = LabelEncoder()
        X, y_raw = [], []

        total_labels = 0
        skipped_unknown = 0
        skipped_invalid_color = 0
        skipped_missing_file = 0
        skipped_invalid_features = 0
        used_per_class = {'blue': 0, 'yellow': 0, 'orange': 0}

        print('\n📦 Building dataset from labeled clusters...')

        for track_name, track_info in self.tracks.items():
            labels = track_info['labels']
            track_path = track_info['path']

            for cluster_file, label_data in tqdm(labels.items(), desc=f'  {track_name}'):
                total_labels += 1
                color = str(label_data.get('color', '')).lower().strip()

                if color == 'unknown':
                    skipped_unknown += 1
                    continue

                if color not in VALID_COLORS:
                    skipped_invalid_color += 1
                    continue

                pcd_path = track_path / cluster_file

                if not pcd_path.exists():
                    pcd_path = self.dataset_root / "raw" / track_name / cluster_file

                if not pcd_path.exists():
                    found = list(self.dataset_root.rglob(cluster_file))
                    if found:
                        pcd_path = found[0]

                if not pcd_path.exists():
                    skipped_missing_file += 1
                    print(f"Missing file: track={track_name}, cluster={cluster_file}")
                    continue

                points = load_pcd_binary(pcd_path)
                features = extract_features(points)

                if features is None:
                    skipped_invalid_features += 1
                    continue

                X.append(features)
                y_raw.append(color)
                used_per_class[color] += 1

        X = np.array(X, dtype=np.float32)
        y = self.label_encoder.fit_transform(y_raw)

        total_skipped = (
            skipped_unknown +
            skipped_invalid_color +
            skipped_missing_file +
            skipped_invalid_features
        )

        print(f'\n✅ Dataset built: {len(X)} samples')
        print(f'   Total labels seen:      {total_labels}')
        print(f'   Total skipped:          {total_skipped}')
        print(f'   Skipped unknown:        {skipped_unknown}')
        print(f'   Skipped invalid color:  {skipped_invalid_color}')
        print(f'   Skipped missing file:   {skipped_missing_file}')
        print(f'   Skipped invalid feats:  {skipped_invalid_features}')
        print(f'   Used blue:              {used_per_class["blue"]}')
        print(f'   Used yellow:            {used_per_class["yellow"]}')
        print(f'   Used orange:            {used_per_class["orange"]}')

        return X, y, self.label_encoder

class RandomForestConeDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.best_params = None
        self.label_encoder = None  # Will be set during training
        self.feature_names = [
            'norm_avg_i',
            'norm_std_i',
            'v_grad',
            'height',
            'aspect_ratio',
            'contrast',
            'reflective_point_pct',
            'contrast_bot_mid',
            'contrast_mid_top',
            'contrast_bot_top',
        ]
    
    def cross_validate(self, X_scaled, y, cv_folds=5):
        """5-fold cross-validation on full dataset."""
        from sklearn.model_selection import cross_val_score, cross_validate
        print(f'\n🔍 {cv_folds}-Fold Cross-Validation...')
        
        rf_temp = RandomForestClassifier(**self.best_params, random_state=42, n_jobs=-1)
        cv_results = cross_validate(rf_temp, X_scaled, y, cv=cv_folds,
                                  scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
                                  return_train_score=True)
        
        print(f'  CV F1 Macro:  {cv_results["test_f1_macro"].mean():.4f} ± {cv_results["test_f1_macro"].std():.4f}')
        print(f'  CV Accuracy:  {cv_results["test_accuracy"].mean():.4f} ± {cv_results["test_accuracy"].std():.4f}')
        print(f'  CV Precision: {cv_results["test_precision_macro"].mean():.4f} ± {cv_results["test_precision_macro"].std():.4f}')
        print(f'  CV Recall:    {cv_results["test_recall_macro"].mean():.4f} ± {cv_results["test_recall_macro"].std():.4f}')
        print(f'  Train F1:     {cv_results["train_f1_macro"].mean():.4f} (overfitting check)')
        
        return cv_results

    def gridsearch(self, X_train, y_train):
        
        rf = RandomForestClassifier(random_state=42, class_weight='balanced_subsample')
    
        # Phase 1: COARSE (higher estimators + your full param space)
        print('🔍 Phase 1: Coarse search (100-400 estimators)...')
        coarse_grid = {
            'n_estimators': [20, 50, 100, 150],
            'max_depth': [5, 10, 15, 25, None],
            'min_samples_split': [10, 20, 30],
            'min_samples_leaf': [2, 5, 10, 20],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True]
        }
        
        coarse_search = GridSearchCV(rf, coarse_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
        coarse_search.fit(X_train, y_train)
        best_coarse = coarse_search.best_params_
        print(f'  Coarse best F1: {coarse_search.best_score_:.4f} → {best_coarse}')
        
        # Phase 2: MEDIUM around coarse winner (±30 range)
        print('🔍 Phase 2: Medium refinement...')
        n_est_start = max(10, best_coarse['n_estimators'] - 30)
        n_est_end = min(450, best_coarse['n_estimators'] + 31)
        med_grid = {
            'n_estimators': list(range(n_est_start, n_est_end, 10)),
            'max_depth': [best_coarse['max_depth']] if best_coarse['max_depth'] is not None else [None, 15, 25],
            'min_samples_split': [best_coarse['min_samples_split']],
            'min_samples_leaf': [best_coarse['min_samples_leaf']],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True]
        }
        
        med_search = GridSearchCV(rf, med_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
        med_search.fit(X_train, y_train)
        best_med = med_search.best_params_
        print(f'  Medium best F1: {med_search.best_score_:.4f} → {best_med}')
        
        # Phase 3: FINE around medium winner (±15 range)
        print('🔍 Phase 3: Fine tuning...')
        n_est_fine_start = max(10, best_med['n_estimators'] - 15)
        n_est_fine_end = min(450, best_med['n_estimators'] + 16)
        fine_grid = {
            'n_estimators': list(range(n_est_fine_start, n_est_fine_end, 5)),
            'max_depth': [None, 10, 15, 20, 25, 30] if best_med['max_depth'] is None else 
                        list(range(max(5, best_med['max_depth']-5), min(31, best_med['max_depth']+6))),
            'min_samples_split': [max(2, best_med['min_samples_split']-3), 
                                best_med['min_samples_split'], 
                                min(20, best_med['min_samples_split']+4)],
            'min_samples_leaf': [max(1, best_med['min_samples_leaf']-2), 
                                best_med['min_samples_leaf'], 
                                min(10, best_med['min_samples_leaf']+3)],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True]
        }
        
        fine_search = GridSearchCV(rf, fine_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
        fine_search.fit(X_train, y_train)
        
        print(f'\n✓ Progressive GridSearch Complete!')
        print(f'  Final Best F1: {fine_search.best_score_:.4f}')
        print(f'  Final Best Params: {fine_search.best_params_}')
        
        self.best_params = fine_search.best_params_
        self.model = fine_search.best_estimator_
        return self.model

    def train(self, X, y, use_gridsearch=True):
        """Train model."""
        # Split data training: 80% test: 20%
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f'\n[Random Forest Training]')
        print(f'  Train: {len(X_train)} | Test: {len(X_test)}')
        print('  Train class counts:', np.bincount(y_train))
        print('  Test class counts: ', np.bincount(y_test))
    
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Starting gridsearch...")
        self.gridsearch(X_train_scaled, y_train)

        X_full_scaled = self.scaler.transform(X)
        print("Cross validating...")
        cv_results = self.cross_validate(X_full_scaled, y)

        self.plot_feature_correlation(X)

        # Evaluate
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
        test_f1_score = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        
        print(f'\n✓ Evaluation Results:')
        print(f'  Train Acc: {train_accuracy:.2%}')
        print(f'  Test Acc:   {test_acc:.2%}')
        print(f'  Precision: {test_precision:.2%}')
        print(f'  Recall:    {test_recall:.2%}')
        print(f'  F1 Score:  {test_f1_score:.2%}')
        
        logSaver = LogSaver(log_dir='logs')
        logSaver.save(X_train_scaled, X_test_scaled, y_train, y_test,
                     train_accuracy, test_acc, test_precision, test_recall, test_f1_score,
                     self.feature_names, self.model.feature_importances_, self.best_params, cv_results)

        self.visualize_confusion_matrix(y_test, y_test_pred)
        self.visualize_feature_importances()
            
    def plot_feature_correlation(self, X):
        """Plot feature correlation matrix."""
        feat_df = pd.DataFrame(X, columns=self.feature_names)
        corr = feat_df.corr()
        
        plt.figure(figsize=(20, 18))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        out_dir = REPO_ROOT / 'figures' / 'color'
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✓ Saved: figures/detection/feature_correlation.png')


    def visualize_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        
        # Use label_encoder to get color names, fallback to numbers
        if self.label_encoder is not None:
            labels = list(self.label_encoder.classes_)
        else:
            labels = [str(i) for i in range(cm.shape[0])]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, 
                   yticklabels=labels)
        plt.title('Confusion Matrix - Cone Color Classification')
        plt.ylabel('True Color')
        plt.xlabel('Predicted Color')
        plt.tight_layout()

        out_dir = REPO_ROOT / 'figures' / 'color'
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✓ Saved: figures/color/confusion_matrix.png')
        

    def visualize_feature_importances(self):
        importances = self.model.feature_importances_
        feat_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feat_importance_df, x='importance', y='feature', hue='feature', palette='viridis', legend=False)
        plt.title('Feature Importances')
        plt.xlabel('Importance Score')
        plt.tight_layout()

        out_dir = REPO_ROOT / 'figures' / 'color'
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'feature_importances.png', dpi=300, bbox_inches='tight')
        print('✓ Saved: figures/color/feature_importances.png')

    def save(self, path='cone_detector_rf.pkl'):
        """Save model."""
        data = {
            'scaler': self.scaler,
            'model': self.model,
            'best_params': self.best_params
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f'\n✓ Saved to {path}')
 
    def save_cpp_ready(self, path='models/color_classifier_rf.bin'):
        """Save multiclass model in raw binary format for C++"""
        import struct
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        scaler_mean = self.scaler.mean_.astype(np.float32)
        scaler_std = self.scaler.scale_.astype(np.float32)
        n_features = len(scaler_mean)
        n_classes = len(self.label_encoder.classes_)
        
        print(f'Saving {n_features}-feature, {n_classes}-class model to {path}')
        
        with open(path, 'wb') as f:
            # 1. HEADER: features (int), classes (int)
            f.write(struct.pack('ii', n_features, n_classes))
            
            # 2. SCALER: mean and std arrays
            f.write(scaler_mean.tobytes())
            f.write(scaler_std.tobytes())
            
            # 3. FOREST: number of trees (int)
            f.write(struct.pack('i', self.model.n_estimators))
            
            # 4. TREES
            for tree_obj in self.model.estimators_:
                tree = tree_obj.tree_
                # Node count for this specific tree
                f.write(struct.pack('i', tree.node_count))
                
                for i in range(tree.node_count):
                    f.write(struct.pack('i', int(tree.feature[i])))
                    f.write(struct.pack('f', float(tree.threshold[i])))
                    f.write(struct.pack('i', int(tree.children_left[i])))
                    f.write(struct.pack('i', int(tree.children_right[i])))

                    # probabilities for this node
                    node_values = tree.value[i][0].astype(np.float32)
                    probabilities = node_values / (np.sum(node_values) + 1e-6)
                    f.write(probabilities.tobytes())

            
        print(f'✓ C++ Ready ({n_features} feats, {n_classes} classes): {path}')

    def load(self, path='cone_detector_rf.pkl'):
        """Load model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.scaler = data['scaler']
        self.model = data['model']
        self.best_params = data.get('best_params')
        print(f'✓ Loaded from {path}')
    
    def predict(self, X):
        """Predict on new data."""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        return predictions, probabilities

# ============================================================================
# Log Saver
# ============================================================================
class LogSaver:
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / 'color' / 'training_log.txt'
    
    def save(self, X_train, X_test, y_train, y_test, train_acc, test_acc, precision, recall, f1, 
         features, importances, best_params=None, cv_results=None):
        results = {
            'dataset_size': len(X_train) + len(X_test),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'best_params': best_params,
            'cv_f1_mean': float(cv_results['test_f1_macro'].mean()) if cv_results is not None else None,
            'cv_f1_std': float(cv_results['test_f1_macro'].std()) if cv_results is not None else None,
            'feature_importances': dict(zip(features, importances.tolist()))
        }
        
        with open(self.log_file, 'a') as f:
            json.dump(results, f)
            f.write('\n\n')
        
        print(f'Training log saved: {self.log_file}')
        return self.log_file


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Random Forest for cone color classification")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to Dataset folder (the one containing Processed/Color)"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path
    dataset_path = dataset_path.resolve()
    print(f"\n🚀 Dataset root: {dataset_path}\n")

    builder = MultiTrackDatasetBuilder(dataset_path)

    print(f"\n✅ Tracks found: {len(builder.tracks)}\n")
    for name, data in builder.tracks.items():
        print(f"  {name}: {data['label_count']} labels")
    
    # Build dataset
    X, y, label_encoder = builder.build_dataset()
    
    if len(X) == 0:
        print("❌ No valid samples!")
        return
    
    # Train
    detector = RandomForestConeDetector()
    detector.label_encoder = label_encoder  # Set for confusion matrix labels

    start_time = time.perf_counter()

    detector.train(X, y)
    detector.save(str(REPO_ROOT / 'models' / 'color' / 'color_classifier_rf.pkl'))
    detector.save_cpp_ready(str(REPO_ROOT / 'models' / 'color' / 'color_classifier.bin'))

    elapsed_time = time.perf_counter() - start_time
    print(f"\n⏱️  Total training time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()