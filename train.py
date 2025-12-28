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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

sns.set(style='whitegrid', context='talk')
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
            except:
                break
    return np.array(points, dtype=np.float32) if points else None

def extract_features(cluster_points):

    xyz = cluster_points[:, :3]
    intensity = cluster_points[:, 3]
    
    height = float(xyz[:, 2].max() - xyz[:, 2].min()) # Height of the cluster
    width = float(xyz[:, 0].max() - xyz[:, 0].min()) # Width of the cluster
    depth = float(xyz[:, 1].max() - xyz[:, 1].min()) # Depth of the cluster
    aspect_ratio = height / (width + 1e-6) # Height to width ratio
    
    num_points = len(xyz) # Number of points in the cluster
    # Bounding box volume height * width * depth -> box oppossed to a convex hull
    volume = (height + 1e-6) * (width + 1e-6) * (depth + 1e-6) 
    density = num_points / volume # Point per volume cones are not as dense
    
    center = xyz.mean(axis=0)
    distances = np.linalg.norm(xyz - center, axis=1) # Average distance from the centroid
    compactness = np.std(distances) / (np.mean(distances) + 1e-6) # Std deviation of distances
    
    # Average intensity of the cluster. Average of the min and max for each frame, oppossed to cap from 1 to 255 -> will work with different LiDARs
    intensity_mean = float(intensity.mean()) 
    intensity_std = float(intensity.std())
    
    cov = np.cov(xyz.T) # Covariance matrix of the points
    eigenvalues = np.linalg.eigvalsh(cov) 
    linearity = (eigenvalues[2] - eigenvalues[1]) / (eigenvalues[2] + 1e-6) # Cones are not a line
    planarity = (eigenvalues[1] - eigenvalues[0]) / (eigenvalues[2] + 1e-6) # Cones re not a 2D plane
    
    return np.array([
        height, width, depth, aspect_ratio, num_points,
        density, compactness, intensity_mean, intensity_std,
        linearity, planarity, volume
    ], dtype=np.float32)

# ============================================================================
# DATASET BUILDER
# ============================================================================

class MultiTrackDatasetBuilder:
    """Combines labeled clusters from multiple track folders into one dataset."""
    
    def __init__(self, base_dataset_path):
        """
        Args:
            base_dataset_path: Path to Dataset folder (contains Acceleration/, Skidpad/, Autocross/)
        """
        self.base_path = Path(base_dataset_path).expanduser()
        self.tracks = {}
        self.X = []
        self.y = []
        self.track_stats = {}
        
        # Discover all track folders with labeled_clusters.json
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
                X_track.append(features)
                y_track.append(1 if label['is_cone'] else 0)
                pbar.update(1)
            
            X_track = np.array(X_track, dtype=np.float32)
            y_track = np.array(y_track, dtype=np.int64)
            
            # Store track stats
            self.track_stats[track_name] = {
                'samples': len(X_track),
                'cones': int(np.sum(y_track)),
                'non_cones': int(len(y_track) - np.sum(y_track)),
                'missing': missing
            }
            
            self.X.extend(X_track)
            self.y.extend(y_track)
        
        pbar.close()
        
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)
        
        # Print statistics
        self._print_stats()
        
        return self.X, self.y
    
    def _print_stats(self):
        """Print dataset statistics."""
        print('\n' + '='*70)
        print('📈 DATASET STATISTICS')
        print('='*70)
        
        for track_name, stats in self.track_stats.items():
            print(f'\n{track_name}:')
            print(f'  Samples: {stats["samples"]}')
            print(f'  Cones: {stats["cones"]} ({100*stats["cones"]/(stats["samples"]+1e-6):.1f}%)')
            print(f'  Non-cones: {stats["non_cones"]} ({100*stats["non_cones"]/(stats["samples"]+1e-6):.1f}%)')
            if stats["missing"] > 0:
                print(f'  ⚠️  Missing PCDs: {stats["missing"]}')
        
        print(f'\n{"-"*70}')
        print(f'✓ COMBINED DATASET: {len(self.X)} total samples')
        print(f'  Cones: {sum(self.y)} ({100*sum(self.y)/len(self.y):.1f}%)')
        print(f'  Non-cones: {len(self.y)-sum(self.y)} ({100*(len(self.y)-sum(self.y))/len(self.y):.1f}%)')
        print('='*70 + '\n')


# ============================================================================
# GridSearch with Random Forest Classifier
# ============================================================================

class RandomForestConeDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.best_params = None
        self.feature_names = ['height', 'width', 'depth', 'aspect_ratio', 'num_points',
                        'density', 'compactness', 'intensity_mean', 'intensity_std',
                        'linearity', 'planarity', 'volume']
    
    def gridsearch(self, X_train, y_train):

        param_grid = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [10, 15, 20, 25, 30, None],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 5, 6, 10],
            'max_features': ['sqrt', 'log2']
        }
        

        rf = RandomForestClassifier(random_state=42)
        
        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=rf, 
            param_grid=param_grid,
            cv=5,   
            scoring='f1',  
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f'\n✓ GridSearch Complete!')
        print(f'  Best F1 Score: {grid_search.best_score_:.4f}')
        print(f'  Best Params: {grid_search.best_params_}')
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        return self.model
    
    def train(self, X, y, use_gridsearch=True):
        """Train model."""
        # Split data training: 80% test: 20%
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f'\n[Random Forest Training]')
        print(f'  Train: {len(X_train)} | Test: {len(X_test)}')
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.gridsearch(X_train_scaled, y_train)
  
        # Evaluate
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1_score = f1_score(y_test, y_test_pred, zero_division=0)
        
        print(f'\n✓ Evaluation Results:')
        print(f'  Train Acc: {train_accuracy:.2%}')
        print(f'  Val Acc:   {test_acc:.2%}')
        print(f'  Precision: {test_precision:.2%}')
        print(f'  Recall:    {test_recall:.2%}')
        print(f'  F1 Score:  {test_f1_score:.2%}')
        
        logSaver = LogSaver(log_dir='logs')
        logSaver.save(X_train_scaled, X_test_scaled, y_train, y_test,
                     train_accuracy, test_acc, test_precision, test_recall, test_f1_score,
                     self.feature_names, self.model.feature_importances_, self.best_params)

        self.visualize_confusion_matrix(y_test, y_test_pred)
        self.visualize_feature_importances()
            
    def visualize_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-cone', 'Cone'], 
                   yticklabels=['Non-cone', 'Cone'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        script_dir = Path(__file__).parent
        (script_dir / 'figures').mkdir(parents=True, exist_ok=True)
        plt.savefig(script_dir / 'figures' / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✓ Saved: figures/confusion_matrix.png')
        

    def visualize_feature_importances(self):
        importances = self.model.feature_importances_
        feat_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feat_importance_df, x='importance', y='feature', palette='viridis')
        plt.title('Feature Importances')
        plt.xlabel('Importance Score')
        plt.tight_layout()

        script_dir = Path(__file__).parent
        (script_dir / 'figures').mkdir(parents=True, exist_ok=True)
        plt.savefig(script_dir / 'figures' / 'feature_importances.png', dpi=300, bbox_inches='tight')
        print('✓ Saved: figures/feature_importances.png')

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

    def save_cpp_ready(self, path='cone_detector.bin'):
        """Save model in raw binary format for C++"""
        import struct
        
        scaler_mean = self.scaler.mean_.astype(np.float32)
        scaler_std = self.scaler.scale_.astype(np.float32)
        
        with open(path, 'wb') as f:
            # Write scaler mean (12 floats)
            f.write(scaler_mean.tobytes())
            
            # Write scaler std (12 floats)
            f.write(scaler_std.tobytes())
            
            # Write number of trees
            f.write(struct.pack('i', self.model.n_estimators))
            
            # Write each tree
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
        
        print(f'\n✓ C++ Ready: {path}')

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
        self.log_file = self.log_dir / 'training_log.txt'
    
    def save(self, X_train, X_test, y_train, y_test, train_acc, test_acc, precision, recall, f1, 
             features, importances, best_params=None):
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
            'feature_importances': dict(zip(features, importances.tolist()))
        }
        
        with open(self.log_file, 'a') as f:
            json.dump(results, f)
            f.write('\n\n')  # Separate entries
        
        print(f'Training log saved: {self.log_file}')
        return self.log_file
# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Random Forest Cone Detector')
    script_dir = Path(__file__).parent

    default_dataset = script_dir / 'Dataset'
    default_output = script_dir / 'models' / 'cone_detector_rf.pkl'
    default_output_bin = script_dir / 'models' / 'cone_detector.bin'

    parser.add_argument('--dataset', default=str(default_dataset),
                       help='Path to Dataset folder (contains Acceleration/, Skidpad/, Autocross/)')
    parser.add_argument('--output', default=str(default_output))
    parser.add_argument('--output-bin', default=str(default_output_bin))
    
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_bin).parent.mkdir(parents=True, exist_ok=True)

    builder = MultiTrackDatasetBuilder(args.dataset)
    
    X, y = builder.build()
    
    if len(X) < 50:
        print('⚠️  Too few samples!')
        return
    
    model = RandomForestConeDetector()
    model.train(X, y)
    model.save(args.output)
    model.save_cpp_ready(args.output_bin)
    
    print('\n🚀 Model trained and saved!')

if __name__ == '__main__':
    main()
