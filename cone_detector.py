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

class DatasetBuilder:
    def __init__(self, labels_json, clusters_dir='/tmp/cone_clusters'):
        self.labels_json = Path(labels_json).expanduser()
        self.clusters_dir = Path(clusters_dir)
        
        with open(self.labels_json) as f:
            self.labels = json.load(f)
        
        self.X = []
        self.y = []
    
    def build(self):
        """Extract features from labeled clusters."""
        print(f'Building dataset from {len(self.labels)} labels...')
        
        for filename, label in tqdm(self.labels.items()):
            pcd_path = self.clusters_dir / filename
            
            if not pcd_path.exists():
                continue
            
            cluster = load_pcd_binary(str(pcd_path))
            if cluster is None or len(cluster) < 3:
                continue
            
            features = extract_features(cluster)
            self.X.append(features)
            self.y.append(1 if label['is_cone'] else 0)
        
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)
        
        if len(self.X) == 0:
            print('⚠️  No valid clusters found!')
            return np.array([]), np.array([])
        
        print(f'\n✓ Dataset: {len(self.X)} samples')
        if len(self.y) > 0:
            print(f'  Cones: {sum(self.y)} ({100*sum(self.y)/len(self.y):.1f}%)')
            print(f'  Non-cones: {len(self.y)-sum(self.y)} ({100*(len(self.y)-sum(self.y))/len(self.y):.1f}%)')
        
        return self.X, self.y


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
            verbose=2
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
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print('✓ Saved: confusion_matrix.png')

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
        plt.savefig('feature_importances.png', dpi=300, bbox_inches='tight')
        plt.show()
        print('✓ Saved: feature_importances.png')

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

    def save_cpp_ready(self, path='cone_detector_cpp.bin'):
        scaler_mean = self.scaler.mean_.astype(np.float32)
        scaler_std = self.scaler.scale_.astype(np.float32)
        
        trees = []
        for i in range(self.model.n_estimators):
            tree = self.model.estimators_[i].tree_
            trees.append({
                'node_count': tree.node_count,
                'children_left': tree.children_left.astype(np.int32),
                'children_right': tree.children_right.astype(np.int32),
                'feature': tree.feature.astype(np.int32),
                'threshold': tree.threshold.astype(np.float32),
                'value': tree.value.reshape(-1).astype(np.float32)
            })
        
        data = {
            'scaler_mean': scaler_mean,
            'scaler_std': scaler_std,
            'n_features': len(self.feature_names),
            'n_estimators': self.model.n_estimators,
            'trees': trees
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
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
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Random Forest Cone Detector')
    parser.add_argument('--labels', default='~/FRT2026/Cone-Labeler/Cone-Labeler/Dataset/labeled_clusters.json')
    parser.add_argument('--clusters', default='/home/praksz/FRT2026/Cone-Labeler/Cone-Labeler/Dataset/cone_clusters')
    parser.add_argument('--output', default='cone_detector_rf.pkl')
    
    args = parser.parse_args()
    
    builder = DatasetBuilder(args.labels, args.clusters)
    X, y = builder.build()
    
    if len(X) < 50:
        print('⚠️  Too few samples!')
        return
    
    model = RandomForestConeDetector()
    model.train(X, y)
    model.save(args.output)
    model.save_cpp_ready('cone_detector.bin')
    
    print('\n🚀 Model trained and saved!')

if __name__ == '__main__':
    main()
