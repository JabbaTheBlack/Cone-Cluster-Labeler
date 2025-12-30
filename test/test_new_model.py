#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import struct

def load_pcd_binary(filepath):
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

def extract_features_7(cluster_points):
    """EXACT 7 features from your train.py"""
    xyz = cluster_points[:, :3]
    intensity = cluster_points[:, 3]
    
    height = float(xyz[:, 2].max() - xyz[:, 2].min())
    width = float(xyz[:, 0].max() - xyz[:, 0].min())
    depth = float(xyz[:, 1].max() - xyz[:, 1].min())
    aspect_ratio = height / (width + 1e-6)
    
    num_points = len(xyz)
    volume = (height + 1e-6) * (width + 1e-6) * (depth + 1e-6)
    density = num_points / volume
    
    intensity_std = float(intensity.std())
    
    return np.array([
        height, width, depth, aspect_ratio,
        density, intensity_std, volume
    ], dtype=np.float32)

class NewModelTester:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.n_features = 0
        self.scaler_mean = None
        self.scaler_std = None
        self.n_trees = 0
        self.trees = []
        self.load_model()
    
    def load_model(self):
        print(f'🔍 Loading NEW 7-feature model: {self.model_path}')
        with open(self.model_path, 'rb') as f:
            file_content = f.read()
        
        offset = 0
        self.n_features = struct.unpack_from('i', file_content, offset)[0]
        offset += 4
        
        print(f'  Model has {self.n_features} features (expecting 7)')
        
        self.scaler_mean = np.frombuffer(file_content[offset:offset+self.n_features*4], dtype=np.float32)
        offset += self.n_features * 4
        
        self.scaler_std = np.frombuffer(file_content[offset:offset+self.n_features*4], dtype=np.float32)
        offset += self.n_features * 4
        
        self.n_trees = struct.unpack_from('i', file_content, offset)[0]
        offset += 4
        
        print(f'  Loading {self.n_trees} trees...')
        for tree_idx in range(self.n_trees):
            node_count = struct.unpack_from('i', file_content, offset)[0]
            offset += 4
            
            tree = {
                'node_count': node_count,
                'feature': np.empty(node_count, dtype=np.int32),
                'threshold': np.empty(node_count, dtype=np.float32),
                'children_left': np.empty(node_count, dtype=np.int32),
                'children_right': np.empty(node_count, dtype=np.int32),
                'value_noncone': np.empty(node_count, dtype=np.float32),  # tree.value[i][0][0]
                'value_cone': np.empty(node_count, dtype=np.float32),    # tree.value[i][0][1]
            }
            
            for i in range(node_count):
                tree['feature'][i] = struct.unpack_from('i', file_content, offset)[0]; offset += 4
                tree['threshold'][i] = struct.unpack_from('f', file_content, offset)[0]; offset += 4
                tree['children_left'][i] = struct.unpack_from('i', file_content, offset)[0]; offset += 4
                tree['children_right'][i] = struct.unpack_from('i', file_content, offset)[0]; offset += 4
                tree['value_noncone'][i] = struct.unpack_from('f', file_content, offset)[0]; offset += 4  # class 0
                tree['value_cone'][i] = struct.unpack_from('f', file_content, offset)[0]; offset += 4      # class 1
            
            self.trees.append(tree)
        
        print(f'✅ Model loaded: {self.n_features} feats, {self.n_trees} trees')
    
    def predict_proba(self, X):
      assert X.shape[1] == 7, f"Expected 7 features, got {X.shape[1]}"
      assert self.n_features == 7, f"Model expects {self.n_features} features"
      
      X_scaled = (X - self.scaler_mean) / (self.scaler_std + 1e-8)
      n_samples = X_scaled.shape[0]
      votes_cone = np.zeros(n_samples)
      
      for tree in self.trees:
          tree_votes = np.zeros(n_samples)
          
          for i in range(n_samples):
              node = 0
              while node < tree['node_count']:
                  if tree['children_left'][node] == tree['children_right'][node]:
                      # FIXED: Use value_cone for cone prediction (class 1)
                      tree_votes[i] = tree['value_cone'][node]  # ← THIS WAS WRONG
                      break
                  
                  feat_idx = tree['feature'][node]
                  if X_scaled[i, feat_idx] <= tree['threshold'][node]:
                      node = tree['children_left'][node]
                  else:
                      node = tree['children_right'][node]
          
          votes_cone += tree_votes
      
      proba_cone = votes_cone / self.n_trees
      proba_noncone = 1 - proba_cone
      return np.stack([proba_noncone, proba_cone], axis=1)

    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

def test_single_model():
    script_dir = Path(__file__).parent
    ROOT_DIR = script_dir.parent
    
    # Load test dataset (7 features)
    builder = MultiTrackDatasetBuilder(ROOT_DIR / 'Dataset')
    X_test, y_test = builder.build()
    
    if len(X_test) == 0:
        print('❌ No test data!')
        return
    
    print(f'✅ Loaded {len(X_test)} test samples, {X_test.shape[1]} features')
    print(f'Cones: {np.sum(y_test)} ({100*np.sum(y_test)/len(y_test):.1f}%)')
    
    # Test NEW 7-feature model
    model_path = ROOT_DIR / 'models' / 'cone_detector.bin'
    if not model_path.exists():
        print(f'❌ Model not found: {model_path}')
        return
    
    model = NewModelTester(model_path)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f'\n{"="*60}')
    print(f'🚀 NEW 7-FEATURE MODEL RESULTS')
    print(f'{"="*60}')
    print(f'Accuracy:  {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall:    {rec:.4f}')
    print(f'F1-Score:  {f1:.4f}')
    print(f'{"="*60}')
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f'Confusion Matrix:\n{np.array2string(cm, formatter={"int": lambda x: str(int(x))})}')
    
    print(f'\n✅ Test complete! F1: {f1:.4f}')

class MultiTrackDatasetBuilder:
    def __init__(self, base_dataset_path):
        self.base_path = Path(base_dataset_path).expanduser()
        self.tracks = {}
        self.X = []
        self.y = []
        self._discover_tracks()
    
    def _discover_tracks(self):
        for labels_path in self.base_path.rglob('labeled_clusters.json'):
            track_folder = labels_path.parent
            track_name = str(track_folder.relative_to(self.base_path)).strip('/')
            try:
                with open(labels_path) as f:
                    labels = json.load(f)
                self.tracks[track_name] = {
                    'path': track_folder,
                    'labels': labels,
                    'label_count': len(labels)
                }
                print(f'✓ Found {track_name}: {len(labels)} labels')
            except Exception as e:
                print(f'⚠️  Error loading {labels_path}: {e}')
    
    def build(self):
        print(f'\n📊 Loading test data from {len(self.tracks)} tracks...')
        total_labels = sum(t['label_count'] for t in self.tracks.values())
        from tqdm import tqdm
        pbar = tqdm(total=total_labels, desc='Processing clusters')
        
        for track_name, track_data in self.tracks.items():
            track_path = track_data['path']
            labels = track_data['labels']
            
            X_track, y_track = [], []
            for filename, label in labels.items():
                pcd_path = track_path / filename
                if not pcd_path.exists():
                    pbar.update(1)
                    continue
                
                cluster = load_pcd_binary(str(pcd_path))
                if cluster is None or len(cluster) < 3:
                    pbar.update(1)
                    continue
                
                features = extract_features_7(cluster)
                X_track.append(features)
                y_track.append(1 if label['is_cone'] else 0)
                pbar.update(1)
            
            self.X.extend(X_track)
            self.y.extend(y_track)
        
        pbar.close()
        self.X = np.array(self.X, dtype=np.float32) if self.X else np.array([])
        self.y = np.array(self.y, dtype=np.int64) if self.y else np.array([])
        print(f'✓ Test dataset: {len(self.X)} samples, {self.X.shape[1]} features')
        return self.X, self.y

if __name__ == '__main__':
    test_single_model()
