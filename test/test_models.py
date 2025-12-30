#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
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

def extract_features(cluster_points):
    xyz = cluster_points[:, :3]
    intensity = cluster_points[:, 3]
    
    height = float(xyz[:, 2].max() - xyz[:, 2].min())
    width = float(xyz[:, 0].max() - xyz[:, 0].min())
    depth = float(xyz[:, 1].max() - xyz[:, 1].min())
    aspect_ratio = height / (width + 1e-6)
    
    num_points = len(xyz)
    volume = (height + 1e-6) * (width + 1e-6) * (depth + 1e-6)
    density = num_points / volume
    
    center = xyz.mean(axis=0)
    distances = np.linalg.norm(xyz - center, axis=1)
    compactness = np.std(distances) / (np.mean(distances) + 1e-6)
    
    intensity_mean = float(intensity.mean())
    intensity_std = float(intensity.std())
    
    cov = np.cov(xyz.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    linearity = (eigenvalues[2] - eigenvalues[1]) / (eigenvalues[2] + 1e-6)
    planarity = (eigenvalues[1] - eigenvalues[0]) / (eigenvalues[2] + 1e-6)
    
    return np.array([
        height, width, depth, aspect_ratio, num_points,
        density, compactness, intensity_mean, intensity_std,
        linearity, planarity, volume
    ], dtype=np.float32)

class MultiTrackDatasetBuilder:
    def __init__(self, base_dataset_path):
        self.base_path = Path(base_dataset_path).expanduser()
        self.tracks = {}
        self.X = []
        self.y = []
        self._discover_tracks()
    
    def _discover_tracks(self):
        print(f'Searching for labeled_clusters.json in: {self.base_path}')
        found = 0
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
                found += 1
            except Exception as e:
                print(f'⚠️  Error loading {labels_path}: {e}')
        
        print(f'Total labeled_clusters.json found: {found}')
        if not self.tracks:
            print('No labeled data found! Checking structure:')
            for p in self.base_path.rglob('*labeled*'):
                print(f'  {p}')
    
    def build(self):
        if not self.tracks:
            print('❌ No tracks found!')
            return np.array([]), np.array([])
          
        print(f'\n📊 Building test dataset from {len(self.tracks)} tracks...')
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
          
            self.X.extend(X_track)
            self.y.extend(y_track)
          
            print(f'{track_name}: {len(X_track)}/{len(labels)} samples (missing: {missing})')
      
        pbar.close()
        self.X = np.array(self.X, dtype=np.float32) if self.X else np.array([])
        self.y = np.array(self.y, dtype=np.int64) if self.y else np.array([])
        print(f'✓ Final test dataset: {len(self.X)} samples, {self.X.shape[1] if self.X.size else 0} features')
        return self.X, self.y

class CppModelLoader:
    def __init__(self):
        self.n_features = 0
        self.scaler_mean = None
        self.scaler_std = None
        self.n_trees = 0
        self.trees = []
    
    def load_bin(self, path):
        print(f'Loading C++ model: {path}')
        with open(path, 'rb') as f:
            file_content = f.read()
        
        offset = 0
        
        # Detect format: try reading n_features, if reasonable (1-20) use new format
        try:
            potential_n_features = struct.unpack_from('i', file_content, offset)[0]
            if 1 <= potential_n_features <= 20:
                self.n_features = potential_n_features
                offset += 4
                format_type = "new"
            else:
                self.n_features = 12
                format_type = "old"
        except:
            self.n_features = 12
            format_type = "old"
        
        print(f'  Format: {format_type}, Features: {self.n_features}')
        
        # Read scaler mean
        self.scaler_mean = np.frombuffer(file_content[offset:offset + self.n_features*4], dtype=np.float32)
        offset += self.n_features * 4
        
        # Read scaler std
        self.scaler_std = np.frombuffer(file_content[offset:offset + self.n_features*4], dtype=np.float32)
        offset += self.n_features * 4
        
        # Read n_trees
        self.n_trees = struct.unpack_from('i', file_content, offset)[0]
        offset += 4
        
        # Read trees
        for _ in range(self.n_trees):
            node_count = struct.unpack_from('i', file_content, offset)[0]
            offset += 4
            tree = {
                'node_count': node_count,
                'feature': np.empty(node_count, dtype=np.int32),
                'threshold': np.empty(node_count, dtype=np.float32),
                'children_left': np.empty(node_count, dtype=np.int32),
                'children_right': np.empty(node_count, dtype=np.int32),
                'value_cone': np.empty(node_count, dtype=np.float32),
                'value_noncone': np.empty(node_count, dtype=np.float32)
            }
            
            for i in range(node_count):
                tree['feature'][i] = struct.unpack_from('i', file_content, offset)[0]; offset += 4
                tree['threshold'][i] = struct.unpack_from('f', file_content, offset)[0]; offset += 4
                tree['children_left'][i] = struct.unpack_from('i', file_content, offset)[0]; offset += 4
                tree['children_right'][i] = struct.unpack_from('i', file_content, offset)[0]; offset += 4
                tree['value_cone'][i] = struct.unpack_from('f', file_content, offset)[0]; offset += 4
                tree['value_noncone'][i] = struct.unpack_from('f', file_content, offset)[0]; offset += 4
            
            self.trees.append(tree)
        print(f'✓ Loaded {self.n_trees} trees ({self.n_features} features)')
    
    def predict_proba(self, X):
        # Pad/truncate to match model features
        if X.shape[1] < self.n_features:
            X = np.pad(X, ((0,0),(0,self.n_features-X.shape[1])), mode='constant')
        elif X.shape[1] > self.n_features:
            X = X[:, :self.n_features]
        
        X_scaled = (X - self.scaler_mean) / (self.scaler_std + 1e-8)
        n_samples = X_scaled.shape[0]
        proba_cone = np.zeros(n_samples)
        
        for tree in self.trees:
            votes_cone = np.zeros(n_samples)
            
            for i in range(n_samples):
                node = 0
                while node < tree['node_count']:
                    feat_idx = tree['feature'][node]
                    # EXACT ORIGINAL LOGIC: children_left == children_right detects leaf
                    if tree['children_left'][node] == tree['children_right'][node]:
                        # EXACT ORIGINAL LOGIC: use value_noncone
                        votes_cone[i] = tree['value_noncone'][node]
                        break
                    
                    if X_scaled[i, feat_idx] <= tree['threshold'][node]:
                        node = tree['children_left'][node]
                    else:
                        node = tree['children_right'][node]
            
            proba_cone += votes_cone
        
        proba_cone /= self.n_trees
        proba_noncone = 1 - proba_cone
        return np.stack([proba_noncone, proba_cone], axis=1)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(np.int64)

# [evaluate_model and main unchanged - copy from your current version]
def evaluate_model(model, X_test, y_test, name):
    if len(X_test) == 0:
        return {'name': name, 'acc': 0, 'prec': 0, 'rec': 0, 'f1': 0}
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f'\n{"="*50}')
    print(f'📊 {name}')
    print(f'{"="*50}')
    print(f'Accuracy:  {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall:    {rec:.4f}')
    print(f'F1-Score:  {f1:.4f}')
    print(f'{"="*50}')
    
    return {'name': name, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}

def main():
    script_dir = Path(__file__).parent
    ROOT_DIR = script_dir.parent
    
    print(f'Root dir: {ROOT_DIR}')
    print(f'../Dataset exists: { (ROOT_DIR / "Dataset").exists() }')
    
    builder = MultiTrackDatasetBuilder(ROOT_DIR / 'Dataset')
    X_test, y_test = builder.build()
    
    if len(X_test) == 0:
        print('❌ No test data! Check Dataset structure.')
        return
    
    model_dirs = [
        script_dir / 'models',
        ROOT_DIR / 'models'
    ]
    
    all_models = []
    for model_dir in model_dirs:
        if model_dir.exists():
            models = list(model_dir.glob('*.bin'))
            print(f'Found {len(models)} models in {model_dir}')
            all_models.extend(models)
    
    if not all_models:
        print('❌ No .bin models found! Expected in:')
        for d in model_dirs:
            print(f'  {d}')
        return
    
    print(f'\n🔍 Testing {len(all_models)} models on {len(X_test)} samples...')
    
    results = []
    for model_path in sorted(all_models):
        model = CppModelLoader()
        try:
            model.load_bin(model_path)
            result = evaluate_model(model, X_test, y_test, model_path.stem)
            results.append(result)
        except Exception as e:
            print(f'⚠️  Failed to load {model_path.name}: {e}')
    
    if not results:
        print('❌ No models evaluated successfully.')
        return
    
    results.sort(key=lambda x: x['f1'], reverse=True)
    print(f'\n🏆 BEST MODEL: {results[0]["name"]} (F1: {results[0]["f1"]:.4f})')
    
    (ROOT_DIR / 'test').mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(ROOT_DIR / 'test' / 'model_comparison.csv', index=False)
    print('✓ Saved: model_comparison.csv')
    
    test_results = {
        'test_dataset_size': len(X_test),
        'test_features': int(X_test.shape[1]),
        'cones_total': int(np.sum(y_test)),
        'non_cones_total': int(len(y_test) - np.sum(y_test)),
        'models_tested': []
    }
    
    for r in results:
        test_results['models_tested'].append({
            'model_name': r['name'],
            'test_accuracy': float(r['acc']),
            'precision': float(r['prec']),
            'recall': float(r['rec']),
            'f1_score': float(r['f1'])
        })
    
    json_path = ROOT_DIR / 'test' / 'model_test_results.json'
    with open(json_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f'✓ Saved JSON: {json_path}')
    
    print('\n📊 TOP MODELS:')
    print('name,acc,prec,rec,f1')
    for r in results[:3]:
        print(f'{r["name"]},{r["acc"]:.4f},{r["prec"]:.4f},{r["rec"]:.4f},{r["f1"]:.4f}')

if __name__ == '__main__':
    main()
