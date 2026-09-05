#!/usr/bin/env python3
import json
import struct
import argparse
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from features import extract_extended_features
from rf_model import run_rf_pipeline
from xgb_model import run_xgb_pipeline
from pointnet_model import run_pointnet_pipeline

def find_project_root(start_path: Path | None = None) -> Path:
    current = (start_path or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "Dataset").exists() and (candidate / "src").exists():
            return candidate
    return current

REPO_ROOT = find_project_root()

def load_pcd_binary(filepath):
    """Load binary PCD file into numpy array."""
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

class MultiTrackDatasetBuilder:
    def __init__(self, base_dataset_path):
        self.root_path = Path(base_dataset_path).expanduser().resolve()
        if (self.root_path / "Processed" / "Color").exists():
            self.dataset_root = self.root_path
            self.label_base = self.root_path / "Processed" / "Color"
        elif self.root_path.name == "Color" and self.root_path.is_dir():
            self.dataset_root = self.root_path.parents[1]
            self.label_base = self.root_path
        else:
            raise FileNotFoundError(f"Could not locate Processed/Color from: {self.root_path}")
        self.tracks = {}
        self._discover_tracks()

    def _discover_tracks(self):
        label_files = list(self.label_base.rglob('labeled_clusters.json'))
        for labels_path in label_files:
            track_folder = labels_path.parent
            track_name = str(track_folder.relative_to(self.label_base))
            with open(labels_path) as f:
                labels = json.load(f)
            self.tracks[track_name] = {'path': track_folder, 'labels': labels}

    def build_dataset(self, extract_feats=True):
        VALID_COLORS = {'orange', 'blue', 'yellow'}
        label_encoder = LabelEncoder()
        data_list, y_raw = [], []

        print('\n📦 Building dataset from labeled clusters...')
        for track_name, track_info in self.tracks.items():
            labels = track_info['labels']
            track_path = track_info['path']
            for cluster_file, label_data in tqdm(labels.items(), desc=f'  {track_name}'):
                color = str(label_data.get('color', '')).lower().strip()
                if color not in VALID_COLORS:
                    continue
                pcd_path = track_path / cluster_file
                if not pcd_path.exists():
                    pcd_path = self.dataset_root / "raw" / track_name / cluster_file
                if not pcd_path.exists():
                    found = list(self.dataset_root.rglob(cluster_file))
                    if found:
                        pcd_path = found[0]
                if not pcd_path.exists():
                    continue

                points = load_pcd_binary(pcd_path)
                if points is None:
                    continue

                if extract_feats:
                    feats = extract_extended_features(points)
                    if feats is not None:
                        data_list.append(feats)
                        y_raw.append(color)
                else:
                    data_list.append(points)
                    y_raw.append(color)

        y = label_encoder.fit_transform(y_raw)
        return (np.array(data_list, dtype=np.float32) if extract_feats else data_list), y, label_encoder

def main():
    parser = argparse.ArgumentParser(description="Multi-model Cone Color Classification Training")
    parser.add_argument("--dataset", type=str, required=True, help="Path to Dataset folder")
    parser.add_argument("--model", type=str, choices=['rf', 'xgb', 'pointnet'], default='rf', help="Model modality")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path

    builder = MultiTrackDatasetBuilder(dataset_path)
    start_time = time.perf_counter()

    if args.model in ['rf', 'xgb']:
        X, y, label_encoder = builder.build_dataset(extract_feats=True)
        if args.model == 'rf':
            run_rf_pipeline(X, y, label_encoder, REPO_ROOT)
        else:
            run_xgb_pipeline(X, y, label_encoder, REPO_ROOT)
    elif args.model == 'pointnet':
        raw_clouds, y, label_encoder = builder.build_dataset(extract_feats=False)
        run_pointnet_pipeline(raw_clouds, y, label_encoder, REPO_ROOT)

    elapsed = time.perf_counter() - start_time
    print(f"\n⏱️ Execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()