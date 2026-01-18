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

def extract_features(points):
    if points is None or len(points) < 3:
        return None

    xyz = points[:, :3]
    intensity = points[:, 3]

    height = xyz[:, 2].ptp()
    width = max(xyz[:, 0].std(), xyz[:, 1].std(), 1e-6)
    aspect_ratio = height / width

    avg_intensity = intensity.mean()
    std_intensity = intensity.std()

    z = xyz[:, 2]
    v_grad = np.sum((z - z.mean()) * (intensity - avg_intensity)) / (
        np.sum((z - z.mean()) ** 2) + 1e-6
    )

    contrast = (intensity.max() - intensity.min()) / (avg_intensity + 1e-6)
    reflective_ratio = np.mean(intensity > avg_intensity * 1.5)

    return np.array([
        height,
        width,
        aspect_ratio,
        avg_intensity,
        std_intensity,
        v_grad,
        contrast,
        reflective_ratio
    ], dtype=np.float32)


class MultiTrackDatasetBuilder:
    """Combines labeled clusters from multiple track folders into one dataset."""
    
    def __init__(self, base_dataset_path):
        """
        Args:
            base_dataset_path: Path to Dataset folder (contains Acceleration/, Skidpad/, Autocross/)
        """
        self.base_path = Path(base_dataset_path).expanduser() / "Processed" / "Detection"
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

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test labeled_clusters.json discovery")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to Dataset folder (the one containing Processed/Detection)"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    print(f"\n🚀 Dataset root: {dataset_path}\n")

    builder = MultiTrackDatasetBuilder(dataset_path)

    print("\n✅ DATASET DISCOVERY SUCCESS")
    print(f"Tracks found: {len(builder.tracks)}\n")

    for name, data in builder.tracks.items():
        print(f"Track: {name}")
        print(f"  Path: {data['path']}")
        print(f"  Labels: {data['label_count']}")
        print("-" * 50)


if __name__ == "__main__":
    main()