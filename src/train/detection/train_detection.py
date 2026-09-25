#!/usr/bin/env python3
"""Unified detection trainer: select a model and dispatch to its module."""

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from features import extract_features
from rf_model import run_rf_pipeline
from xgb_model import run_xgb_pipeline
from pointnet_model import run_pointnet_pipeline
from pointnet2_model import run_pointnet2_pipeline


def find_project_root(start_path=None):
    current = Path(start_path or __file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "Dataset").exists() and (candidate / "src").exists():
            return candidate
    return current


REPO_ROOT = find_project_root()


def load_pcd_binary(filepath):
    with open(filepath, "rb") as file:
        while True:
            line = file.readline()
            if not line:
                return None
            if line.decode("utf-8", errors="ignore").strip().upper().startswith("DATA"):
                break
        payload = file.read()

    payload = payload[:len(payload) - len(payload) % 16]
    if not payload:
        return None
    return np.frombuffer(payload, dtype="<f4").reshape(-1, 4).copy()


class MultiTrackDatasetBuilder:
    """Builds one dataset from exactly the folder passed with --dataset."""

    def __init__(self, base_dataset_path, split_group_level="parent"):
        self.base_path = Path(base_dataset_path).expanduser().resolve()
        if not self.base_path.is_dir():
            raise FileNotFoundError(f"Dataset folder does not exist: {self.base_path}")

        self.split_group_level = split_group_level
        self.tracks = {}
        self._discover_tracks()

    def _discover_tracks(self):
        for labels_path in self.base_path.rglob("labeled_clusters.json"):
            with labels_path.open(encoding="utf-8") as file:
                labels = json.load(file)
            track_folder = labels_path.parent
            track_name = str(track_folder.relative_to(self.base_path))
            self.tracks[track_name] = {
                "path": track_folder,
                "labels": labels,
            }
            print(f"✓ Found {track_name}: {len(labels)} labels")

        if not self.tracks:
            raise FileNotFoundError(
                f"No labeled_clusters.json files found below {self.base_path}"
            )

    def _make_group_id(self, track_name, filename):
        if self.split_group_level == "track":
            return track_name
        parent = Path(filename).parent
        return f"{track_name}::{parent.as_posix() if str(parent) != '.' else '__root__'}"

    @staticmethod
    def _is_cone(label):
        if isinstance(label, dict):
            if "is_cone" in label:
                return bool(label["is_cone"])
            label = label.get("label", label.get("class", ""))
        return str(label).lower().strip().replace("_", "-") in {
            "cone", "1", "true", "yes",
        }

    def _resolve_pcd(self, track_path, filename):
        filename_path = Path(filename)
        candidates = [
            track_path / filename,
            self.base_path / filename,
            self.base_path / filename_path.name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        matches = list(self.base_path.rglob(filename_path.name))
        return matches[0] if matches else None

    def build_raw_dataset(self):
        data, labels = [], []
        total_labels = sum(len(track["labels"]) for track in self.tracks.values())
        progress = tqdm(total=total_labels, desc="Processing clusters")

        for track_name, track_data in self.tracks.items():
            for filename, label in track_data["labels"].items():
                try:
                    pcd_path = self._resolve_pcd(track_data["path"], filename)
                    if pcd_path is None:
                        continue
                    points = load_pcd_binary(pcd_path)
                    if points is None or len(points) < 3:
                        continue
                    data.append(points)
                    labels.append(int(self._is_cone(label)))
                finally:
                    progress.update(1)

        progress.close()
        if not data:
            raise RuntimeError(f"No usable samples found below {self.base_path}")
        return data, np.asarray(labels, dtype=np.int64)

    def build_feature_dataset(self):
        raw_clouds, labels = self.build_raw_dataset()
        features = [extract_features(points) for points in raw_clouds]
        valid = [feature is not None for feature in features]
        return (
            np.asarray([feature for feature in features if feature is not None], dtype=np.float32),
            labels[np.asarray(valid)],
        )

    def build_color_dataset(self):
        """Builds raw clouds and string labels for PointNet++ color models."""
        data, labels = [], []
        total_labels = sum(len(track["labels"]) for track in self.tracks.values())
        progress = tqdm(total=total_labels, desc="Processing clusters")

        for track_data in self.tracks.values():
            for filename, label_data in track_data["labels"].items():
                try:
                    pcd_path = self._resolve_pcd(track_data["path"], filename)
                    if pcd_path is None:
                        continue
                    points = load_pcd_binary(pcd_path)
                    if points is None or len(points) < 3:
                        continue
                    label = label_data.get("color", "") if isinstance(label_data, dict) else label_data
                    label = str(label).lower().strip()
                    if label:
                        data.append(points)
                        labels.append(label)
                finally:
                    progress.update(1)

        progress.close()
        if not data:
            raise RuntimeError(f"No usable color samples found below {self.base_path}")
        encoder = LabelEncoder()
        return data, encoder.fit_transform(labels), encoder


def main():
    parser = argparse.ArgumentParser(description="Unified detection model trainer")
    parser.add_argument("--dataset", required=True, help="Folder to process directly")
    parser.add_argument("--model", choices=["rf", "xgb", "pointnet", "pointnet2"], required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--no-gridsearch", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset).expanduser()
    if not dataset.is_absolute():
        dataset = REPO_ROOT / dataset

    builder = MultiTrackDatasetBuilder(dataset)
    start = time.perf_counter()

    if args.model == "rf":
        X, y = builder.build_feature_dataset()
        run_rf_pipeline(X, y, REPO_ROOT, use_gridsearch=not args.no_gridsearch)
    elif args.model == "xgb":
        X, y = builder.build_feature_dataset()
        run_xgb_pipeline(X, y, REPO_ROOT, use_gridsearch=not args.no_gridsearch)
    elif args.model == "pointnet":
        clouds, y = builder.build_raw_dataset()
        run_pointnet_pipeline(clouds, y, None, REPO_ROOT, epochs=args.epochs)
    elif args.model == "pointnet2":
        clouds, y = builder.build_raw_dataset()
        run_pointnet2_pipeline(clouds, y, None, REPO_ROOT, epochs=args.epochs)

    print(f"\nExecution time: {time.perf_counter() - start:.2f} seconds")


if __name__ == "__main__":
    main()