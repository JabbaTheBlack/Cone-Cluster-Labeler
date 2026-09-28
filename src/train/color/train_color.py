#!/usr/bin/env python3
import json
import struct
import argparse
import time
from itertools import combinations
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


from features import extract_extended_features, extract_svm_intensity_feature
from rf_model import run_rf_pipeline
from xgb_model import run_xgb_pipeline
from pointnet_model import run_pointnet_pipeline
from pointnet2_model import run_pointnet_pipeline as run_pointnet2_pipeline
from svm_model import run_svm_pipeline


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


def grouped_train_test_split_best_effort(X, y, groups, test_size=0.2):
    """
    Split by complete parent/run groups only.

    The function searches group combinations for the test set closest to the
    requested sample-level test ratio. A group can never occur in both train
    and test. Both output sets must contain yellow and non-yellow samples.
    """
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError('Need at least 2 groups for grouped train/test split.')

    total_samples = len(X)
    group_to_indices = {group: np.where(groups == group)[0] for group in unique_groups}

    best = None
    best_score = None

    for group_count in range(1, len(unique_groups)):
        for test_group_combo in combinations(unique_groups, group_count):
            test_group_set = set(test_group_combo)
            train_group_set = set(unique_groups) - test_group_set

            if not train_group_set or not test_group_set:
                continue

            test_idx = np.concatenate([group_to_indices[group] for group in test_group_combo])
            train_idx = np.concatenate([group_to_indices[group] for group in train_group_set])

            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue

            actual_test_ratio = len(test_idx) / total_samples
            ratio_error = abs(actual_test_ratio - test_size)

            train_yellow_ratio = float(y[train_idx].mean())
            test_yellow_ratio = float(y[test_idx].mean())
            class_balance_error = abs(train_yellow_ratio - test_yellow_ratio)

            score = (ratio_error, class_balance_error)
            if best is None or score < best_score:
                best = (
                    train_idx,
                    test_idx,
                    actual_test_ratio,
                    class_balance_error,
                    train_group_set,
                    test_group_set,
                )
                best_score = score

    if best is None:
        raise RuntimeError(
            'Failed to create a valid group-disjoint split with both classes '
            'present in train and test.'
        )

    train_idx, test_idx, actual_test_ratio, class_balance_error, train_groups, test_groups = best

    overlap = set(groups[train_idx]).intersection(groups[test_idx])
    if overlap:
        raise RuntimeError(f'Data leakage detected: groups in both train and test: {overlap}')

    print('\n[Grouped Train/Test Split]')
    print(f'  Requested test ratio: {test_size:.2%}')
    print(f'  Actual test ratio:    {actual_test_ratio:.2%}')
    print(f'  Ratio error:          {abs(actual_test_ratio - test_size):.2%}')
    print(f'  Train samples:        {len(train_idx)}')
    print(f'  Test samples:         {len(test_idx)}')
    print(f'  Train groups:         {len(train_groups)}')
    print(f'  Test groups:          {len(test_groups)}')
    print(f'  Class balance delta:  {class_balance_error:.4f}')
    print('  ✓ Group overlap:      0 (a parent/run cannot be split)')

    return train_idx, test_idx


class MultiTrackDatasetBuilder:
    def __init__(self, base_dataset_path, split_group_level="parent"):
        self.root_path = Path(base_dataset_path).expanduser().resolve()
        self.split_group_level = split_group_level
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


    def _make_group_id(self, track_name: str, cluster_file: str) -> str:
        """
        Create an immutable split group ID.


        Default 'parent':
            All clusters whose file paths share a parent directory are grouped
            together and must remain in the same train/test partition.


        'track':
            The complete labelled track is one group.
        """
        if self.split_group_level == "track":
            return track_name


        if self.split_group_level == "parent":
            parent = Path(cluster_file).parent
            parent_name = parent.as_posix() if str(parent) != "." else "__root__"
            return f"{track_name}::{parent_name}"


        raise ValueError(
            f"Unknown split_group_level: {self.split_group_level}. "
            "Expected 'parent' or 'track'."
        )
    
    def _discover_tracks(self):
        label_files = list(self.label_base.rglob('labeled_clusters.json'))
        for labels_path in label_files:
            track_folder = labels_path.parent
            track_name = str(track_folder.relative_to(self.label_base))
            with open(labels_path) as f:
                labels = json.load(f)
            self.tracks[track_name] = {'path': track_folder, 'labels': labels}


    def build_dataset(
        self,
        extract_feats=True,
        feature_extractor=extract_extended_features,
    ):
        VALID_COLORS = {'orange', 'blue', 'yellow'}
        label_encoder = LabelEncoder()
        data_list, y_raw, groups = [], [], []


        raw_base = self.dataset_root / "raw"
        raw_pcd_map = {p.name: p for p in raw_base.rglob('*.pcd')} if raw_base.exists() else {}
        
        def remap_color(color: str) -> str:
            color = str(color).lower().strip()
            if color not in VALID_COLORS:
                return None
            return 'yellow' if color == 'yellow' else 'non-yellow'


        print('\n📦 Building dataset from labeled clusters...')
        for track_name, track_info in self.tracks.items():
            labels = track_info['labels']
            track_path = track_info['path']
            for cluster_file, label_data in tqdm(labels.items(), desc=f'  {track_name}'):
                binary_color = remap_color(label_data.get('color', ''))
                if binary_color is None:
                    continue
                pcd_path = track_path / cluster_file
                if not pcd_path.exists() and cluster_file in raw_pcd_map:
                    pcd_path = raw_pcd_map[cluster_file]
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
                    feats = feature_extractor(points)
                    if feats is not None:
                        data_list.append(feats)
                        y_raw.append(binary_color)
                        groups.append(self._make_group_id(track_name, cluster_file))
                else:
                    data_list.append(points)
                    y_raw.append(binary_color)
                    groups.append(self._make_group_id(track_name, cluster_file))


        if not data_list:
            raise RuntimeError("No valid labelled colour clusters were found.")


        y = label_encoder.fit_transform(y_raw).astype(np.int64)
        groups = np.asarray(groups)


        X = np.asarray(data_list, dtype=np.float32) if extract_feats else data_list


        print(f"\n✓ Dataset built: {len(y)} samples")
        print(f"  Classes: {list(label_encoder.classes_)}")
        print(f"  Unique immutable run groups: {len(np.unique(groups))}")
        print(f"  Grouping level: {self.split_group_level}")


        return X, y, groups, label_encoder
        
def main():
    parser = argparse.ArgumentParser(description="Multi-model Cone Color Classification Training")
    parser.add_argument("--dataset", type=str, required=True, help="Path to Dataset folder")
    parser.add_argument("--model", type=str, choices=['rf', 'xgb', 'pointnet', 'pointnet2', 'svm'], default='rf', help="Model modality")
    parser.add_argument("--test-size", type=float, default=0.2, help="Requested test ratio using whole parent/run groups")
    args = parser.parse_args()


    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path


    builder = MultiTrackDatasetBuilder(
        dataset_path,
        split_group_level="parent",
    )
    start_time = time.perf_counter()


    if args.model == "rf":
        X, y, groups, label_encoder = builder.build_dataset(
            extract_feats=True,
            feature_extractor=extract_extended_features,
        )

    elif args.model == "xgb":
        X, y, groups, label_encoder = builder.build_dataset(
            extract_feats=True,
            feature_extractor=extract_extended_features,
        )

    elif args.model == "svm":
        X, y, groups, label_encoder = builder.build_dataset(
            extract_feats=True,
            feature_extractor=extract_svm_intensity_feature,
        )

    elif args.model == "pointnet":
        X, y, groups, label_encoder = builder.build_dataset(extract_feats=False)

    elif args.model == "pointnet2":
        X, y, groups, label_encoder = builder.build_dataset(extract_feats=False)


    train_idx, test_idx = grouped_train_test_split_best_effort(
        X,
        y,
        groups,
        test_size=args.test_size,
    )


    if args.model == "rf":
        run_rf_pipeline(X, y, label_encoder, REPO_ROOT, train_idx=train_idx, test_idx=test_idx)

    elif args.model == "xgb":
        run_xgb_pipeline(X, y, label_encoder, REPO_ROOT, train_idx=train_idx, test_idx=test_idx)

    elif args.model == "svm":
        run_svm_pipeline(X, y, label_encoder, REPO_ROOT, train_idx=train_idx, test_idx=test_idx)

    elif args.model == "pointnet":
        run_pointnet_pipeline(X, y, label_encoder, REPO_ROOT, train_idx=train_idx, test_idx=test_idx)

    elif args.model == "pointnet2":
        run_pointnet2_pipeline(X, y, label_encoder, REPO_ROOT, train_idx=train_idx, test_idx=test_idx)


    elapsed = time.perf_counter() - start_time
    print(f"\n⏱️ Execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()