# #!/usr/bin/env python3
# import json
# import struct
# import argparse
# import time
# from itertools import combinations
# from pathlib import Path
# import numpy as np
# from tqdm import tqdm
# from sklearn.preprocessing import LabelEncoder


# from features import extract_extended_features, extract_svm_intensity_feature
# from rf_model import run_rf_pipeline
# from xgb_model import run_xgb_pipeline
# from pointnet_model import run_pointnet_pipeline
# from pointnet2_model import run_pointnet_pipeline as run_pointnet2_pipeline
# from svm_model import run_svm_pipeline


# def find_project_root(start_path: Path | None = None) -> Path:
#     current = (start_path or Path(__file__)).resolve()
#     if current.is_file():
#         current = current.parent
#     for candidate in [current, *current.parents]:
#         if (candidate / "Dataset").exists() and (candidate / "src").exists():
#             return candidate
#     return current


# REPO_ROOT = find_project_root()


# def load_pcd_binary(filepath):
#     with open(filepath, 'rb') as f:
#         while True:
#             line = f.readline().decode('utf-8').strip()
#             if line.startswith('DATA'):
#                 break
#         points = []
#         while True:
#             data = f.read(16)
#             if not data:
#                 break
#             try:
#                 x, y, z, intensity = struct.unpack('ffff', data)
#                 points.append([x, y, z, intensity])
#             except:
#                 break
#     return np.array(points, dtype=np.float32) if points else None


# def grouped_train_test_split_best_effort(X, y, groups, test_size=0.2):
#     """
#     Split by complete parent/run groups only.

#     The function searches group combinations for the test set closest to the
#     requested sample-level test ratio. A group can never occur in both train
#     and test. Both output sets must contain yellow and non-yellow samples.
#     """
#     unique_groups = np.unique(groups)
#     if len(unique_groups) < 2:
#         raise ValueError('Need at least 2 groups for grouped train/test split.')

#     total_samples = len(X)
#     group_to_indices = {group: np.where(groups == group)[0] for group in unique_groups}

#     best = None
#     best_score = None

#     for group_count in range(1, len(unique_groups)):
#         for test_group_combo in combinations(unique_groups, group_count):
#             test_group_set = set(test_group_combo)
#             train_group_set = set(unique_groups) - test_group_set

#             if not train_group_set or not test_group_set:
#                 continue

#             test_idx = np.concatenate([group_to_indices[group] for group in test_group_combo])
#             train_idx = np.concatenate([group_to_indices[group] for group in train_group_set])

#             if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
#                 continue

#             actual_test_ratio = len(test_idx) / total_samples
#             ratio_error = abs(actual_test_ratio - test_size)

#             train_yellow_ratio = float(y[train_idx].mean())
#             test_yellow_ratio = float(y[test_idx].mean())
#             class_balance_error = abs(train_yellow_ratio - test_yellow_ratio)

#             score = (ratio_error, class_balance_error)
#             if best is None or score < best_score:
#                 best = (
#                     train_idx,
#                     test_idx,
#                     actual_test_ratio,
#                     class_balance_error,
#                     train_group_set,
#                     test_group_set,
#                 )
#                 best_score = score

#     if best is None:
#         raise RuntimeError(
#             'Failed to create a valid group-disjoint split with both classes '
#             'present in train and test.'
#         )

#     train_idx, test_idx, actual_test_ratio, class_balance_error, train_groups, test_groups = best

#     overlap = set(groups[train_idx]).intersection(groups[test_idx])
#     if overlap:
#         raise RuntimeError(f'Data leakage detected: groups in both train and test: {overlap}')

#     print('\n[Grouped Train/Test Split]')
#     print(f'  Requested test ratio: {test_size:.2%}')
#     print(f'  Actual test ratio:    {actual_test_ratio:.2%}')
#     print(f'  Ratio error:          {abs(actual_test_ratio - test_size):.2%}')
#     print(f'  Train samples:        {len(train_idx)}')
#     print(f'  Test samples:         {len(test_idx)}')
#     print(f'  Train groups:         {len(train_groups)}')
#     print(f'  Test groups:          {len(test_groups)}')
#     print(f'  Class balance delta:  {class_balance_error:.4f}')
#     print('  ✓ Group overlap:      0 (a parent/run cannot be split)')

#     return train_idx, test_idx


# class MultiTrackDatasetBuilder:
#     def __init__(self, base_dataset_path, split_group_level="parent"):
#         self.root_path = Path(base_dataset_path).expanduser().resolve()
#         self.split_group_level = split_group_level

#         # 1. Path contains 'Processed/Color' somewhere in its hierarchy
#         if (self.root_path / "Processed" / "Color").exists():
#             self.dataset_root = self.root_path
#             self.label_base = self.root_path / "Processed" / "Color"
#         elif "Processed/Color" in self.root_path.as_posix() or "Processed/Color" in str(self.root_path):
#             # Find the root project folder containing 'Dataset'
#             parents = [self.root_path, *self.root_path.parents]
#             dataset_root = next((p for p in parents if p.name == "Dataset" or (p / "Dataset").exists()), None)
            
#             if dataset_root and (dataset_root / "Dataset").exists():
#                 self.dataset_root = dataset_root
#             elif dataset_root and dataset_root.name == "Dataset":
#                 self.dataset_root = dataset_root.parent
#             else:
#                 self.dataset_root = REPO_ROOT

#             self.label_base = self.root_path
#         else:
#             raise FileNotFoundError(f"Could not locate Processed/Color from: {self.root_path}")

#         self.tracks = {}
#         self._discover_tracks()


#     def _make_group_id(self, track_name: str, cluster_file: str) -> str:
#         """
#         Create an immutable split group ID.


#         Default 'parent':
#             All clusters whose file paths share a parent directory are grouped
#             together and must remain in the same train/test partition.


#         'track':
#             The complete labelled track is one group.
#         """
#         if self.split_group_level == "track":
#             return track_name


#         if self.split_group_level == "parent":
#             parent = Path(cluster_file).parent
#             parent_name = parent.as_posix() if str(parent) != "." else "__root__"
#             return f"{track_name}::{parent_name}"


#         raise ValueError(
#             f"Unknown split_group_level: {self.split_group_level}. "
#             "Expected 'parent' or 'track'."
#         )
    
#     def _discover_tracks(self):
#         label_files = list(self.label_base.rglob('labeled_clusters.json'))
#         for labels_path in label_files:
#             track_folder = labels_path.parent
#             track_name = str(track_folder.relative_to(self.label_base))
#             with open(labels_path) as f:
#                 labels = json.load(f)
#             self.tracks[track_name] = {'path': track_folder, 'labels': labels}


#     def build_dataset(
#         self,
#         extract_feats=True,
#         feature_extractor=extract_extended_features,
#         multiclass=False,
#     ):
#         VALID_COLORS = {'orange', 'blue', 'yellow'}
#         label_encoder = LabelEncoder()
#         data_list, y_raw, groups = [], [], []


#         raw_base = self.dataset_root / "raw"
#         raw_pcd_map = {p.name: p for p in raw_base.rglob('*.pcd')} if raw_base.exists() else {}
        
#         def remap_color(color: str) -> str:
#             color = str(color).lower().strip()
#             if color not in VALID_COLORS:
#                 return None
#             if multiclass:
#                 return color
#             return 'yellow' if color == 'yellow' else 'non-yellow'


#         print('\n📦 Building dataset from labeled clusters...')
#         for track_name, track_info in self.tracks.items():
#             labels = track_info['labels']
#             track_path = track_info['path']
#             for cluster_file, label_data in tqdm(labels.items(), desc=f'  {track_name}'):
#                 binary_color = remap_color(label_data.get('color', ''))
#                 if binary_color is None:
#                     continue
#                 pcd_path = track_path / cluster_file
#                 if not pcd_path.exists() and cluster_file in raw_pcd_map:
#                     pcd_path = raw_pcd_map[cluster_file]
#                 if not pcd_path.exists():
#                     pcd_path = self.dataset_root / "raw" / track_name / cluster_file
#                 if not pcd_path.exists():
#                     found = list(self.dataset_root.rglob(cluster_file))
#                     if found:
#                         pcd_path = found[0]
#                 if not pcd_path.exists():
#                     continue


#                 points = load_pcd_binary(pcd_path)
#                 if points is None:
#                     continue


#                 if extract_feats:
#                     feats = feature_extractor(points)
#                     if feats is not None:
#                         data_list.append(feats)
#                         y_raw.append(binary_color)
#                         groups.append(self._make_group_id(track_name, cluster_file))
#                 else:
#                     data_list.append(points)
#                     y_raw.append(binary_color)
#                     groups.append(self._make_group_id(track_name, cluster_file))


#         if not data_list:
#             raise RuntimeError("No valid labelled colour clusters were found.")


#         y = label_encoder.fit_transform(y_raw).astype(np.int64)
#         groups = np.asarray(groups)


#         X = np.asarray(data_list, dtype=np.float32) if extract_feats else data_list


#         print(f"\n✓ Dataset built: {len(y)} samples")
#         print(f"  Classes: {list(label_encoder.classes_)}")
#         print(f"  Unique immutable run groups: {len(np.unique(groups))}")
#         print(f"  Grouping level: {self.split_group_level}")


#         return X, y, groups, label_encoder
        
# def main():
#     parser = argparse.ArgumentParser(description="Multi-model Cone Color Classification Training")
#     parser.add_argument("--dataset", type=str, required=True, help="Path to Dataset folder")
#     parser.add_argument("--model", type=str, choices=['rf', 'xgb', 'pointnet', 'pointnet2', 'svm'], default='rf', help="Model modality")
#     parser.add_argument("--test-size", type=float, default=0.2, help="Requested test ratio using whole parent/run groups")
#     parser.add_argument("--multiclass", action='store_true', help="Use multiclass classification")

#     args = parser.parse_args()


#     dataset_path = Path(args.dataset).expanduser()
#     if not dataset_path.is_absolute():
#         dataset_path = REPO_ROOT / dataset_path


#     builder = MultiTrackDatasetBuilder(
#         dataset_path,
#         split_group_level="parent",
#     )
#     start_time = time.perf_counter()


#     if args.model == "rf":
#         X, y, groups, label_encoder = builder.build_dataset(
#             extract_feats=True,
#             feature_extractor=extract_extended_features,
#         )

#     elif args.model == "xgb":
#         X, y, groups, label_encoder = builder.build_dataset(
#             extract_feats=True,
#             feature_extractor=extract_extended_features,
#         )

#     elif args.model == "svm":
#         X, y, groups, label_encoder = builder.build_dataset(
#             extract_feats=True,
#             feature_extractor=extract_svm_intensity_feature,
#         )

#     elif args.model == "pointnet":
#         X, y, groups, label_encoder = builder.build_dataset(extract_feats=False, multiclass=args.multiclass)

#     elif args.model == "pointnet2":
#         X, y, groups, label_encoder = builder.build_dataset(extract_feats=False, multiclass=args.multiclass)


#     train_idx, test_idx = grouped_train_test_split_best_effort(
#         X,
#         y,
#         groups,
#         test_size=args.test_size,
#     )


#     if args.model == "rf":
#         run_rf_pipeline(X, y, label_encoder, REPO_ROOT, train_idx=train_idx, test_idx=test_idx)

#     elif args.model == "xgb":
#         run_xgb_pipeline(X, y, label_encoder, REPO_ROOT, train_idx=train_idx, test_idx=test_idx)

#     elif args.model == "svm":
#         run_svm_pipeline(X, y, label_encoder, REPO_ROOT, train_idx=train_idx, test_idx=test_idx)

#     elif args.model == "pointnet":
#         run_pointnet_pipeline(X, y, label_encoder, REPO_ROOT, train_idx=train_idx, test_idx=test_idx)

#     elif args.model == "pointnet2":
#         run_pointnet2_pipeline(X, y, label_encoder, REPO_ROOT, train_idx=train_idx, test_idx=test_idx)


#     elapsed = time.perf_counter() - start_time
#     print(f"\n⏱️ Execution time: {elapsed:.2f} seconds")


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
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
from sklearn.mixture import GaussianMixture

from features import extract_extended_features, extract_svm_intensity_feature
from rf_model import run_rf_pipeline
from xgb_model import run_xgb_pipeline
from pointnet_model import run_pointnet_pipeline
from pointnet2_model import run_pointnet_pipeline as run_pointnet2_pipeline
from svm_model import run_svm_pipeline

RECLASS_DIR_NAME = "Ground Filter Reclassification"

# Fixed-threshold stripe rule (mirrors the C++ constants). Applied to normalized intensity
# on a 0-100 scale: norm = raw_intensity / intensity_max * 100.
WHITE_STRIPE_NORM_MIN = 70.0
BLACK_STRIPE_NORM_MAX = 35.0
DEFAULT_INTENSITY_MAX = 255.0  # raw intensity value that maps to norm = 100; verify for your sensor

STRIPE_MODES = ('no_stripes', 'only_stripes', 'fixed_no_stripes', 'fixed_only_stripes')


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
            except Exception:
                break
    return np.array(points, dtype=np.float32) if points else None


def filter_stripe_points(points: np.ndarray, color: str, mode: str) -> np.ndarray:
    if points is None or len(points) < 3 or mode not in ("no_stripes", "only_stripes"):
        return points

    intensity = points[:, 3]
    stripe_mask = np.zeros(len(points), dtype=bool)

    if color in ("orange", "blue"):
        thresh = np.percentile(intensity, 75)
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            labels = gmm.fit_predict(intensity.reshape(-1, 1))
            means = gmm.means_.flatten()
            retro_cluster = np.argmax(means)
            if means[retro_cluster] > means[1 - retro_cluster] * 1.3:
                stripe_mask = (labels == retro_cluster)
            else:
                stripe_mask = (intensity >= thresh)
        except Exception:
            stripe_mask = (intensity >= thresh)

    elif color == "yellow":
        thresh_low = np.percentile(intensity, 25)
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            labels = gmm.fit_predict(intensity.reshape(-1, 1))
            means = gmm.means_.flatten()
            black_cluster = np.argmin(means)
            if means[black_cluster] < means[1 - black_cluster] * 0.7:
                stripe_mask = (labels == black_cluster)
            else:
                stripe_mask = (intensity <= thresh_low)
        except Exception:
            stripe_mask = (intensity <= thresh_low)

    if mode == "only_stripes":
        return points[stripe_mask]
    elif mode == "no_stripes":
        return points[~stripe_mask]

    return points


def filter_stripe_points_fixed(points: np.ndarray, mode: str, intensity_max: float) -> np.ndarray:
    """Label-agnostic fixed-threshold stripe filter.

    A point is a stripe point if its normalized intensity is a white stripe
    (norm >= WHITE_STRIPE_NORM_MIN) or a black stripe (norm <= BLACK_STRIPE_NORM_MAX).
    The same rule is applied to every cluster regardless of its color label.
    """
    if points is None or len(points) == 0:
        return points
    norm = points[:, 3] / intensity_max * 100.0
    stripe_mask = (norm >= WHITE_STRIPE_NORM_MIN) | (norm <= BLACK_STRIPE_NORM_MAX)
    if mode == "fixed_only_stripes":
        return points[stripe_mask]
    if mode == "fixed_no_stripes":
        return points[~stripe_mask]
    return points


def grouped_train_test_split_best_effort(X, y, groups, test_size=0.2):
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
        raise RuntimeError('Failed to create a valid group-disjoint split with both classes present in train and test.')

    train_idx, test_idx, actual_test_ratio, class_balance_error, train_groups, test_groups = best

    overlap = set(groups[train_idx]).intersection(groups[test_idx])
    if overlap:
        raise RuntimeError(f'Data leakage detected: groups in both train and test: {overlap}')

    return train_idx, test_idx


class MultiTrackDatasetBuilder:
    def __init__(self, base_dataset_path, split_group_level="parent", exclude_dirs=(), intensity_max=DEFAULT_INTENSITY_MAX):
        self.root_path = Path(base_dataset_path).expanduser().resolve()
        self.split_group_level = split_group_level
        self.exclude_dirs = set(exclude_dirs)
        self.intensity_max = float(intensity_max)

        if (self.root_path / "Processed" / "Color").exists():
            self.dataset_root = self.root_path
            self.label_base = self.root_path / "Processed" / "Color"
        elif "Processed/Color" in self.root_path.as_posix() or "Processed/Color" in str(self.root_path):
            parents = [self.root_path, *self.root_path.parents]
            dataset_root = next((p for p in parents if p.name == "Dataset" or (p / "Dataset").exists()), None)
            if dataset_root and (dataset_root / "Dataset").exists():
                self.dataset_root = dataset_root
            elif dataset_root and dataset_root.name == "Dataset":
                self.dataset_root = dataset_root.parent
            else:
                self.dataset_root = REPO_ROOT
            self.label_base = self.root_path
        else:
            raise FileNotFoundError(f"Could not locate Processed/Color from: {self.root_path}")

        self.tracks = {}
        self._discover_tracks()

    def _make_group_id(self, track_name: str, cluster_file: str) -> str:
        if self.split_group_level == "track":
            return track_name
        if self.split_group_level == "parent":
            parent = Path(cluster_file).parent
            parent_name = parent.as_posix() if str(parent) != "." else "__root__"
            return f"{track_name}::{parent_name}"
        raise ValueError(f"Unknown split_group_level: {self.split_group_level}")

    def _discover_tracks(self):
        label_files = list(self.label_base.rglob('labeled_clusters.json'))
        excluded = 0
        for labels_path in label_files:
            rel_parts = labels_path.relative_to(self.label_base).parts[:-1]
            if self.exclude_dirs and any(part in self.exclude_dirs for part in rel_parts):
                excluded += 1
                continue
            track_folder = labels_path.parent
            track_name = str(track_folder.relative_to(self.label_base))
            with open(labels_path) as f:
                labels = json.load(f)
            self.tracks[track_name] = {'path': track_folder, 'labels': labels}
        print(f"🔎 Tracks discovered: {len(self.tracks)} (excluded: {excluded}"
              f"{', dirs=' + str(sorted(self.exclude_dirs)) if self.exclude_dirs else ''})")

    def build_dataset(
        self,
        extract_feats=True,
        feature_extractor=extract_extended_features,
        multiclass=False,
        stripe_mode=None,
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
            if multiclass:
                return color
            return 'yellow' if color == 'yellow' else 'non-yellow'

        print(f'\n📦 Building dataset (Stripe Mode: {stripe_mode} | Multiclass: {multiclass})...')
        stats = {'pts_in': 0, 'pts_out': 0, 'dropped_small': 0, 'i_min': np.inf, 'i_max': -np.inf}
        for track_name, track_info in self.tracks.items():
            labels = track_info['labels']
            track_path = track_info['path']
            for cluster_file, label_data in tqdm(labels.items(), desc=f'  {track_name}'):
                raw_color = str(label_data.get('color', '')).lower().strip()
                binary_color = remap_color(raw_color)
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

                stats['i_min'] = min(stats['i_min'], float(points[:, 3].min()))
                stats['i_max'] = max(stats['i_max'], float(points[:, 3].max()))

                if stripe_mode in ("fixed_no_stripes", "fixed_only_stripes"):
                    processed_points = filter_stripe_points_fixed(points, stripe_mode, self.intensity_max)
                else:
                    processed_points = filter_stripe_points(points, raw_color, stripe_mode)

                stats['pts_in'] += len(points)
                stats['pts_out'] += 0 if processed_points is None else len(processed_points)
                if processed_points is None or len(processed_points) < 3:
                    stats['dropped_small'] += 1
                    continue

                if extract_feats:
                    feats = feature_extractor(processed_points)
                    if feats is not None:
                        data_list.append(feats)
                        y_raw.append(binary_color)
                        groups.append(self._make_group_id(track_name, cluster_file))
                else:
                    data_list.append(processed_points)
                    y_raw.append(binary_color)
                    groups.append(self._make_group_id(track_name, cluster_file))

        if not data_list:
            raise RuntimeError("No valid labelled colour clusters were found.")

        print(f"   Raw intensity range: [{stats['i_min']:.2f}, {stats['i_max']:.2f}] "
              f"(intensity_max={self.intensity_max}) | kept {stats['pts_out']}/{stats['pts_in']} points "
              f"| clusters dropped (<3 pts): {stats['dropped_small']} | clusters kept: {len(data_list)}")
        if stripe_mode and stripe_mode.startswith("fixed_"):
            print(f"   Fixed thresholds: white >= {WHITE_STRIPE_NORM_MIN}, black <= {BLACK_STRIPE_NORM_MAX} (normalized 0-100)")

        y = label_encoder.fit_transform(y_raw).astype(np.int64)
        groups = np.asarray(groups)
        X = np.asarray(data_list, dtype=np.float32) if extract_feats else data_list

        return X, y, groups, label_encoder


def main():
    parser = argparse.ArgumentParser(description="Multi-model Cone Color Classification Training")
    parser.add_argument("--dataset", type=str, default="Dataset/Processed/Color", help="Path to Dataset folder")
    parser.add_argument("--model", type=str, choices=['rf', 'xgb', 'pointnet', 'pointnet2', 'svm'], default='rf', help="Model modality")
    parser.add_argument("--test-size", type=float, default=0.2, help="Requested test ratio using whole parent/run groups")
    parser.add_argument("--multiclass", action='store_true', help="Use multiclass classification")
    parser.add_argument("--stripe-mode", type=str, choices=list(STRIPE_MODES), default=None,
                        help="Stripe filtering: no_stripes/only_stripes use per-cluster GMM/percentile; "
                             "fixed_* use the fixed normalized-intensity thresholds")
    parser.add_argument("--intensity-max", type=float, default=DEFAULT_INTENSITY_MAX,
                        help="Raw intensity that maps to normalized 100 for the fixed thresholds")
    parser.add_argument("--exclude-reclass", action='store_true',
                        help=f"Skip tracks under '{RECLASS_DIR_NAME}' when discovering labeled clusters")
    parser.add_argument("--run-id", type=str, default="", help="Unique suffix attached to generated output figure filenames")

    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path

    builder = MultiTrackDatasetBuilder(
        dataset_path,
        split_group_level="parent",
        exclude_dirs=(RECLASS_DIR_NAME,) if args.exclude_reclass else (),
        intensity_max=args.intensity_max,
    )
    start_time = time.perf_counter()

    # Snapshot existing figures so only this run's new files get renamed afterwards
    fig_dir = REPO_ROOT / 'figures' / 'color'
    figs_before = {p.resolve() for p in fig_dir.rglob('*.png')} if fig_dir.exists() else set()

    if args.model in ("rf", "xgb"):
        X, y, groups, label_encoder = builder.build_dataset(
            extract_feats=True,
            feature_extractor=extract_extended_features,
            multiclass=args.multiclass,
            stripe_mode=args.stripe_mode,
        )

    elif args.model == "svm":
        X, y, groups, label_encoder = builder.build_dataset(
            extract_feats=True,
            feature_extractor=extract_svm_intensity_feature,
            multiclass=args.multiclass,
            stripe_mode=args.stripe_mode,
        )

    elif args.model in ("pointnet", "pointnet2"):
        X, y, groups, label_encoder = builder.build_dataset(
            extract_feats=False,
            multiclass=args.multiclass,
            stripe_mode=args.stripe_mode,
        )

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

    # Rename ONLY figures produced by this run (new paths or overwritten-in-place files)
    if args.run_id and fig_dir.exists():
        run_start_wall = time.time() - (time.perf_counter() - start_time) - 1.0
        for fig_file in list(fig_dir.rglob('*.png')):
            resolved = fig_file.resolve()
            is_new = resolved not in figs_before
            is_rewritten = fig_file.stat().st_mtime >= run_start_wall
            if not (is_new or is_rewritten):
                continue
            if fig_file.stem.endswith(args.run_id):
                continue
            new_path = fig_file.with_name(f"{fig_file.stem}_{args.run_id}{fig_file.suffix}")
            if new_path.exists():
                new_path.unlink()
            fig_file.rename(new_path)

    elapsed = time.perf_counter() - start_time
    print(f"\n⏱️ Execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()