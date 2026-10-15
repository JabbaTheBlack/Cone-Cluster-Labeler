"""Labelled cone clusters with the metadata needed for grouped evaluation and provenance."""
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

NAME_RE = re.compile(r'scan_(\d+)_frame_(\d+)_cluster_(\d+)')


def load_pcd_xyzi(path: Path) -> np.ndarray:
    """Reads a binary PCD with fields x y z intensity (float32) into an (N, 4) array."""
    raw = path.read_bytes()
    marker = b'DATA binary\n'
    start = raw.index(marker) + len(marker)
    usable = (len(raw) - start) // 16 * 16
    return np.frombuffer(raw[start:start + usable], dtype=np.float32).reshape(-1, 4).copy()


@dataclass
class ConeDataset:
    points: list                 # list of (N_i, 4) float32 arrays
    meta: pd.DataFrame           # one row per sample: recording, frame, color, distance, ...
    skipped: Counter = field(default_factory=Counter)
    fingerprint: dict = field(default_factory=dict)

    def subset(self, mask) -> 'ConeDataset':
        idx = np.flatnonzero(np.asarray(mask))
        return ConeDataset([self.points[i] for i in idx],
                           self.meta.iloc[idx].reset_index(drop=True),
                           self.skipped, self.fingerprint)

    def context(self, i: int) -> dict:
        """Per-sample information that comes from the rest of the scan, not from the cluster itself."""
        row = self.meta.iloc[i]
        return {f'{p}_{k}': row[f'{p}_{k}'] for p in ('scene', 'yellow') for k in ('level', 'level_near', 'n')}

    def class_counts(self) -> dict:
        table = self.meta.groupby(['recording', 'color']).size().unstack(fill_value=0)
        return {rec: row.to_dict() for rec, row in table.iterrows()}


def locate_color_labels(dataset_path: Path) -> Path:
    dataset_path = Path(dataset_path).expanduser().resolve()
    for candidate in (dataset_path / 'Processed' / 'Color', dataset_path):
        if candidate.name == 'Color' and candidate.is_dir():
            return candidate
    raise FileNotFoundError(f'Could not locate Processed/Color from: {dataset_path}')


def cluster_level(points: np.ndarray) -> float:
    return float(np.median(np.log1p(np.clip(points[:, 3], 0.0, None))))


def scene_context(labels_path: Path, labels: dict, samples: list[dict], near_m: float = 3.0) -> None:
    """Adds scan-level intensity references to every sample (median intensity level of other clusters
    in the same scan):
      scene_*  — all other cone clusters, colour-blind. Caveat: depends on scene composition.
      yellow_* — yellow cones only. At inference these come from the yellow/non-yellow classifier
                 (F1 ~0.91); here the labelled yellows are used, which is slightly optimistic."""
    scans = {}
    for cluster_file, label in labels.items():
        match = NAME_RE.search(cluster_file)
        pcd_path = labels_path.parent / cluster_file
        if not label.get('is_cone') or match is None or not pcd_path.exists():
            continue
        pts = load_pcd_xyzi(pcd_path)
        if len(pts) == 0:
            continue
        centroid = pts[:, :3].mean(axis=0)
        colour = str(label.get('color') or '').lower()
        scans.setdefault(int(match.group(1)), []).append(
            (cluster_file, colour, float(np.hypot(centroid[0], centroid[1])), cluster_level(pts)))
    for s in samples:
        for prefix, keep in (('scene', lambda c: True), ('yellow', lambda c: c == 'yellow')):
            others = [(d, lv) for f, c, d, lv in scans.get(s['scan'], []) if f != s['cluster_file'] and keep(c)]
            near = [lv for d, lv in others if abs(d - s['distance']) < near_m]
            s[f'{prefix}_n'] = len(others)
            s[f'{prefix}_level'] = float(np.median([lv for _, lv in others])) if others else np.nan
            s[f'{prefix}_level_near'] = float(np.median(near)) if near else np.nan


def load_cone_dataset(dataset_path: Path, colors=('blue', 'orange'), min_points: int = 3) -> ConeDataset:
    label_base = locate_color_labels(dataset_path)
    rows, clouds, skipped = [], [], Counter()
    hasher = hashlib.sha256()

    for labels_path in sorted(label_base.rglob('labeled_clusters.json')):
        recording = str(labels_path.parent.relative_to(label_base))
        hasher.update(labels_path.read_bytes())
        labels = json.loads(labels_path.read_text())
        rec_rows = []
        for cluster_file, label in sorted(labels.items()):
            color = str(label.get('color') or '').lower().strip()
            if color not in colors:
                skipped[f'color={color or "none"}'] += 1
                continue
            pcd_path = labels_path.parent / cluster_file
            if not pcd_path.exists():
                skipped['missing_pcd'] += 1
                continue
            pts = load_pcd_xyzi(pcd_path)
            if len(pts) < min_points:
                skipped[f'lt_{min_points}_points'] += 1
                continue
            match = NAME_RE.search(cluster_file)
            if match is None:
                skipped['unparsable_name'] += 1
                continue
            scan, frame, cluster = (int(g) for g in match.groups())
            centroid = pts[:, :3].mean(axis=0)
            hasher.update(f'{recording}/{cluster_file}:{color}:{len(pts)}'.encode())
            rec_rows.append({
                'sample_id': f'{recording}/{cluster_file}',
                'recording': recording,
                'cluster_file': cluster_file,
                'scan': scan,
                'frame': frame,
                'cluster': cluster,
                'color': color,
                'n_points': len(pts),
                'distance': float(np.hypot(centroid[0], centroid[1])),
            })
            clouds.append(pts)
        scene_context(labels_path, labels, rec_rows)
        rows.extend(rec_rows)

    meta = pd.DataFrame(rows)
    fingerprint = {
        'sha256': hasher.hexdigest(),
        'label_root': str(label_base),
        'colors': list(colors),
        'min_points': min_points,
        'n_samples': len(meta),
        'class_totals': meta['color'].value_counts().to_dict() if len(meta) else {},
    }
    return ConeDataset(clouds, meta, skipped, fingerprint)
