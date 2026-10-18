"""Write a copy of a dataset where Color-folder cluster intensities are min-max normalised per frame.

The Detection clusters came from an older perception pipeline that normalised intensity per
frame to [0, 1]; the Color clusters are raw. This reproduces that normalisation for the Color
clusters so intensity no longer reveals which folder a sample came from:

    I_norm = (I - frame_min) / (frame_max - frame_min)

where frame_min/max are taken over all PCD files of that frame (scan_<id>_frame_<n>_cluster_*.pcd)
in the same folder, labelled or not. Detection files are copied unchanged; the source is only read.

usage: python scripts/detection/frame_normalize.py Dataset/Processed Dataset/Processed_FrameNorm
"""
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
if dst.exists():
    sys.exit(f"{dst} already exists")


def read_pcd(path):
    raw = path.read_bytes()
    idx = raw.index(b"DATA binary\n") + len(b"DATA binary\n")
    payload = raw[idx:]
    payload = payload[:len(payload) - len(payload) % 16]
    return raw[:idx], np.frombuffer(payload, dtype="<f4").reshape(-1, 4).copy()


shutil.copytree(src / "Detection", dst / "Detection")
n_frames = n_files = 0
for folder in sorted({p.parent for p in (src / "Color").rglob("*.pcd")}):
    out = dst / folder.relative_to(src)
    out.mkdir(parents=True)
    for extra in folder.glob("*.json"):
        shutil.copy2(extra, out / extra.name)
    frames = defaultdict(list)
    for p in folder.glob("*.pcd"):
        frames[re.match(r"(scan_\d+_frame_\d+)_cluster_\d+", p.name).group(1)].append(p)
    for paths in frames.values():
        clouds = [(p, *read_pcd(p)) for p in paths]
        all_i = np.concatenate([pts[:, 3] for _, _, pts in clouds])
        lo, hi = float(all_i.min()), float(all_i.max())
        for p, header, pts in clouds:
            pts[:, 3] = (pts[:, 3] - lo) / (hi - lo) if hi > lo else 0.0
            (out / p.name).write_bytes(header + pts.astype("<f4").tobytes())
        n_frames += 1
        n_files += len(paths)
print(f"Color: {n_frames} frames, {n_files} PCD files normalised; Detection copied unchanged -> {dst}")
