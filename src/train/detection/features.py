#!/usr/bin/env python3
import numpy as np

FEATURE_NAMES = [
    'height',
    'width',
    'depth',
    'aspect_ratio',
    'density',
    'volume',
    'distancefromlidar'
]

def extract_features(cluster_points: np.ndarray) -> np.ndarray:
    """
    Extracts geometric features from a point cloud cluster (N x 3 or N x 4 array).
    Returns a 7-element float32 numpy array.
    """
    if cluster_points is None or len(cluster_points) < 3:
        return None

    xyz = cluster_points[:, :3]

    sigma_x = float(xyz[:, 0].std())
    sigma_y = float(xyz[:, 1].std())

    height = float(xyz[:, 2].max() - xyz[:, 2].min())  # Height of the cluster
    width = float(max(sigma_x, sigma_y))               # Width of the cluster
    depth = float(min(sigma_x, sigma_y))               # Depth of the cluster
    aspect_ratio = float(height / (width + 1e-6))      # Height to width ratio

    num_points = len(xyz)                              # Number of points in the cluster
    volume = float((height + 1e-6) * (width + 1e-6) * (depth + 1e-6))
    density = float(num_points / volume)               # Point per volume

    center = xyz.mean(axis=0)
    distancefromlidar = float(np.linalg.norm(center))  # Centroid distance from LiDAR origin (0,0,0)

    return np.array([
        height, width, depth, aspect_ratio,
        density, volume, distancefromlidar
    ], dtype=np.float32)