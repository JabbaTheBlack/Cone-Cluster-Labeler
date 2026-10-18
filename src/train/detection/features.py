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

# The 12 features of the July 2026 model (commit 3e1c34b1) plus distancefromlidar.
# width/depth use the adaptive definition of commit d6992c35 (larger/smaller of the x/y
# standard deviations) instead of 3e1c34b1's axis-aligned x/y extents, so an object's
# orientation relative to the car does not change which axis counts as its width.
EXTENDED_FEATURE_NAMES = [
    'height',
    'width',
    'depth',
    'aspect_ratio',
    'num_points',
    'density',
    'compactness',
    'intensity_mean',
    'intensity_std',
    'linearity',
    'planarity',
    'volume',
    'distancefromlidar'
]

def extract_extended_features(cluster_points: np.ndarray) -> np.ndarray:
    """
    Extracts the 13 EXTENDED_FEATURE_NAMES features from an N x 4 (x, y, z, intensity) cluster.
    Returns a 13-element float32 numpy array.
    """
    if cluster_points is None or len(cluster_points) < 3:
        return None

    xyz = cluster_points[:, :3]
    intensity = cluster_points[:, 3]

    sigma_x = float(xyz[:, 0].std())
    sigma_y = float(xyz[:, 1].std())

    height = float(xyz[:, 2].max() - xyz[:, 2].min())
    width = float(max(sigma_x, sigma_y))
    depth = float(min(sigma_x, sigma_y))
    aspect_ratio = float(height / (width + 1e-6))

    num_points = len(xyz)
    volume = float((height + 1e-6) * (width + 1e-6) * (depth + 1e-6))
    density = float(num_points / volume)

    center = xyz.mean(axis=0)
    distances = np.linalg.norm(xyz - center, axis=1)
    compactness = float(np.std(distances) / (np.mean(distances) + 1e-6))  # Spread of point-to-centroid distances

    intensity_mean = float(intensity.mean())
    intensity_std = float(intensity.std())

    eigenvalues = np.linalg.eigvalsh(np.cov(xyz.T))                   # Ascending order
    linearity = float((eigenvalues[2] - eigenvalues[1]) / (eigenvalues[2] + 1e-6))
    planarity = float((eigenvalues[1] - eigenvalues[0]) / (eigenvalues[2] + 1e-6))

    distancefromlidar = float(np.linalg.norm(center))

    return np.array([
        height, width, depth, aspect_ratio, num_points,
        density, compactness, intensity_mean, intensity_std,
        linearity, planarity, volume, distancefromlidar
    ], dtype=np.float32)
