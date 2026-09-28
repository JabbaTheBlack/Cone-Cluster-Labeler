import numpy as np
import pandas as pd

FEATURE_NAMES = [
    'norm_avg_i', 'norm_std_i', 'v_grad', 'height', 'aspect_ratio',
    'contrast', 'reflective_point_pct', 'contrast_bot_mid',
    'contrast_mid_top', 'contrast_bot_top', 'distance', 'num_points',
    'skew_i', 'q10', 'q50', 'q90', 'reflectivity_ratio'
]

def extract_extended_features(points, eta=2.0, r_ref=10.0):
    """
    Extracts enhanced 17-feature representation from point cloud clusters.
    Expects points array with shape (N, 4+) -> [X, Y, Z, Intensity (0-255)].
    """
    if points is None or len(points) < 3:
        return None

    xyz = points[:, :3]
    raw_intensity = points[:, 3]
    z = xyz[:, 2]

    num_points = float(len(points))
    distance = float(np.linalg.norm(xyz.mean(axis=0)))
    
    # 1. Physical Range Compensation for Intensity (0.0 to 1.0 scale)
    i_normalized = np.clip(raw_intensity / 255.0, 0.0, 1.0)
    range_factor = (max(distance, 0.5) / r_ref) ** eta
    intensity = i_normalized * range_factor

    # 2. Geometric Features with Clipping Boundaries
    height = float(z.max() - z.min())
    width = float(max(xyz[:, 0].std(), xyz[:, 1].std(), 1e-3))
    aspect_ratio = float(np.clip(height / width, 0.0, 20.0))

    avg_i = float(intensity.mean())
    std_i = float(intensity.std())
    skew_i = float(pd.Series(intensity).skew()) if len(points) > 3 else 0.0

    norm_avg_i = float(np.log1p(avg_i))
    norm_std_i = float(np.log1p(std_i))

    # 3. Percentiles and Contrast
    i_lo, i_hi = np.percentile(intensity, [5, 95])
    contrast = float((i_hi - i_lo) / (avg_i + 1e-4))
    
    # Threshold for highly reflective points
    reflective_point_pct = float(np.mean(i_normalized > 0.8))

    q10, q50, q90 = np.percentile(intensity, [10, 50, 90])

    # 4. Vertical Profile Segmentation
    z_min, z_max = z.min(), z.max()
    z_range = max(z_max - z_min, 1e-4)
    bot_mask = z < (z_min + 0.33 * z_range)
    top_mask = z > (z_min + 0.67 * z_range)
    mid_mask = ~bot_mask & ~top_mask

    bot_i = intensity[bot_mask].mean() if bot_mask.sum() > 0 else avg_i
    mid_i = intensity[mid_mask].mean() if mid_mask.sum() > 0 else avg_i
    top_i = intensity[top_mask].mean() if top_mask.sum() > 0 else avg_i

    contrast_bot_mid = float((mid_i - bot_i) / (avg_i + 1e-4))
    contrast_mid_top = float((top_i - mid_i) / (avg_i + 1e-4))
    contrast_bot_top = float((top_i - bot_i) / (avg_i + 1e-4))

    # Vertical gradient calculation
    z_centered = z - z.mean()
    i_centered = intensity - avg_i
    v_grad = float(np.sum(z_centered * i_centered) / (np.sum(z_centered ** 2) + 1e-4))

    # Ratio of uncompensated to compensated intensity (detects retroreflectors)
    reflectivity_ratio = float(avg_i / (float(i_normalized.mean()) + 1e-4))

    return np.array([
        norm_avg_i, norm_std_i, v_grad, height, aspect_ratio,
        contrast, reflective_point_pct, contrast_bot_mid,
        contrast_mid_top, contrast_bot_top, distance, num_points,
        skew_i, q10, q50, q90, reflectivity_ratio
    ], dtype=np.float32)


# import numpy as np


# FEATURE_NAMES = [
#     "norm_avg_i",
#     "norm_std_i",
#     "contrast_bot_mid",
#     "contrast_mid_top",
#     "q90",
# ]


# def extract_extended_features(points, eta=2.0, r_ref=10.0):
#     """
#     Extract the selected six-feature representation for color classification.

#     Input points:
#         Shape (N, 4+) with columns [x, y, z, intensity].

#     Output feature order:
#         [norm_avg_i, norm_std_i, height,
#          contrast_bot_mid, contrast_mid_top, q90]
#     """
#     if points is None or len(points) < 3:
#         return None

#     points = np.asarray(points, dtype=np.float32)

#     if points.ndim != 2 or points.shape[1] < 4:
#         return None

#     xyz = points[:, :3]
#     raw_intensity = points[:, 3]
#     z = xyz[:, 2]

#     distance = float(np.linalg.norm(xyz.mean(axis=0)))

#     # Raw LiDAR intensity normalized to [0, 1], before range compensation.
#     i_normalized = np.clip(raw_intensity / 255.0, 0.0, 1.0)

#     # Range-compensated intensity.
#     range_factor = (max(distance, 0.5) / r_ref) ** eta
#     intensity = i_normalized * range_factor

#     avg_i = float(intensity.mean())
#     std_i = float(intensity.std())

#     norm_avg_i = float(np.log1p(avg_i))
#     norm_std_i = float(np.log1p(std_i))

#     height = float(z.max() - z.min())

#     q90 = float(np.percentile(intensity, 90))

#     # Vertical profile segmentation.
#     z_min = float(z.min())
#     z_max = float(z.max())
#     z_range = max(z_max - z_min, 1e-4)

#     bot_mask = z < (z_min + 0.33 * z_range)
#     top_mask = z > (z_min + 0.67 * z_range)
#     mid_mask = ~bot_mask & ~top_mask

#     bot_i = float(intensity[bot_mask].mean()) if bot_mask.any() else avg_i
#     mid_i = float(intensity[mid_mask].mean()) if mid_mask.any() else avg_i
#     top_i = float(intensity[top_mask].mean()) if top_mask.any() else avg_i

#     denominator = avg_i + 1e-4

#     contrast_bot_mid = float((mid_i - bot_i) / denominator)
#     contrast_mid_top = float((top_i - mid_i) / denominator)

#     return np.array(
#         [
#             norm_avg_i,
#             norm_std_i,
#             contrast_bot_mid,
#             contrast_mid_top,
#             q90,
#         ],
#         dtype=np.float32,
#     )

SVM_FEATURE_NAMES = [
    "distance_normalized_mean_intensity",
]


def extract_svm_intensity_feature(points, eta=2.0, r_ref=10.0):
    """
    One scalar per cluster for yellow-vs-non-yellow SVM.

    I_normalized = mean(I_raw / 255) * (distance / r_ref)^eta

    With eta=2 and r_ref=10 m:
        I_normalized = mean(I_raw / 255) * (distance / 10)^2
    """
    if points is None or len(points) < 3:
        return None

    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] < 4:
        return None

    xyz = points[:, :3]
    raw_intensity = points[:, 3]

    distance = float(np.linalg.norm(xyz.mean(axis=0)))
    mean_raw_intensity = float(raw_intensity.mean())

    mean_intensity_01 = np.clip(mean_raw_intensity / 255.0, 0.0, 1.0)

    distance_normalized_mean_intensity = float(
        mean_intensity_01
        * (max(distance, 0.5) / r_ref) ** eta
    )

    return np.array(
        [distance_normalized_mean_intensity],
        dtype=np.float32,
    )