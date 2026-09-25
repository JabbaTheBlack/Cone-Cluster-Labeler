# import numpy as np
# import pandas as pd

# FEATURE_NAMES = [
#     'norm_avg_i', 'norm_std_i', 'v_grad', 'height', 'aspect_ratio',
#     'contrast', 'reflective_point_pct', 'contrast_bot_mid',
#     'contrast_mid_top', 'contrast_bot_top', 'distance', 'num_points',
#     'skew_i', 'q10', 'q50', 'q90'
# ]

# def extract_extended_features(points):
#     """Extracts enhanced 16-feature representation from point cloud clusters."""
#     if points is None or len(points) < 3:
#         return None

#     xyz = points[:, :3]
#     intensity = points[:, 3]
#     z = xyz[:, 2]

#     num_points = float(len(points))
#     distance = float(np.linalg.norm(xyz.mean(axis=0)))
#     dist_sq = (distance ** 2) + 1e-6

#     height = float(z.max() - z.min())
#     width = float(max(xyz[:, 0].std(), xyz[:, 1].std(), 1e-6))
#     aspect_ratio = height / width

#     avg_i = float(intensity.mean())
#     std_i = float(intensity.std())
#     skew_i = float(pd.Series(intensity).skew()) if len(points) > 3 else 0.0

#     norm_avg_i = np.log1p(avg_i * dist_sq)
#     norm_std_i = np.log1p(std_i * dist_sq)

#     i_lo, i_hi = np.percentile(intensity, [5, 95])
#     contrast = (i_hi - i_lo) / (avg_i + 1e-6)
#     reflective_point_pct = float(np.mean(intensity > (avg_i * 1.5)))

#     q10, q50, q90 = np.percentile(intensity, [10, 50, 90])

#     z_min, z_max = z.min(), z.max()
#     z_range = max(z_max - z_min, 1e-6)
#     bot_mask = z < (z_min + 0.33 * z_range)
#     top_mask = z > (z_min + 0.67 * z_range)
#     mid_mask = ~bot_mask & ~top_mask

#     bot_i = intensity[bot_mask].mean() if bot_mask.sum() > 0 else avg_i
#     mid_i = intensity[mid_mask].mean() if mid_mask.sum() > 0 else avg_i
#     top_i = intensity[top_mask].mean() if top_mask.sum() > 0 else avg_i

#     contrast_bot_mid = (mid_i - bot_i) / (avg_i + 1e-6)
#     contrast_mid_top = (top_i - mid_i) / (avg_i + 1e-6)
#     contrast_bot_top = (top_i - bot_i) / (avg_i + 1e-6)

#     z_centered = z - z.mean()
#     i_centered = intensity - avg_i
#     v_grad = np.sum(z_centered * i_centered) / (np.sum(z_centered ** 2) + 1e-6)

#     return np.array([
#         norm_avg_i, norm_std_i, v_grad, height, aspect_ratio,
#         contrast, reflective_point_pct, contrast_bot_mid,
#         contrast_mid_top, contrast_bot_top, distance, num_points,
#         skew_i, q10, q50, q90
#     ], dtype=np.float32)

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