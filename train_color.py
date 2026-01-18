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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

sns.set(style='whitegrid', context='talk')

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
#
# FEATURE DOCUMENTATION FOR CONE COLOR CLASSIFICATION (Orange, Blue, Yellow)
# =========================================================================
#
# Since we only have LiDAR data (no camera), we rely on two main signals:
#   1. INTENSITY: Different colored surfaces reflect laser light differently
#   2. GEOMETRY: Shape characteristics that may vary with viewing angle/distance
#
# KEY INSIGHT: LiDAR intensity correlates with surface reflectivity
#   - Yellow cones: HIGHEST reflectivity (bright, reflects most light)
#   - Orange cones: MEDIUM-HIGH reflectivity
#   - Blue cones: LOWEST reflectivity (darker, absorbs more light)
#
# ============================================================================
# FEATURE LIST (25 total features)
# ============================================================================
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ GEOMETRY FEATURES (8 features) - Indices 0-7                            │
# ├─────────────────────────────────────────────────────────────────────────┤
# │                                                                         │
# │ 0. height (z.ptp)                                                       │
# │    - Definition: Peak-to-peak height of the point cloud (max_z - min_z) │
# │    - Range: ~0.2m to 0.5m for typical cones                             │
# │    - Why: All cones should have similar height (~0.325m for small,      │
# │           ~0.505m for large). Outliers may indicate noise/partial scans │
# │    - Color relevance: LOW (same size for all colors)                    │
# │                                                                         │
# │ 1. width (max of x_std, y_std)                                          │
# │    - Definition: Lateral spread of points (standard deviation)          │
# │    - Range: Depends on cone base size and scan angle                    │
# │    - Why: Wider at base, narrower at top. Helps filter non-cone objects │
# │    - Color relevance: LOW                                               │
# │                                                                         │
# │ 2. aspect_ratio (height / width)                                        │
# │    - Definition: Ratio of vertical to horizontal extent                 │
# │    - Range: Typically 2-6 for cones (taller than wide)                  │
# │    - Why: Cones have characteristic tall-narrow shape                   │
# │    - Color relevance: LOW                                               │
# │                                                                         │
# │ 3. point_count                                                          │
# │    - Definition: Total number of LiDAR points in the cluster            │
# │    - Range: 5 to 500+ depending on distance                             │
# │    - Why: More points = closer cone = more reliable intensity readings  │
# │    - Color relevance: INDIRECT (affects measurement confidence)         │
# │                                                                         │
# │ 4. point_density (points / volume)                                      │
# │    - Definition: Points per cubic meter of bounding volume              │
# │    - Range: Varies with LiDAR resolution and distance                   │
# │    - Why: Higher density = denser scan = more reliable features         │
# │    - Color relevance: LOW                                               │
# │                                                                         │
# │ 5. linearity ((λ1 - λ2) / λ1)                                           │
# │    - Definition: PCA eigenvalue ratio measuring 1D structure            │
# │    - Range: 0 (sphere) to 1 (perfect line)                              │
# │    - Why: Cones should NOT be linear; high linearity = edge artifact    │
# │    - Color relevance: LOW                                               │
# │                                                                         │
# │ 6. planarity ((λ2 - λ3) / λ1)                                           │
# │    - Definition: PCA eigenvalue ratio measuring 2D structure            │
# │    - Range: 0 to 1                                                      │
# │    - Why: Partial cone scans may appear planar (one side visible)       │
# │    - Color relevance: LOW                                               │
# │                                                                         │
# │ 7. sphericity (λ3 / λ1)                                                 │
# │    - Definition: PCA eigenvalue ratio measuring 3D spread               │
# │    - Range: 0 to 1 (1 = perfect sphere)                                 │
# │    - Why: Full cone scans have moderate sphericity                      │
# │    - Color relevance: LOW                                               │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ INTENSITY STATISTICS (12 features) - Indices 8-19                       │
# │ *** MOST IMPORTANT FOR COLOR CLASSIFICATION ***                         │
# ├─────────────────────────────────────────────────────────────────────────┤
# │                                                                         │
# │ 8. avg_intensity (mean)                                                 │
# │    - Definition: Mean intensity value across all points                 │
# │    - Range: 0-255 (typical) or 0-1 (normalized)                         │
# │    - Why: PRIMARY COLOR DISCRIMINATOR                                   │
# │           Yellow > Orange > Blue (expected ordering)                    │
# │    - Color relevance: ★★★★★ CRITICAL                                   │
# │    - Caveat: Decreases with distance (inverse square law)               │
# │                                                                         │
# │ 9. std_intensity (standard deviation)                                   │
# │    - Definition: Spread of intensity values around the mean             │
# │    - Range: 0 to ~50                                                    │
# │    - Why: Uniform surfaces have low std; multi-surface have high        │
# │           Cones with stripes (orange) may have higher std               │
# │    - Color relevance: ★★★☆☆ MODERATE                                   │
# │                                                                         │
# │ 10. median_intensity                                                    │
# │    - Definition: 50th percentile of intensity                           │
# │    - Range: Same as avg_intensity                                       │
# │    - Why: More robust to outliers than mean                             │
# │           Useful when a few points have extreme values                  │
# │    - Color relevance: ★★★★☆ HIGH                                       │
# │                                                                         │
# │ 11. intensity_skewness                                                  │
# │    - Definition: Asymmetry of intensity distribution                    │
# │    - Formula: E[(X-μ)³] / σ³                                            │
# │    - Range: Typically -2 to +2                                          │
# │    - Why: Positive skew = tail toward high values (few bright spots)    │
# │           Negative skew = tail toward low values                        │
# │           Different cone materials may have characteristic skew         │
# │    - Color relevance: ★★★☆☆ MODERATE                                   │
# │                                                                         │
# │ 12. intensity_kurtosis                                                  │
# │    - Definition: "Tailedness" of intensity distribution                 │
# │    - Formula: E[(X-μ)⁴] / σ⁴ - 3 (excess kurtosis)                      │
# │    - Range: Typically -2 to +10                                         │
# │    - Why: High kurtosis = sharp peak with heavy tails                   │
# │           Low kurtosis = flat distribution                              │
# │           Different surface textures create different patterns          │
# │    - Color relevance: ★★☆☆☆ LOW-MODERATE                               │
# │                                                                         │
# │ 13. p10 (10th percentile)                                               │
# │    - Definition: Intensity value below which 10% of points fall         │
# │    - Range: Lower bound of intensity distribution                       │
# │    - Why: Captures the "floor" of reflectivity                          │
# │           Blue cones should have lower p10 than yellow                  │
# │    - Color relevance: ★★★★☆ HIGH                                       │
# │                                                                         │
# │ 14. p90 (90th percentile)                                               │
# │    - Definition: Intensity value below which 90% of points fall         │
# │    - Range: Upper bound of intensity distribution                       │
# │    - Why: Captures the "ceiling" of reflectivity                        │
# │           Yellow cones should have higher p90                           │
# │    - Color relevance: ★★★★☆ HIGH                                       │
# │                                                                         │
# │ 15. iqr (interquartile range = p75 - p25)                               │
# │    - Definition: Spread of middle 50% of intensity values               │
# │    - Range: 0 to ~100                                                   │
# │    - Why: Robust measure of variability (ignores outliers)              │
# │           Uniform color = low IQR; striped/textured = high IQR          │
# │    - Color relevance: ★★★☆☆ MODERATE                                   │
# │                                                                         │
# │ 16. intensity_cv (coefficient of variation = std/mean)                  │
# │    - Definition: Normalized measure of dispersion                       │
# │    - Range: 0 to 1+ (0 = no variation)                                  │
# │    - Why: DISTANCE-INVARIANT measure of intensity spread                │
# │           A cone far away has same CV as one nearby                     │
# │    - Color relevance: ★★★☆☆ MODERATE                                   │
# │                                                                         │
# │ 17. contrast ((max - min) / mean)                                       │
# │    - Definition: Normalized range of intensity                          │
# │    - Range: 0 to 10+ (higher = more contrast)                           │
# │    - Why: DISTANCE-INVARIANT; captures dynamic range                    │
# │           High contrast may indicate mixed surfaces or noise            │
# │    - Color relevance: ★★☆☆☆ LOW-MODERATE                               │
# │                                                                         │
# │ 18. reflective_ratio (fraction of points > 1.5 × mean)                  │
# │    - Definition: Proportion of "highly reflective" points               │
# │    - Range: 0 to 1                                                      │
# │    - Why: Yellow cones may have more high-reflectivity points           │
# │           Captures the upper tail of the distribution                   │
# │    - Color relevance: ★★★☆☆ MODERATE                                   │
# │                                                                         │
# │ 19. high_intensity_ratio (fraction of points > p75)                     │
# │    - Definition: Proportion of points in top quartile                   │
# │    - Range: Always ~0.25 by definition (unless tied values)             │
# │    - Why: Combined with other percentiles, characterizes shape          │
# │    - Color relevance: ★★☆☆☆ LOW                                        │
# │    - NOTE: Consider removing (always ~0.25)                             │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ HEIGHT-INTENSITY CORRELATION (4 features) - Indices 20-23               │
# │ *** CAPTURES VERTICAL INTENSITY PATTERNS ***                            │
# ├─────────────────────────────────────────────────────────────────────────┤
# │                                                                         │
# │ 20. v_grad (vertical intensity gradient)                                │
# │    - Definition: Slope of intensity vs height (linear regression)       │
# │    - Formula: Σ(z-z̄)(I-Ī) / Σ(z-z̄)²                                    │
# │    - Range: Negative to positive                                        │
# │    - Why: Some cones have intensity patterns that vary with height      │
# │           Striped cones, dirty bottoms, etc.                            │
# │    - Color relevance: ★★★☆☆ MODERATE                                   │
# │                                                                         │
# │ 21. height_intensity_corr (Pearson correlation)                         │
# │    - Definition: Correlation coefficient between z and intensity        │
# │    - Range: -1 to +1                                                    │
# │    - Why: Normalized version of v_grad                                  │
# │           +1 = brighter at top, -1 = brighter at bottom                 │
# │    - Color relevance: ★★★☆☆ MODERATE                                   │
# │                                                                         │
# │ 22. top_bot_ratio (top_intensity / bottom_intensity)                    │
# │    - Definition: Ratio of mean intensity in upper vs lower half         │
# │    - Range: 0.5 to 2.0 typically                                        │
# │    - Why: Detects cones with dirty bases or worn tops                   │
# │           Different cone types may have different wear patterns         │
# │    - Color relevance: ★★☆☆☆ LOW-MODERATE                               │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ DISTANCE-NORMALIZED (1 feature) - Index 24                              │
# │ *** COMPENSATES FOR RANGE-DEPENDENT INTENSITY ***                       │
# ├─────────────────────────────────────────────────────────────────────────┤
# │                                                                         │
# │ 23. intensity_per_distance (avg_intensity × distance)                   │
# │    - Definition: Intensity scaled by distance from sensor               │
# │    - Formula: mean(intensity) × mean(√(x² + y²))                        │
# │    - Range: Varies                                                      │
# │    - Why: LiDAR intensity follows inverse-square law (I ∝ 1/d²)         │
# │           Multiplying by distance partially compensates                 │
# │           A yellow cone at 10m should have similar value as at 5m       │
# │    - Color relevance: ★★★★☆ HIGH (distance-invariant intensity)        │
# │    - NOTE: Consider using d² for true inverse-square compensation       │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ============================================================================
# FEATURE IMPORTANCE FOR COLOR CLASSIFICATION (Expected)
# ============================================================================
#
#  CRITICAL (★★★★★):
#    - avg_intensity, median_intensity
#
#  HIGH (★★★★☆):
#    - p10, p90, intensity_per_distance
#
#  MODERATE (★★★☆☆):
#    - std_intensity, intensity_skewness, iqr, intensity_cv
#    - reflective_ratio, v_grad, height_intensity_corr
#
#  LOW (★★☆☆☆ or ★☆☆☆☆):
#    - All geometry features (same shape for all colors)
#    - contrast, intensity_kurtosis, high_intensity_ratio
#
# ============================================================================
# TIPS FOR IMPROVING MODEL PERFORMANCE
# ============================================================================
#
#  1. DATA COLLECTION: Ensure balanced samples of each color at various
#     distances (5m, 10m, 15m, 20m) to learn distance-invariant patterns
#
#  2. FEATURE SELECTION: After training, check feature_importances_ to
#     see which features the Random Forest actually uses. Remove useless ones.
#
#  3. DISTANCE BINNING: Consider training separate models for near/far cones
#     or adding distance as an explicit feature
#
#  4. INTENSITY CALIBRATION: If your LiDAR provides calibrated intensity
#     (reflectivity), the raw values may already be distance-compensated
#
#  5. AUGMENTATION: Add noise to training data to improve robustness
#
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

    # ==================== GEOMETRY FEATURES ====================
    height = xyz[:, 2].ptp()
    width = max(xyz[:, 0].std(), xyz[:, 1].std(), 1e-6)
    aspect_ratio = height / width
    point_count = len(points)
    distance_from_lidar = np.linalg.norm(xyz.mean(axis=0))
    
    # Point density (points per volume)
    volume = np.prod(xyz.ptp(axis=0) + 1e-6)
    point_density = point_count / volume
    
    # PCA-based shape features (eigenvalue ratios)
    if len(xyz) >= 3:
        cov = np.cov(xyz.T)
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
        planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
        sphericity = eigenvalues[2] / eigenvalues[0]
    else:
        linearity = planarity = sphericity = 0.0

    # ==================== INTENSITY FEATURES ====================
    # Basic statistics
    avg_intensity = intensity.mean()
    std_intensity = intensity.std()
    median_intensity = np.median(intensity)
    
    # Distribution shape (key for color discrimination!)
    intensity_skewness = ((intensity - avg_intensity) ** 3).mean() / (std_intensity ** 3 + 1e-6)
    intensity_kurtosis = ((intensity - avg_intensity) ** 4).mean() / (std_intensity ** 4 + 1e-6) - 3
    
    # Percentile-based features (robust to outliers)
    p10 = np.percentile(intensity, 10)
    p25 = np.percentile(intensity, 25)
    p75 = np.percentile(intensity, 75)
    p90 = np.percentile(intensity, 90)
    iqr = p75 - p25
    intensity_range = intensity.max() - intensity.min()
    
    # Reflectivity ratios
    high_intensity_ratio = np.mean(intensity > p75)
    low_intensity_ratio = np.mean(intensity < p25)
    contrast = intensity_range / (avg_intensity + 1e-6)
    reflective_ratio = np.mean(intensity > avg_intensity * 1.5)
    
    # Coefficient of variation (normalized spread)
    intensity_cv = std_intensity / (avg_intensity + 1e-6)

    # ==================== HEIGHT-INTENSITY CORRELATION ====================
    z = xyz[:, 2]
    z_centered = z - z.mean()
    i_centered = intensity - avg_intensity
    
    # Vertical intensity gradient
    v_grad = np.sum(z_centered * i_centered) / (np.sum(z_centered ** 2) + 1e-6)
    
    # Correlation coefficient
    height_intensity_corr = np.corrcoef(z, intensity)[0, 1] if len(z) > 2 else 0.0
    if np.isnan(height_intensity_corr):
        height_intensity_corr = 0.0
    
    # Top vs bottom intensity (cones have different patterns)
    z_mid = (z.max() + z.min()) / 2
    top_mask = z > z_mid
    bot_mask = z <= z_mid
    top_intensity = intensity[top_mask].mean() if top_mask.sum() > 0 else avg_intensity
    bot_intensity = intensity[bot_mask].mean() if bot_mask.sum() > 0 else avg_intensity
    top_bot_ratio = top_intensity / (bot_intensity + 1e-6)

    # ==================== DISTANCE-BASED NORMALIZATION ====================
    # Intensity drops with distance - normalize for it
    distance = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2).mean()
    intensity_per_distance = avg_intensity * (distance + 1e-6)  # Compensate for range

    return np.array([
        # Geometry (7)
        height,
        width,
        aspect_ratio,
        point_count,
        point_density,
        linearity,
        planarity,
        sphericity,
        #distance_from_lidar, for the start to avoid overfitting
        # Intensity statistics (12)
        avg_intensity,
        std_intensity,
        median_intensity,
        intensity_skewness,
        intensity_kurtosis,
        p10,
        p90,
        iqr,
        intensity_cv,
        contrast,
        reflective_ratio,
        high_intensity_ratio,
        # Height-intensity (4)
        v_grad,
        height_intensity_corr,
        top_bot_ratio,
        # Distance-normalized (1)
        intensity_per_distance,
    ], dtype=np.float32)


class MultiTrackDatasetBuilder:
    """Combines labeled clusters from multiple track folders into one dataset."""
    
    def __init__(self, base_dataset_path):
        """
        Args:
            base_dataset_path: Path to Dataset folder (contains Acceleration/, Skidpad/, Autocross/)
        """
        self.base_path = Path(base_dataset_path).expanduser() / "Processed" / "Color"
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
    
    def build_dataset(self):
        """Load all PCD files and extract features."""
        VALID_COLORS = {'orange', 'blue', 'yellow'}
        self.label_encoder = LabelEncoder()
        self.X = []
        self.y_raw = []  # String labels
        skipped = 0
        
        print('\n📦 Building dataset from labeled clusters...')
        
        for track_name, track_info in self.tracks.items():
            track_path = track_info['path']
            labels = track_info['labels']
            
            for cluster_file, label_data in tqdm(labels.items(), desc=f'  {track_name}'):
                # Handle nested structure: {"is_cone": true, "color": "orange"}
                if isinstance(label_data, dict):
                    if not label_data.get('is_cone', True):
                        skipped += 1
                        continue
                    color_lower = label_data.get('color', '').lower()
                else:
                    # Simple structure: "orange"
                    color_lower = label_data.lower()
                
                if color_lower not in VALID_COLORS:
                    skipped += 1
                    continue
                
                pcd_path = track_path / cluster_file
                if not pcd_path.exists():
                    skipped += 1
                    continue
                
                points = load_pcd_binary(pcd_path)
                features = extract_features(points)
                
                if features is None:
                    skipped += 1
                    continue
                
                self.X.append(features)
                self.y_raw.append(color_lower)
        
        self.X = np.array(self.X)
        self.y = self.label_encoder.fit_transform(self.y_raw)
        
        print(f'\n✅ Dataset built: {len(self.X)} samples, {skipped} skipped')
        print(f'  Classes: {list(self.label_encoder.classes_)}')
        for idx, name in enumerate(self.label_encoder.classes_):
            print(f'  {name}: {np.sum(self.y == idx)}')
        
        return self.X, self.y, self.label_encoder

class RandomForestConeDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.best_params = None
        self.label_encoder = None  # Will be set during training
        #self.feature_names = ['height', 'width', 'depth', 'aspect_ratio',
        #                'density', 'volume', 'distance_from_lidar']
    
    def cross_validate(self, X_scaled, y, cv_folds=5):
        """5-fold cross-validation on full dataset."""
        from sklearn.model_selection import cross_val_score, cross_validate
        print(f'\n🔍 {cv_folds}-Fold Cross-Validation...')
        
        rf_temp = RandomForestClassifier(**self.best_params, random_state=42, n_jobs=-1)
        cv_results = cross_validate(rf_temp, X_scaled, y, cv=cv_folds,
                                  scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
                                  return_train_score=True)
        
        print(f'  CV F1 Macro:  {cv_results["test_f1_macro"].mean():.4f} ± {cv_results["test_f1_macro"].std():.4f}')
        print(f'  CV Accuracy:  {cv_results["test_accuracy"].mean():.4f} ± {cv_results["test_accuracy"].std():.4f}')
        print(f'  CV Precision: {cv_results["test_precision_macro"].mean():.4f} ± {cv_results["test_precision_macro"].std():.4f}')
        print(f'  CV Recall:    {cv_results["test_recall_macro"].mean():.4f} ± {cv_results["test_recall_macro"].std():.4f}')
        print(f'  Train F1:     {cv_results["train_f1_macro"].mean():.4f} (overfitting check)')
        
        return cv_results

    def gridsearch(self, X_train, y_train):
        
        # param_grid = {
        #     'n_estimators': list(range(10, 110, 10)),
        #     'max_depth': [10, 15, 20, 25, 30, None],
        #     'min_samples_split': [2, 5, 10, 15, 20],
        #     'min_samples_leaf': [1, 2, 4, 5, 6, 10],
        #     'max_features': ['sqrt', 'log2']
        # }


        # rf = RandomForestClassifier(random_state=42)

        # # GridSearchCV
        # grid_search = GridSearchCV(
        #     estimator=rf, 
        #     param_grid=param_grid,
        #     cv=5,   
        #     scoring='f1',  
        #     n_jobs=-1,
        #     verbose=0
        # )

        # grid_search.fit(X_train, y_train)

        # print(f'\n✓ GridSearch Complete!')
        # print(f'  Best F1 Score: {grid_search.best_score_:.4f}')
        # print(f'  Best Params: {grid_search.best_params_}')

        # self.best_params = grid_search.best_params_
        # self.model = grid_search.best_estimator_

        # return self.model

        rf = RandomForestClassifier(random_state=42)
    
        # Phase 1: COARSE (higher estimators + your full param space)
        print('🔍 Phase 1: Coarse search (100-400 estimators)...')
        coarse_grid = {
            'n_estimators': [10, 50 ,100, 150],
            'max_depth': [5, 10, 15, 25, None],
            'min_samples_split': [2, 5, 15],
            'min_samples_leaf': [1, 2, 6],
            'max_features': ['sqrt', 'log2']
        }
        
        coarse_search = GridSearchCV(rf, coarse_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
        coarse_search.fit(X_train, y_train)
        best_coarse = coarse_search.best_params_
        print(f'  Coarse best F1: {coarse_search.best_score_:.4f} → {best_coarse}')
        
        # Phase 2: MEDIUM around coarse winner (±30 range)
        print('🔍 Phase 2: Medium refinement...')
        n_est_start = max(10, best_coarse['n_estimators'] - 30)
        n_est_end = min(450, best_coarse['n_estimators'] + 31)
        med_grid = {
            'n_estimators': list(range(n_est_start, n_est_end, 10)),
            'max_depth': [best_coarse['max_depth']] if best_coarse['max_depth'] is not None else [None, 15, 25],
            'min_samples_split': [best_coarse['min_samples_split']],
            'min_samples_leaf': [best_coarse['min_samples_leaf']],
            'max_features': ['sqrt', 'log2']
        }
        
        med_search = GridSearchCV(rf, med_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
        med_search.fit(X_train, y_train)
        best_med = med_search.best_params_
        print(f'  Medium best F1: {med_search.best_score_:.4f} → {best_med}')
        
        # Phase 3: FINE around medium winner (±15 range)
        print('🔍 Phase 3: Fine tuning...')
        n_est_fine_start = max(10, best_med['n_estimators'] - 15)
        n_est_fine_end = min(450, best_med['n_estimators'] + 16)
        fine_grid = {
            'n_estimators': list(range(n_est_fine_start, n_est_fine_end, 5)),
            'max_depth': [None, 10, 15, 20, 25, 30] if best_med['max_depth'] is None else 
                        list(range(max(5, best_med['max_depth']-5), min(31, best_med['max_depth']+6))),
            'min_samples_split': [max(2, best_med['min_samples_split']-3), 
                                best_med['min_samples_split'], 
                                min(20, best_med['min_samples_split']+4)],
            'min_samples_leaf': [max(1, best_med['min_samples_leaf']-2), 
                                best_med['min_samples_leaf'], 
                                min(10, best_med['min_samples_leaf']+3)],
            'max_features': ['sqrt', 'log2']
        }
        
        fine_search = GridSearchCV(rf, fine_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
        fine_search.fit(X_train, y_train)
        
        print(f'\n✓ Progressive GridSearch Complete!')
        print(f'  Final Best F1: {fine_search.best_score_:.4f}')
        print(f'  Final Best Params: {fine_search.best_params_}')
        
        self.best_params = fine_search.best_params_
        self.model = fine_search.best_estimator_
        return self.model

    def train(self, X, y, use_gridsearch=True):
        """Train model."""
        # Split data training: 80% test: 20%
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f'\n[Random Forest Training]')
        print(f'  Train: {len(X_train)} | Test: {len(X_test)}')
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Starting gridsearch...")
        self.gridsearch(X_train_scaled, y_train)

        X_full_scaled = self.scaler.transform(X)
        print("Cross validating...")
        cv_results = self.cross_validate(X_full_scaled, y)

        self.plot_feature_correlation(X)

        # Evaluate
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
        test_f1_score = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        
        print(f'\n✓ Evaluation Results:')
        print(f'  Train Acc: {train_accuracy:.2%}')
        print(f'  Test Acc:   {test_acc:.2%}')
        print(f'  Precision: {test_precision:.2%}')
        print(f'  Recall:    {test_recall:.2%}')
        print(f'  F1 Score:  {test_f1_score:.2%}')
        
        logSaver = LogSaver(log_dir='logs')
        logSaver.save(X_train_scaled, X_test_scaled, y_train, y_test,
                     train_accuracy, test_acc, test_precision, test_recall, test_f1_score,
                     self.feature_names, self.model.feature_importances_, self.best_params, cv_results)

        self.visualize_confusion_matrix(y_test, y_test_pred)
        self.visualize_feature_importances()
            
    def plot_feature_correlation(self, X):
        """Plot feature correlation matrix."""
        feat_df = pd.DataFrame(X, columns=self.feature_names)
        corr = feat_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        script_dir = Path(__file__).parent
        (script_dir / 'figures').mkdir(parents=True, exist_ok=True)
        plt.savefig(script_dir / 'figures' / 'color' / 'feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✓ Saved: figures/detection/feature_correlation.png')


    def visualize_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        
        # Use label_encoder to get color names, fallback to numbers
        if self.label_encoder is not None:
            labels = list(self.label_encoder.classes_)
        else:
            labels = [str(i) for i in range(cm.shape[0])]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, 
                   yticklabels=labels)
        plt.title('Confusion Matrix - Cone Color Classification')
        plt.ylabel('True Color')
        plt.xlabel('Predicted Color')
        plt.tight_layout()

        script_dir = Path(__file__).parent
        (script_dir / 'figures' / 'color').mkdir(parents=True, exist_ok=True)
        plt.savefig(script_dir / 'figures' / 'color' / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✓ Saved: figures/color/confusion_matrix.png')
        

    def visualize_feature_importances(self):
        importances = self.model.feature_importances_
        feat_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feat_importance_df, x='importance', y='feature', hue='feature', palette='viridis', legend=False)
        plt.title('Feature Importances')
        plt.xlabel('Importance Score')
        plt.tight_layout()

        script_dir = Path(__file__).parent
        (script_dir / 'figures').mkdir(parents=True, exist_ok=True)
        plt.savefig(script_dir / 'figures' / 'color' / 'feature_importances.png', dpi=300, bbox_inches='tight')
        print('✓ Saved: figures/color/feature_importances.png')

    def save(self, path='cone_detector_rf.pkl'):
        """Save model."""
        data = {
            'scaler': self.scaler,
            'model': self.model,
            'best_params': self.best_params
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f'\n✓ Saved to {path}')
 
    def save_cpp_ready(self, path='cone_detector.bin'):
        """Save model in raw binary format for C++"""
        import struct
        
        scaler_mean = self.scaler.mean_.astype(np.float32)
        scaler_std = self.scaler.scale_.astype(np.float32)
        n_features = len(scaler_mean)  # 7
        
        print(f'Saving {n_features}-feature model to {path}')
        
        with open(path, 'wb') as f:
            # HEADER: n_features (int32)
            f.write(struct.pack('i', n_features))
            
            # Scaler mean (7 floats)
            f.write(scaler_mean.tobytes())
            
            # Scaler std (7 floats)  
            f.write(scaler_std.tobytes())
            
            # n_trees (int32)
            f.write(struct.pack('i', self.model.n_estimators))
            
            # Trees
            for tree_obj in self.model.estimators_:
                tree = tree_obj.tree_
                f.write(struct.pack('i', tree.node_count))
                for i in range(tree.node_count):
                    f.write(struct.pack('i', int(tree.feature[i])))
                    f.write(struct.pack('f', float(tree.threshold[i])))
                    f.write(struct.pack('i', int(tree.children_left[i])))
                    f.write(struct.pack('i', int(tree.children_right[i])))
                    f.write(struct.pack('f', float(tree.value[i][0][0])))  # non-cone
                    f.write(struct.pack('f', float(tree.value[i][0][1])))  # cone
            
        print(f'✓ C++ Ready ({n_features} feats): {path}')

    def load(self, path='cone_detector_rf.pkl'):
        """Load model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.scaler = data['scaler']
        self.model = data['model']
        self.best_params = data.get('best_params')
        print(f'✓ Loaded from {path}')
    
    def predict(self, X):
        """Predict on new data."""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        return predictions, probabilities

# ============================================================================
# Log Saver
# ============================================================================
class LogSaver:
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / 'color' / 'training_log.txt'
    
    def save(self, X_train, X_test, y_train, y_test, train_acc, test_acc, precision, recall, f1, 
         features, importances, best_params=None, cv_results=None):
        results = {
            'dataset_size': len(X_train) + len(X_test),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'cones_total': int(np.sum(y_train) + np.sum(y_test)),
            'non_cones_total': int(len(y_train) + len(X_test) - np.sum(y_train) - np.sum(y_test)),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'best_params': best_params,
            'cv_f1_mean': float(cv_results['test_f1_macro'].mean()) if cv_results is not None else None,
            'cv_f1_std': float(cv_results['test_f1_macro'].std()) if cv_results is not None else None,
            'feature_importances': dict(zip(features, importances.tolist()))
        }
        
        with open(self.log_file, 'a') as f:
            json.dump(results, f)
            f.write('\n\n')
        
        print(f'Training log saved: {self.log_file}')
        return self.log_file


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Random Forest for cone color classification")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to Dataset folder (the one containing Processed/Color)"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    print(f"\n🚀 Dataset root: {dataset_path}\n")

    builder = MultiTrackDatasetBuilder(dataset_path)

    print(f"\n✅ Tracks found: {len(builder.tracks)}\n")
    for name, data in builder.tracks.items():
        print(f"  {name}: {data['label_count']} labels")
    
    # Build dataset
    X, y, label_encoder = builder.build_dataset()
    
    if len(X) == 0:
        print("❌ No valid samples!")
        return
    
    # Train
    detector = RandomForestConeDetector()
    detector.label_encoder = label_encoder  # Set for confusion matrix labels
    detector.feature_names = [
        'height', 'width', 'aspect_ratio', 'point_count', 'point_density',
        'linearity', 'planarity', 'sphericity', 
        'avg_intensity', 'std_intensity', 'median_intensity',
        'intensity_skewness', 'intensity_kurtosis',
        'p10', 'p90', 'iqr', 'intensity_cv', 'contrast',
        'reflective_ratio', 'high_intensity_ratio',
        'v_grad', 'height_intensity_corr', 'top_bot_ratio',
        'intensity_per_distance',
        #'distance_from_lidar'
    ]
    detector.train(X, y)
    detector.save('models/color_classifier_rf.pkl')


if __name__ == "__main__":
    main()