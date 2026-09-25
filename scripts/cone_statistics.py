# #!/usr/bin/env python3
# """
# Statistical Analysis of LiDAR Cone Clusters without Pandas Dependency.

# Key Capabilities:
# 1. Distance-based Point Count Estimator: automatic best-fit model search
#    (power law, inverse-square, exponential decay, rational decay) across all
#    Dataset PCDs, selected by AIC (Akaike Information Criterion) so that
#    models with more free parameters aren't unfairly favored over simpler
#    ones just because they achieve a marginally higher R^2.
# 2. Distance-based Intensity Loss Compensation: models baseline average
#    intensity decay per color (also selected via AIC) and produces a
#    distance-compensated average intensity feature normalized to a reference
#    distance.
# 3. Pure NumPy & Standard Library implementation (compatible with NumPy 1.21.5).
# """

# import csv
# import json
# import os
# import re
# import struct
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional, Callable

# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from scipy.optimize import curve_fit
# from scipy import stats
# from tqdm import tqdm

# sns.set_theme(style='whitegrid', context='talk')


# def find_project_root(start_path: Optional[Path] = None) -> Path:
#     """Walk upward until finding the repository root containing Dataset and src."""
#     current = (start_path or Path(__file__)).resolve()
#     if current.is_file():
#         current = current.parent

#     for candidate in [current, *current.parents]:
#         if (candidate / "Dataset").exists() and (candidate / "src").exists():
#             return candidate

#     return current


# REPO_ROOT = find_project_root()

# # Reference distance (meters) used to normalize distance-compensated intensity.
# # All compensated intensities are rescaled to "what the intensity would be at
# # this distance" using the fitted per-color decay model.
# INTENSITY_REF_DISTANCE = 5.0


# def load_pcd_binary(filepath: Path) -> Optional[np.ndarray]:
#     """Load binary PCD file containing (X, Y, Z, Intensity)."""
#     try:
#         with open(filepath, 'rb') as f:
#             while True:
#                 line = f.readline().decode('utf-8', errors='ignore').strip()
#                 if line.startswith('DATA'):
#                     break
#                 if not line:
#                     return None
#             points = []
#             while True:
#                 data = f.read(16)
#                 if len(data) < 16:
#                     break
#                 x, y, z, intensity = struct.unpack('ffff', data)
#                 points.append([x, y, z, intensity])
#         return np.array(points, dtype=np.float32) if points else None
#     except Exception:
#         return None


# def extract_raw_metrics(points: np.ndarray) -> Optional[Dict[str, float]]:
#     """Extract geometry and intensity metrics from raw point cloud."""
#     if points is None or len(points) < 3:
#         return None

#     xyz = points[:, :3]
#     intensity = points[:, 3]
#     z = xyz[:, 2]

#     num_points = len(points)
#     distance = float(np.linalg.norm(xyz.mean(axis=0)))
#     if distance < 1e-3:
#         return None

#     height = float(z.max() - z.min())
#     x_std = float(xyz[:, 0].std())
#     y_std = float(xyz[:, 1].std())
#     width = max(x_std, y_std, 1e-6)
#     aspect_ratio = height / width
#     volume_approx = float((4 * x_std * y_std) * height)

#     avg_i = float(intensity.mean())
#     std_i = float(intensity.std())
#     i_lo, i_hi = np.percentile(intensity, [5, 95])
#     contrast = float((i_hi - i_lo) / (avg_i + 1e-6))
#     reflective_pct = float(np.mean(intensity > (avg_i * 1.5)))

#     # Three-band vertical intensity profiles (bottom / mid / top) are still
#     # extracted here since they are cheap and may be useful for other
#     # purposes, but the derived contrast_* ratios are NO LONGER used in the
#     # statistical summary or plots: with low point counts per cluster (esp.
#     # at range) the per-band split becomes noisy/unreliable, so they are
#     # excluded from `generate_color_feature_stats`.
#     z_min, z_max = z.min(), z.max()
#     z_range = max(z_max - z_min, 1e-6)
#     bot_mask = z < (z_min + 0.33 * z_range)
#     top_mask = z > (z_min + 0.67 * z_range)
#     mid_mask = ~bot_mask & ~top_mask

#     bot_i = float(intensity[bot_mask].mean()) if bot_mask.sum() > 0 else avg_i
#     mid_i = float(intensity[mid_mask].mean()) if mid_mask.sum() > 0 else avg_i
#     top_i = float(intensity[top_mask].mean()) if top_mask.sum() > 0 else avg_i

#     return {
#         "distance": distance,
#         "num_points": num_points,
#         "height": height,
#         "width": width,
#         "aspect_ratio": aspect_ratio,
#         "volume_approx": volume_approx,
#         "avg_intensity": avg_i,
#         "std_intensity": std_i,
#         "contrast": contrast,
#         "reflective_pct": reflective_pct,
#         "bot_intensity": bot_i,
#         "mid_intensity": mid_i,
#         "top_intensity": top_i,
#         # Kept in the raw record dict (cheap, may be useful elsewhere) but
#         # deliberately excluded from generate_color_feature_stats() below.
#         "contrast_bot_mid": (mid_i - bot_i) / (avg_i + 1e-6),
#         "contrast_mid_top": (top_i - mid_i) / (avg_i + 1e-6),
#         "contrast_bot_top": (top_i - bot_i) / (avg_i + 1e-6),
#     }


# # ---------------------------------------------------------------------------
# # Candidate models for N(d) best-fit search
# # ---------------------------------------------------------------------------

# def power_law(d: np.ndarray, a: float, b: float) -> np.ndarray:
#     """Power-law curve model: f(d) = a * d^b"""
#     return a * np.power(d, b)


# def inverse_square_law(d: np.ndarray, a: float) -> np.ndarray:
#     """Theoretical LiDAR density decay model: f(d) = a / d^2"""
#     return a / (d ** 2)


# def exponential_decay(d: np.ndarray, a: float, k: float, c: float) -> np.ndarray:
#     """Exponential decay model: f(d) = a * exp(-k * d) + c"""
#     return a * np.exp(-k * d) + c


# def rational_decay(d: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
#     """Rational decay model: f(d) = a / (d + b) + c"""
#     return a / (d + b) + c


# CANDIDATE_MODELS: Dict[str, Dict] = {
#     "power_law": {
#         "func": power_law,
#         "p0": [500.0, -1.5],
#         "n_params": 2,
#         "label": lambda p: f"N(d) = {p[0]:.2f} * d^({p[1]:.3f})",
#     },
#     "inverse_square": {
#         "func": inverse_square_law,
#         "p0": [500.0],
#         "n_params": 1,
#         "label": lambda p: f"N(d) = {p[0]:.2f} / d^2",
#     },
#     "exponential_decay": {
#         "func": exponential_decay,
#         "p0": [500.0, 0.2, 5.0],
#         "n_params": 3,
#         "label": lambda p: f"N(d) = {p[0]:.2f} * exp(-{p[1]:.3f} d) + {p[2]:.2f}",
#     },
#     "rational_decay": {
#         "func": rational_decay,
#         "p0": [500.0, 1.0, 0.0],
#         "n_params": 3,
#         "label": lambda p: f"N(d) = {p[0]:.2f} / (d + {p[1]:.2f}) + {p[2]:.2f}",
#     },
# }


# def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     ss_res = float(np.sum((y_true - y_pred) ** 2))
#     ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
#     if ss_tot < 1e-12:
#         return 0.0
#     return 1.0 - ss_res / ss_tot


# def akaike_information_criterion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
#     """
#     Compute AIC for a least-squares fit, assuming normally distributed
#     residuals:   AIC = n * ln(SS_res / n) + 2k
#     Lower AIC indicates a better trade-off between fit quality and model
#     complexity (number of free parameters k). This is what should be used to
#     pick between models with different numbers of parameters (e.g. the
#     2-parameter power law vs. the 3-parameter exponential/rational decay),
#     since raw R^2 mechanically favors models with more free parameters even
#     when the extra flexibility is just fitting noise.
#     """
#     n = len(y_true)
#     ss_res = float(np.sum((y_true - y_pred) ** 2))
#     ss_res = max(ss_res, 1e-12)
#     return n * np.log(ss_res / n) + 2 * k


# def fit_best_model(d: np.ndarray, y: np.ndarray, models: Dict[str, Dict] = CANDIDATE_MODELS,
#                     maxfev: int = 10000) -> Tuple[str, np.ndarray, float, float, Callable]:
#     """
#     Try all candidate models, return the one with the LOWEST AIC (best
#     complexity-penalized fit), not simply the highest R^2. Returns
#     (best_name, best_params, best_aic, best_r2, best_func).
#     """
#     best_name, best_params, best_aic, best_r2, best_func = None, None, np.inf, None, None

#     for name, spec in models.items():
#         func = spec["func"]
#         p0 = spec["p0"]
#         k = spec["n_params"]
#         try:
#             popt, _ = curve_fit(func, d, y, p0=p0, maxfev=maxfev)
#             y_pred = func(d, *popt)
#             if not np.all(np.isfinite(y_pred)):
#                 continue
#             aic = akaike_information_criterion(y, y_pred, k)
#             r2 = r_squared(y, y_pred)
#         except Exception:
#             continue

#         if aic < best_aic:
#             best_name, best_params, best_aic, best_r2, best_func = name, popt, aic, r2, func

#     return best_name, best_params, best_aic, best_r2, best_func


# class ConeDatasetAnalyzer:
#     def __init__(self, dataset_path: Path):
#         self.dataset_path = dataset_path.resolve()
#         self.output_dir = REPO_ROOT / "figures" / "statistical_analysis"
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#     def collect_all_pcds_for_point_estimation(self) -> List[Dict[str, float]]:
#         """Collect distance & point count metrics from ALL PCD files into a list of dicts."""
#         print("🔍 Scanning full Dataset directory for point count estimation...")
#         all_pcds = list(self.dataset_path.rglob("*.pcd"))
#         print(f"   Found {len(all_pcds)} total PCD files.")

#         records = []
#         for pcd_file in tqdm(all_pcds, desc="Processing ALL PCDs"):
#             pts = load_pcd_binary(pcd_file)
#             metrics = extract_raw_metrics(pts)
#             if metrics:
#                 records.append(metrics)

#         print(f"   Successfully extracted metrics from {len(records)} valid clusters.")
#         return records

#     def collect_labeled_color_pcds(self) -> List[Dict[str, float]]:
#         """Collect features strictly from Processed/Color and raw folders with labels."""
#         print("\n🎨 Scanning Processed/Color and raw directories for color analysis...")
#         color_base = self.dataset_path / "Processed" / "Color"
#         raw_base = self.dataset_path / "raw"

#         if not color_base.exists():
#             print(f"⚠️ Warning: {color_base} does not exist. Trying base path.")
#             color_base = self.dataset_path

#         label_files = list(color_base.rglob("labeled_clusters.json"))
#         print(f"   Found {len(label_files)} label JSON files.")

#         valid_colors = {"orange", "blue", "yellow"}
#         records = []

#         for label_path in label_files:
#             track_folder = label_path.parent
#             try:
#                 rel_track = track_folder.relative_to(color_base)
#             except ValueError:
#                 rel_track = track_folder.name

#             with open(label_path, "r") as f:
#                 labels = json.load(f)

#             for cluster_file, info in labels.items():
#                 color = str(info.get("color", "")).lower().strip()
#                 if color not in valid_colors:
#                     continue

#                 pcd_path = track_folder / cluster_file

#                 if not pcd_path.exists() and raw_base.exists():
#                     pcd_path = raw_base / rel_track / cluster_file

#                 if not pcd_path.exists():
#                     matches = list(self.dataset_path.rglob(cluster_file))
#                     if matches:
#                         pcd_path = matches[0]

#                 if not pcd_path.exists():
#                     continue

#                 pts = load_pcd_binary(pcd_path)
#                 metrics = extract_raw_metrics(pts)
#                 if metrics:
#                     metrics["color"] = color
#                     metrics["filename"] = cluster_file
#                     records.append(metrics)

#         print(f"   Successfully built color dataset: {len(records)} samples.")
#         return records

#     def fit_distance_point_estimator(
#         self, records: List[Dict[str, float]]
#     ) -> Tuple[str, np.ndarray, float]:
#         """
#         Fits pure decay candidate models (Power Law via Log-Log Pearson,
#         Exponential via Log-Linear Pearson, Inverse-Square, and Rational Decay)
#         without additive (+c) constants. Evaluates fits in log-space using AIC
#         and returns (best_name, best_params, best_r2).
#         """
#         print("\n📈 Searching for Best-Fit Point Count Estimator Model...")
#         d = np.array([r["distance"] for r in records], dtype=np.float64)
#         n = np.array([r["num_points"] for r in records], dtype=np.float64)

#         valid_mask = (d > 0.5) & (d < 30.0) & (n >= 3)
#         d_clean, n_clean = d[valid_mask], n[valid_mask]
#         log_d = np.log(d_clean)
#         log_n = np.log(n_clean)

#         candidates = {}

#         # 1. Power Law via Log-Log Pearson Regression: ln(N) = ln(a) - b * ln(d)
#         r_loglog = float(np.corrcoef(log_d, log_n)[0, 1])
#         slope_pl, intercept_pl = np.polyfit(log_d, log_n, 1)
#         a_pl, b_pl = np.exp(intercept_pl), -slope_pl
#         params_pl = np.array([a_pl, b_pl], dtype=np.float64)
#         func_pl = lambda x, a, b: a * (x ** (-b))
#         candidates["power_law_pearson"] = {
#             "params": params_pl,
#             "func": func_pl,
#             "n_params": 2,
#             "label": f"N(d) = {a_pl:.2f} * d^(-{b_pl:.3f})]"
#         }

#         # 2. Pure Exponential Decay via Log-Linear Pearson Regression: ln(N) = ln(a) - b * d
#         r_loglin = float(np.corrcoef(d_clean, log_n)[0, 1])
#         slope_exp, intercept_exp = np.polyfit(d_clean, log_n, 1)
#         a_exp, b_exp = np.exp(intercept_exp), -slope_exp
#         params_exp = np.array([a_exp, b_exp], dtype=np.float64)
#         func_exp = lambda x, a, b: a * np.exp(-b * x)
#         candidates["pure_exponential"] = {
#             "params": params_exp,
#             "func": func_exp,
#             "n_params": 2,
#             "label": f"N(d) = {a_exp:.2f} * exp(-{b_exp:.3f} d) [r_loglin={r_loglin:.3f}]"
#         }

#         # 3. Pure Inverse-Square Law: N(d) = a / d^2
#         a_inv2 = float(np.exp(np.mean(log_n + 2.0 * log_d)))
#         params_inv2 = np.array([a_inv2], dtype=np.float64)
#         func_inv2 = lambda x, a: a / (x ** 2)
#         candidates["inverse_square"] = {
#             "params": params_inv2,
#             "func": func_inv2,
#             "n_params": 1,
#             "label": f"N(d) = {a_inv2:.2f} / d^2"
#         }

#         # 4. Rational Decay: N(d) = a / (1 + b * d^2)
#         try:
#             func_rat = lambda x, a, b: a / (1.0 + b * (x ** 2))
#             log_func_rat = lambda x, a, b: np.log(np.maximum(func_rat(x, a, b), 1e-6))
#             popt_rat, _ = curve_fit(log_func_rat, d_clean, log_n, p0=[n_clean.max(), 0.1], maxfev=10000)
#             candidates["rational_decay"] = {
#                 "params": popt_rat,
#                 "func": func_rat,
#                 "n_params": 2,
#                 "label": f"N(d) = {popt_rat[0]:.2f} / (1 + {popt_rat[1]:.4f} * d^2)"
#             }
#         except Exception:
#             pass

#         # Evaluate all models in log-space to select the best via AIC
#         results = []
#         for name, spec in candidates.items():
#             y_pred = spec["func"](d_clean, *spec["params"])
#             log_y_pred = np.log(np.maximum(y_pred, 1e-6))

#             aic = akaike_information_criterion(log_n, log_y_pred, spec["n_params"])
#             r2 = r_squared(log_n, log_y_pred)
#             results.append((name, spec["params"], aic, r2, spec["func"], spec["label"]))

#         # Sort candidates by AIC (lowest is best)
#         results.sort(key=lambda x: x[2])
#         best_name, best_params, best_aic, best_r2, best_func, best_label = results[0]

#         print("   --- Non-Linear Pure Decay Models (ranked by Log-Space AIC) ---")
#         for name, params, aic, r2, func, label in results:
#             marker = " <== BEST (lowest AIC)" if name == best_name else ""
#             print(f"   [{name:<20}] AIC={aic:9.2f}  R^2={r2:.4f}  {label}{marker}")

#         print(f"\n   ✅ Best model: {best_name}  (AIC={best_aic:.2f}, R^2={best_r2:.4f})")
#         print(f"   {best_label}")

#         # Plot raw scatter + winning pure-decay fit
#         plt.figure(figsize=(10, 6))
#         plt.scatter(d, n, alpha=0.2, color="gray", label="Observed Cones")

#         d_grid = np.linspace(d_clean.min(), d_clean.max(), 200)
#         plt.plot(d_grid, best_func(d_grid, *best_params), 'r-', lw=2.5, label=f"Fit: {best_label}")

#         plt.title("Cone Point Count Estimation vs. Distance")
#         plt.xlabel("Distance from LiDAR (m)")
#         plt.ylabel("Point Count (N)")
#         plt.yscale("log")
#         plt.ylim(bottom=2)
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(self.output_dir / "distance_vs_point_count.png", dpi=300)
#         plt.close()
#         print(f"   Saved plot: {self.output_dir / 'distance_vs_point_count.png'}")

#         return best_name, best_params, best_r2

#     def fit_and_plot_intensity_decay(self, records: List[Dict[str, float]]) -> Dict[str, Dict]:
#         """
#         Fit the best-available decay model (selected via AIC) for
#         avg_intensity vs distance, per color, and plot the decay curves.
#         """
#         print("\n⚡ Modeling Intensity Decay (via AIC)...")
#         decay_params = {}

#         colors = sorted(list({r["color"] for r in records}))
#         # Standard matplotlib colors for consistency
#         colors_map = {"orange": "tab:orange", "blue": "tab:blue", "yellow": "gold"}

#         plt.figure(figsize=(11, 7))

#         for color in colors:
#             color_records = [r for r in records if r["color"] == color]
#             d = np.array([r["distance"] for r in color_records], dtype=np.float64)
#             i_avg = np.array([r["avg_intensity"] for r in color_records], dtype=np.float64)

#             # Filter out extreme distances or invalid intensities
#             mask = (d > 0.5) & (d < 30.0) & (i_avg > 0)
#             d_c, i_c = d[mask], i_avg[mask]

#             # Require at least 5 points to attempt a curve fit
#             if len(d_c) < 5:
#                 continue

#             # Fit the data
#             name, popt, aic, r2, func = fit_best_model(
#                 d_c, i_c,
#                 models={k: v for k, v in CANDIDATE_MODELS.items() if k != "inverse_square"},
#             )
#             decay_params[color] = {"model": name, "params": popt, "aic": aic, "r2": r2, "func": func}

#             label = CANDIDATE_MODELS[name]["label"](popt).replace("N(d)", "I(d)")
#             print(f"   Color [{color.upper()}]: best model = {name} (AIC={aic:.2f}, R^2={r2:.4f})  {label}")

#             # Plot raw scatter points
#             plt.scatter(d_c, i_c, alpha=0.25, color=colors_map.get(color, "gray"))
            
#             # Plot the best-fit decay curve
#             d_grid = np.linspace(d_c.min(), d_c.max(), 150)
#             plt.plot(d_grid, func(d_grid, *popt), color=colors_map.get(color, "black"), lw=2.5,
#                      label=f"{color.capitalize()} ({name})")

#         plt.title("LiDAR Intensity Decay vs. Distance")
#         plt.xlabel("Distance from LiDAR (m)")
#         plt.ylabel("Raw Intensity")
#         plt.legend()
#         plt.tight_layout()
        
#         # Save the plot
#         plot_path = self.output_dir / "distance_vs_intensity_decay.png"
#         plt.savefig(plot_path, dpi=300)
#         plt.close()
#         print(f"   Saved plot: {plot_path}")

#         return decay_params

#     # def generate_color_feature_stats(self, records: List[Dict[str, float]]):
#     #     """
#     #     Compute summary statistics for features grouped by cone color without pandas.
#     #     Generates clean summary CSV and boxplot distributions per feature.
#     #     """
#     #     print("\n📊 Generating Per-Color Statistical Summaries...")
        
#     #     # Replaced 'avg_intensity_compensated' with 'std_intensity' since compensation logic was dropped
#     #     features_to_analyze = [
#     #         "height", "width", "aspect_ratio", "volume_approx",
#     #         "avg_intensity", "std_intensity", "mid_intensity", "reflective_pct",
#     #     ]

#     #     by_color = {}
#     #     for r in records:
#     #         c = r["color"]
#     #         if c not in by_color:
#     #             by_color[c] = []
#     #         by_color[c].append(r)

#     #     summary_rows = []

#     #     print("\n" + "=" * 95)
#     #     print(f"{'Color':<10} | {'Feature':<24} | {'Mean':<8} | {'Std':<8} | {'Median':<8} | {'IQR':<8} | {'Min':<8} | {'Max':<8}")
#     #     print("=" * 95)

#     #     for color, color_records in sorted(by_color.items()):
#     #         for feat in features_to_analyze:
#     #             vals = np.array([r[feat] for r in color_records if feat in r], dtype=np.float64)
#     #             if vals.size == 0:
#     #                 continue
#     #             mean_val = float(np.mean(vals))
#     #             std_val = float(np.std(vals))
#     #             med_val = float(np.median(vals))
#     #             iqr_val = float(stats.iqr(vals))
#     #             min_val = float(np.min(vals))
#     #             max_val = float(np.max(vals))

#     #             summary_rows.append({
#     #                 "Color": color,
#     #                 "Feature": feat,
#     #                 "Mean": mean_val,
#     #                 "Std": std_val,
#     #                 "Median": med_val,
#     #                 "IQR": iqr_val,
#     #                 "Min": min_val,
#     #                 "Max": max_val
#     #             })

#     #             print(f"{color:<10} | {feat:<24} | {mean_val:<8.3f} | {std_val:<8.3f} | {med_val:<8.3f} | {iqr_val:<8.3f} | {min_val:<8.3f} | {max_val:<8.3f}")

#     #     print("=" * 95 + "\n")

#     #     # Export CSV via stdlib csv.DictWriter
#     #     csv_path = self.output_dir / "color_feature_statistics.csv"
#     #     with open(csv_path, "w", newline="") as f:
#     #         writer = csv.DictWriter(f, fieldnames=["Color", "Feature", "Mean", "Std", "Median", "IQR", "Min", "Max"])
#     #         writer.writeheader()
#     #         writer.writerows(summary_rows)
#     #     print(f"   Saved CSV table: {csv_path}")

#     #     # Visual distributions
#     #     fig, axes = plt.subplots(2, 3, figsize=(18, 10))
#     #     plot_feats = ["height", "width", "avg_intensity", "std_intensity", "mid_intensity", "reflective_pct"]

#     #     colors = sorted(list(by_color.keys()))
        
#     #     # Map color names explicitly to prevent label-color mismatch
#     #     color_map = {
#     #         "orange": "tab:orange",
#     #         "blue": "tab:blue",
#     #         "yellow": "gold"
#     #     }

#     #     for ax, feat in zip(axes.flat, plot_feats):
#     #         data_to_plot = [[r[feat] for r in by_color[c] if feat in r] for c in colors]
#     #         bplot = ax.boxplot(data_to_plot, labels=[c.capitalize() for c in colors], patch_artist=True)
            
#     #         # Explicitly set fill color based on the current label key
#     #         for patch, c in zip(bplot['boxes'], colors):
#     #             patch.set_facecolor(color_map.get(c, "gray"))
#     #             patch.set_alpha(0.7)
                
#     #         ax.set_title(f"Distribution of {feat.replace('_', ' ').capitalize()}", fontsize=12, fontweight="bold")
#     #         ax.grid(True, linestyle="--", alpha=0.5)

#     #     plt.tight_layout()
#     #     plt.savefig(self.output_dir / "color_feature_distributions.png", dpi=300)
#     #     plt.close()
#     #     print(f"   Saved distribution plot: {self.output_dir / 'color_feature_distributions.png'}")
    
#     def generate_color_feature_stats(self, records: List[Dict[str, float]]):
#         """
#         Compute summary statistics for features grouped by cone color without pandas.
#         Generates clean summary CSV and boxplot distributions per feature (including contrast metrics).
#         """
#         print("\n📊 Generating Per-Color Statistical Summaries (including Vertical Contrast)...")
        
#         # Re-included 'contrast' and vertical band contrast ratios
#         features_to_analyze = [
#             "height", "width", "aspect_ratio", "volume_approx",
#             "avg_intensity", "std_intensity", "mid_intensity", "reflective_pct",
#             "contrast", "contrast_bot_mid", "contrast_mid_top", "contrast_bot_top"
#         ]

#         by_color = {}
#         for r in records:
#             c = r["color"]
#             if c not in by_color:
#                 by_color[c] = []
#             by_color[c].append(r)

#         summary_rows = []

#         print("\n" + "=" * 95)
#         print(f"{'Color':<10} | {'Feature':<24} | {'Mean':<8} | {'Std':<8} | {'Median':<8} | {'IQR':<8} | {'Min':<8} | {'Max':<8}")
#         print("=" * 95)

#         for color, color_records in sorted(by_color.items()):
#             for feat in features_to_analyze:
#                 vals = np.array([r[feat] for r in color_records if feat in r and np.isfinite(r[feat])], dtype=np.float64)
#                 if vals.size == 0:
#                     continue
#                 mean_val = float(np.mean(vals))
#                 std_val = float(np.std(vals))
#                 med_val = float(np.median(vals))
#                 iqr_val = float(stats.iqr(vals))
#                 min_val = float(np.min(vals))
#                 max_val = float(np.max(vals))

#                 summary_rows.append({
#                     "Color": color,
#                     "Feature": feat,
#                     "Mean": mean_val,
#                     "Std": std_val,
#                     "Median": med_val,
#                     "IQR": iqr_val,
#                     "Min": min_val,
#                     "Max": max_val
#                 })

#                 print(f"{color:<10} | {feat:<24} | {mean_val:<8.3f} | {std_val:<8.3f} | {med_val:<8.3f} | {iqr_val:<8.3f} | {min_val:<8.3f} | {max_val:<8.3f}")

#         print("=" * 95 + "\n")

#         # Export CSV via stdlib csv.DictWriter
#         csv_path = self.output_dir / "color_feature_statistics.csv"
#         with open(csv_path, "w", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=["Color", "Feature", "Mean", "Std", "Median", "IQR", "Min", "Max"])
#             writer.writeheader()
#             writer.writerows(summary_rows)
#         print(f"   Saved CSV table: {csv_path}")

#         # Visual distributions (3x3 grid to fit general geometry, intensity, and contrast)
#         fig, axes = plt.subplots(3, 3, figsize=(18, 14))
#         plot_feats = [
#             "height", "width", "avg_intensity",
#             "std_intensity", "mid_intensity", "reflective_pct",
#             "contrast", "contrast_bot_mid", "contrast_bot_top"
#         ]

#         colors = sorted(list(by_color.keys()))
        
#         color_map = {
#             "orange": "tab:orange",
#             "blue": "tab:blue",
#             "yellow": "gold"
#         }

#         for ax, feat in zip(axes.flat, plot_feats):
#             data_to_plot = [[r[feat] for r in by_color[c] if feat in r and np.isfinite(r[feat])] for c in colors]
#             bplot = ax.boxplot(data_to_plot, tick_labels=[c.capitalize() for c in colors], patch_artist=True)
            
#             for patch, c in zip(bplot['boxes'], colors):
#                 patch.set_facecolor(color_map.get(c, "gray"))
#                 patch.set_alpha(0.7)
                
#             ax.set_title(f"Distribution of {feat.replace('_', ' ').capitalize()}", fontsize=11, fontweight="bold")
#             ax.grid(True, linestyle="--", alpha=0.5)

#         plt.tight_layout()
#         plt.savefig(self.output_dir / "color_feature_distributions.png", dpi=300)
#         plt.close()
#         print(f"   Saved distribution plot: {self.output_dir / 'color_feature_distributions.png'}")

# def main():
#     print(f"\n🚀 Running Cone Statistical Analysis on Project Root: {REPO_ROOT}")
#     analyzer = ConeDatasetAnalyzer(REPO_ROOT / "Dataset")

#     # Step 1: Distance vs Point Count -- automatic best-fit model search (AIC)
#     records_all = analyzer.collect_all_pcds_for_point_estimation()
#     if records_all:
#         analyzer.fit_distance_point_estimator(records_all)
#     else:
#         print("❌ No dataset PCD files found for point estimator modeling.")
#         return

#     # Step 2 & 3: Color Statistics & Distance-Compensated Intensity
#     records_color = analyzer.collect_labeled_color_pcds()
#     if records_color:
#         analyzer.fit_and_plot_intensity_decay(records_color)
#         analyzer.generate_color_feature_stats(records_color)
#     else:
#         print("⚠️ No labeled color files found in Processed/Color or raw.")

#     print("\n✅ Analysis Complete! All figures and statistics saved to:")
#     print(f"   {REPO_ROOT / 'figures' / 'statistical_analysis'}\n")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Statistical Analysis & Intensity Unmixing of LiDAR Cone Clusters without Pandas.

Key Capabilities:
1. Distance-based Point Count Estimator: Automatic best-fit model search
   (power law, inverse-square, exponential decay, rational decay) across all
   Dataset PCDs, selected by AIC (Akaike Information Criterion).
2. Two-Pass Point-Level Intensity Unmixing & Normalization: Bootstrap-masks
   retroreflective white tape from plastic cone body points, fits per-group 
   power-law intensity decay on the 0-255 scale, and normalizes intensity to a 
   reference distance (5.0m).
3. Visualizations: Generates combined and separated (Orange vs White, Blue vs White)
   raw decay and distance-compensated distribution graphs.
4. Pure NumPy & Standard Library implementation (compatible with NumPy 1.21.5).
"""

import csv
import json
import os
import re
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

sns.set_theme(style='whitegrid', context='talk')


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """Walk upward until finding the repository root containing Dataset and src."""
    current = (start_path or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / "Dataset").exists() and (candidate / "src").exists():
            return candidate

    return current


REPO_ROOT = find_project_root()

# Reference distance (meters) used to normalize distance-compensated intensity.
INTENSITY_REF_DISTANCE = 5.0


def load_pcd_binary(filepath: Path) -> Optional[np.ndarray]:
    """Load binary PCD file containing (X, Y, Z, Intensity)."""
    try:
        with open(filepath, 'rb') as f:
            while True:
                line = f.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('DATA'):
                    break
                if not line:
                    return None
            points = []
            while True:
                data = f.read(16)
                if len(data) < 16:
                    break
                x, y, z, intensity = struct.unpack('ffff', data)
                points.append([x, y, z, intensity])
        return np.array(points, dtype=np.float32) if points else None
    except Exception:
        return None


def extract_raw_metrics(points: np.ndarray) -> Optional[Dict[str, float]]:
    """Extract geometry and intensity metrics from raw point cloud."""
    if points is None or len(points) < 3:
        return None

    xyz = points[:, :3]
    intensity = points[:, 3]
    z = xyz[:, 2]

    num_points = len(points)
    distance = float(np.linalg.norm(xyz.mean(axis=0)))
    if distance < 1e-3:
        return None

    height = float(z.max() - z.min())
    x_std = float(xyz[:, 0].std())
    y_std = float(xyz[:, 1].std())
    width = max(x_std, y_std, 1e-6)
    aspect_ratio = height / width
    volume_approx = float((4 * x_std * y_std) * height)

    avg_i = float(intensity.mean())
    std_i = float(intensity.std())
    i_lo, i_hi = np.percentile(intensity, [5, 95])
    contrast = float((i_hi - i_lo) / (avg_i + 1e-6))
    reflective_pct = float(np.mean(intensity > (avg_i * 1.5)))

    z_min, z_max = z.min(), z.max()
    z_range = max(z_max - z_min, 1e-6)
    bot_mask = z < (z_min + 0.33 * z_range)
    top_mask = z > (z_min + 0.67 * z_range)
    mid_mask = ~bot_mask & ~top_mask

    bot_i = float(intensity[bot_mask].mean()) if bot_mask.sum() > 0 else avg_i
    mid_i = float(intensity[mid_mask].mean()) if mid_mask.sum() > 0 else avg_i
    top_i = float(intensity[top_mask].mean()) if top_mask.sum() > 0 else avg_i

    return {
        "distance": distance,
        "num_points": num_points,
        "height": height,
        "width": width,
        "aspect_ratio": aspect_ratio,
        "volume_approx": volume_approx,
        "avg_intensity": avg_i,
        "std_intensity": std_i,
        "contrast": contrast,
        "reflective_pct": reflective_pct,
        "bot_intensity": bot_i,
        "mid_intensity": mid_i,
        "top_intensity": top_i,
        "contrast_bot_mid": (mid_i - bot_i) / (avg_i + 1e-6),
        "contrast_mid_top": (top_i - mid_i) / (avg_i + 1e-6),
        "contrast_bot_top": (top_i - bot_i) / (avg_i + 1e-6),
    }


# ---------------------------------------------------------------------------
# Candidate models for N(d) best-fit search
# ---------------------------------------------------------------------------

def power_law(d: np.ndarray, a: float, b: float) -> np.ndarray:
    """Power-law curve model: f(d) = a * d^b"""
    return a * np.power(d, b)


def inverse_square_law(d: np.ndarray, a: float) -> np.ndarray:
    """Theoretical LiDAR density decay model: f(d) = a / d^2"""
    return a / (d ** 2)


def exponential_decay(d: np.ndarray, a: float, k: float, c: float) -> np.ndarray:
    """Exponential decay model: f(d) = a * exp(-k * d) + c"""
    return a * np.exp(-k * d) + c


def rational_decay(d: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Rational decay model: f(d) = a / (d + b) + c"""
    return a / (d + b) + c


CANDIDATE_MODELS: Dict[str, Dict] = {
    "power_law": {
        "func": power_law,
        "p0": [500.0, -1.5],
        "n_params": 2,
        "label": lambda p: f"N(d) = {p[0]:.2f} * d^({p[1]:.3f})",
    },
    "inverse_square": {
        "func": inverse_square_law,
        "p0": [500.0],
        "n_params": 1,
        "label": lambda p: f"N(d) = {p[0]:.2f} / d^2",
    },
    "exponential_decay": {
        "func": exponential_decay,
        "p0": [500.0, 0.2, 5.0],
        "n_params": 3,
        "label": lambda p: f"N(d) = {p[0]:.2f} * exp(-{p[1]:.3f} d) + {p[2]:.2f}",
    },
    "rational_decay": {
        "func": rational_decay,
        "p0": [500.0, 1.0, 0.0],
        "n_params": 3,
        "label": lambda p: f"N(d) = {p[0]:.2f} / (d + {p[1]:.2f}) + {p[2]:.2f}",
    },
}


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def akaike_information_criterion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    n = len(y_true)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_res = max(ss_res, 1e-12)
    return n * np.log(ss_res / n) + 2 * k


def fit_best_model(d: np.ndarray, y: np.ndarray, models: Dict[str, Dict] = CANDIDATE_MODELS,
                   maxfev: int = 10000) -> Tuple[str, np.ndarray, float, float, Callable]:
    best_name, best_params, best_aic, best_r2, best_func = None, None, np.inf, None, None

    for name, spec in models.items():
        func = spec["func"]
        p0 = spec["p0"]
        k = spec["n_params"]
        try:
            popt, _ = curve_fit(func, d, y, p0=p0, maxfev=maxfev)
            y_pred = func(d, *popt)
            if not np.all(np.isfinite(y_pred)):
                continue
            aic = akaike_information_criterion(y, y_pred, k)
            r2 = r_squared(y, y_pred)
        except Exception:
            continue

        if aic < best_aic:
            best_name, best_params, best_aic, best_r2, best_func = name, popt, aic, r2, func

    return best_name, best_params, best_aic, best_r2, best_func


class ConeDatasetAnalyzer:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path.resolve()
        self.output_dir = REPO_ROOT / "figures" / "statistical_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_all_pcds_for_point_estimation(self) -> List[Dict[str, float]]:
        print("🔍 Scanning full Dataset directory for point count estimation...")
        all_pcds = list(self.dataset_path.rglob("*.pcd"))
        print(f"   Found {len(all_pcds)} total PCD files.")

        records = []
        for pcd_file in tqdm(all_pcds, desc="Processing ALL PCDs"):
            pts = load_pcd_binary(pcd_file)
            metrics = extract_raw_metrics(pts)
            if metrics:
                records.append(metrics)

        print(f"   Successfully extracted metrics from {len(records)} valid clusters.")
        return records

    def collect_labeled_color_pcds(self) -> List[Dict]:
        print("\n🎨 Scanning Processed/Color and raw directories for color analysis...")
        color_base = self.dataset_path / "Processed" / "Color"
        raw_base = self.dataset_path / "raw"

        if not color_base.exists():
            print(f"⚠️ Warning: {color_base} does not exist. Trying base path.")
            color_base = self.dataset_path

        label_files = list(color_base.rglob("labeled_clusters.json"))
        print(f"   Found {len(label_files)} label JSON files.")

        valid_colors = {"orange", "blue", "yellow"}
        records = []

        for label_path in label_files:
            track_folder = label_path.parent
            try:
                rel_track = track_folder.relative_to(color_base)
            except ValueError:
                rel_track = track_folder.name

            with open(label_path, "r") as f:
                labels = json.load(f)

            for cluster_file, info in labels.items():
                color = str(info.get("color", "")).lower().strip()
                if color not in valid_colors:
                    continue

                pcd_path = track_folder / cluster_file

                if not pcd_path.exists() and raw_base.exists():
                    pcd_path = raw_base / rel_track / cluster_file

                if not pcd_path.exists():
                    matches = list(self.dataset_path.rglob(cluster_file))
                    if matches:
                        pcd_path = matches[0]

                if not pcd_path.exists():
                    continue

                pts = load_pcd_binary(pcd_path)
                metrics = extract_raw_metrics(pts)
                if metrics:
                    metrics["color"] = color
                    metrics["filename"] = cluster_file
                    metrics["pcd_path"] = pcd_path
                    records.append(metrics)

        print(f"   Successfully built color dataset: {len(records)} samples.")
        return records

    def fit_distance_point_estimator(
        self, records: List[Dict[str, float]]
    ) -> Tuple[str, np.ndarray, float]:
        print("\n📈 Searching for Best-Fit Point Count Estimator Model...")
        d = np.array([r["distance"] for r in records], dtype=np.float64)
        n = np.array([r["num_points"] for r in records], dtype=np.float64)

        valid_mask = (d > 0.5) & (d < 30.0) & (n >= 3)
        d_clean, n_clean = d[valid_mask], n[valid_mask]
        log_d = np.log(d_clean)
        log_n = np.log(n_clean)

        candidates = {}

        # 1. Power Law via Log-Log Pearson Regression
        r_loglog = float(np.corrcoef(log_d, log_n)[0, 1])
        slope_pl, intercept_pl = np.polyfit(log_d, log_n, 1)
        a_pl, b_pl = np.exp(intercept_pl), -slope_pl
        params_pl = np.array([a_pl, b_pl], dtype=np.float64)
        func_pl = lambda x, a, b: a * (x ** (-b))
        candidates["power_law_pearson"] = {
            "params": params_pl,
            "func": func_pl,
            "n_params": 2,
            "label": f"N(d) = {a_pl:.2f} * d^(-{b_pl:.3f})"
        }

        # 2. Pure Exponential Decay via Log-Linear Pearson Regression
        r_loglin = float(np.corrcoef(d_clean, log_n)[0, 1])
        slope_exp, intercept_exp = np.polyfit(d_clean, log_n, 1)
        a_exp, b_exp = np.exp(intercept_exp), -slope_exp
        params_exp = np.array([a_exp, b_exp], dtype=np.float64)
        func_exp = lambda x, a, b: a * np.exp(-b * x)
        candidates["pure_exponential"] = {
            "params": params_exp,
            "func": func_exp,
            "n_params": 2,
            "label": f"N(d) = {a_exp:.2f} * exp(-{b_exp:.3f} d)"
        }

        # 3. Pure Inverse-Square Law
        a_inv2 = float(np.exp(np.mean(log_n + 2.0 * log_d)))
        params_inv2 = np.array([a_inv2], dtype=np.float64)
        func_inv2 = lambda x, a: a / (x ** 2)
        candidates["inverse_square"] = {
            "params": params_inv2,
            "func": func_inv2,
            "n_params": 1,
            "label": f"N(d) = {a_inv2:.2f} / d^2"
        }

        # 4. Rational Decay
        try:
            func_rat = lambda x, a, b: a / (1.0 + b * (x ** 2))
            log_func_rat = lambda x, a, b: np.log(np.maximum(func_rat(x, a, b), 1e-6))
            popt_rat, _ = curve_fit(log_func_rat, d_clean, log_n, p0=[n_clean.max(), 0.1], maxfev=10000)
            candidates["rational_decay"] = {
                "params": popt_rat,
                "func": func_rat,
                "n_params": 2,
                "label": f"N(d) = {popt_rat[0]:.2f} / (1 + {popt_rat[1]:.4f} * d^2)"
            }
        except Exception:
            pass

        results = []
        for name, spec in candidates.items():
            y_pred = spec["func"](d_clean, *spec["params"])
            log_y_pred = np.log(np.maximum(y_pred, 1e-6))

            aic = akaike_information_criterion(log_n, log_y_pred, spec["n_params"])
            r2 = r_squared(log_n, log_y_pred)
            results.append((name, spec["params"], aic, r2, spec["func"], spec["label"]))

        results.sort(key=lambda x: x[2])
        best_name, best_params, best_aic, best_r2, best_func, best_label = results[0]

        print("   --- Non-Linear Pure Decay Models (ranked by Log-Space AIC) ---")
        for name, params, aic, r2, func, label in results:
            marker = " <== BEST (lowest AIC)" if name == best_name else ""
            print(f"   [{name:<20}] AIC={aic:9.2f}  R^2={r2:.4f}  {label}{marker}")

        print(f"\n   ✅ Best model: {best_name}  (AIC={best_aic:.2f}, R^2={best_r2:.4f})")
        print(f"   {best_label}")

        plt.figure(figsize=(10, 6))
        plt.scatter(d, n, alpha=0.2, color="gray", label="Observed Cones")

        d_grid = np.linspace(d_clean.min(), d_clean.max(), 200)
        plt.plot(d_grid, best_func(d_grid, *best_params), 'r-', lw=2.5, label=f"Fit: {best_label}")

        plt.title("Cone Point Count Estimation vs. Distance")
        plt.xlabel("Distance from LiDAR (m)")
        plt.ylabel("Point Count (N)")
        plt.yscale("log")
        plt.ylim(bottom=2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "distance_vs_point_count.png", dpi=300)
        plt.close()
        print(f"   Saved plot: {self.output_dir / 'distance_vs_point_count.png'}")

        return best_name, best_params, best_r2

    # def analyze_and_plot_cone_intensity_unmixing(
    #     self, records_color: List[Dict], ref_distance: float = INTENSITY_REF_DISTANCE
    # ):
    #     """
    #     Two-Pass Unmixing Pipeline:
    #     1. Bootstrap initial white tape points (top 20% intensity per cluster).
    #     2. Fit power-law intensity decay I(d) = a * d^(-b) per point category.
    #     3. Normalize intensities to ref_distance (0-255 scale).
    #     4. Plot Combined Overview, Orange vs White, and Blue vs White comparison figures.
    #     """
    #     print("\n🔍 Unmixing White Tape from Orange & Blue Cone Clusters...")

    #     orange_body_d, orange_body_i = [], []
    #     blue_body_d, blue_body_i = [], []
    #     white_tape_d, white_tape_i = [], []

    #     # Step 1: Extract individual point intensities from PCD files
    #     for r in tqdm(records_color, desc="Extracting point intensities"):
    #         color = r["color"]
    #         pcd_path = r.get("pcd_path")
    #         if not pcd_path or not pcd_path.exists():
    #             continue

    #         pts = load_pcd_binary(pcd_path)
    #         if pts is None or len(pts) < 3:
    #             continue

    #         d = r["distance"]
    #         intensities = np.clip(pts[:, 3], 0, 255)

    #         # Bootstrapping: Top 20% intensity = White Tape, Bottom 80% = Plastic Body
    #         p80 = np.percentile(intensities, 80)
    #         tape_mask = intensities >= p80
    #         body_mask = ~tape_mask

    #         white_tape_d.extend([d] * np.sum(tape_mask))
    #         white_tape_i.extend(intensities[tape_mask])

    #         if color == "orange":
    #             orange_body_d.extend([d] * np.sum(body_mask))
    #             orange_body_i.extend(intensities[body_mask])
    #         elif color == "blue":
    #             blue_body_d.extend([d] * np.sum(body_mask))
    #             blue_body_i.extend(intensities[body_mask])

    #     o_d, o_i = np.array(orange_body_d), np.array(orange_body_i)
    #     b_d, b_i = np.array(blue_body_d), np.array(blue_body_i)
    #     w_d, w_i = np.array(white_tape_d), np.array(white_tape_i)

    #     # Step 2: Fit Power Law Decay I(d) = a * d^(-b) per group
    #     def power_decay_decay(d: np.ndarray, a: float, b: float) -> np.ndarray:
    #         return a * np.power(d, -b)

    #     fits = {}
    #     groups = [("Orange Body", o_d, o_i), ("Blue Body", b_d, b_i), ("White Tape", w_d, w_i)]
    #     for name, d_arr, i_arr in groups:
    #         if len(d_arr) > 5:
    #             try:
    #                 popt, _ = curve_fit(power_decay_decay, d_arr, i_arr, p0=[200.0, 1.0], maxfev=10000)
    #                 fits[name] = popt
    #             except Exception:
    #                 fits[name] = np.array([200.0, 1.0])
    #         else:
    #             fits[name] = np.array([200.0, 1.0])

    #     # Step 3: Compute Normalized Intensities
    #     o_i_norm = np.clip(o_i * np.power(o_d / ref_distance, fits["Orange Body"][1]), 0, 255)
    #     b_i_norm = np.clip(b_i * np.power(b_d / ref_distance, fits["Blue Body"][1]), 0, 255)
    #     w_i_norm = np.clip(w_i * np.power(w_d / ref_distance, fits["White Tape"][1]), 0, 255)

    #     d_grid = np.linspace(1.0, 25.0, 200)

    #     # -------------------------------------------------------------------------
    #     # GRAPH 1: COMBINED OVERVIEW (Orange, Blue, White - Raw & Normalized)
    #     # -------------------------------------------------------------------------
    #     fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    #     axes[0].scatter(o_d, o_i, alpha=0.15, color="tab:orange", s=10, label="Orange Body Points")
    #     axes[0].scatter(b_d, b_i, alpha=0.15, color="tab:blue", s=10, label="Blue Body Points")
    #     axes[0].scatter(w_d, w_i, alpha=0.15, color="gray", s=10, label="White Tape Points")

    #     axes[0].plot(d_grid, power_decay_decay(d_grid, *fits["Orange Body"]), color="darkorange", lw=2.5, label="Orange Fit")
    #     axes[0].plot(d_grid, power_decay_decay(d_grid, *fits["Blue Body"]), color="navy", lw=2.5, label="Blue Fit")
    #     axes[0].plot(d_grid, power_decay_decay(d_grid, *fits["White Tape"]), color="black", lw=2.5, linestyle="--", label="White Fit")
    #     axes[0].set_ylabel("Raw Intensity (0-255)")
    #     axes[0].set_title("Combined Raw LiDAR Intensity Decay vs. Distance")
    #     axes[0].legend(loc="upper right")

    #     axes[1].scatter(o_d, o_i_norm, alpha=0.15, color="tab:orange", s=10)
    #     axes[1].scatter(b_d, b_i_norm, alpha=0.15, color="tab:blue", s=10)
    #     axes[1].scatter(w_d, w_i_norm, alpha=0.15, color="gray", s=10)
    #     axes[1].axhline(y=140, color="red", linestyle=":", lw=2, label="Candidate Separation Threshold (T_norm = 140)")
    #     axes[1].set_xlabel("Distance from LiDAR (m)")
    #     axes[1].set_ylabel(f"Normalized Intensity at {ref_distance}m (0-255)")
    #     axes[1].set_title("Combined Normalized Intensity (Distance-Invariance Check)")
    #     axes[1].legend(loc="upper right")

    #     plt.tight_layout()
    #     fig.savefig(self.output_dir / "combined_intensity_decay_and_normalized.png", dpi=300)
    #     plt.close()

    #     # -------------------------------------------------------------------------
    #     # GRAPH 2: ORANGE VS WHITE (Raw & Normalized)
    #     # -------------------------------------------------------------------------
    #     fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    #     axes[0].scatter(o_d, o_i, alpha=0.2, color="tab:orange", s=12, label="Orange Plastic Body")
    #     axes[0].scatter(w_d, w_i, alpha=0.2, color="gray", s=12, label="White Tape")
    #     axes[0].plot(d_grid, power_decay_decay(d_grid, *fits["Orange Body"]), color="darkorange", lw=2.5)
    #     axes[0].plot(d_grid, power_decay_decay(d_grid, *fits["White Tape"]), color="black", lw=2.5, linestyle="--")
    #     axes[0].set_xlabel("Distance (m)")
    #     axes[0].set_ylabel("Raw Intensity (0-255)")
    #     axes[0].set_title("Raw Intensity: Orange Body vs. White Tape")
    #     axes[0].legend()

    #     sns.kdeplot(o_i_norm, ax=axes[1], color="tab:orange", fill=True, label="Orange Body Norm")
    #     sns.kdeplot(w_i_norm, ax=axes[1], color="gray", fill=True, label="White Tape Norm")
    #     axes[1].axvline(x=140, color="red", linestyle="--", label="Threshold = 140")
    #     axes[1].set_xlabel(f"Normalized Intensity at {ref_distance}m (0-255)")
    #     axes[1].set_ylabel("Density")
    #     axes[1].set_title("Normalized Distribution: Orange vs. White")
    #     axes[1].legend()

    #     plt.tight_layout()
    #     fig.savefig(self.output_dir / "orange_vs_white_intensity.png", dpi=300)
    #     plt.close()

    #     # -------------------------------------------------------------------------
    #     # GRAPH 3: BLUE VS WHITE (Raw & Normalized)
    #     # -------------------------------------------------------------------------
    #     fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    #     axes[0].scatter(b_d, b_i, alpha=0.2, color="tab:blue", s=12, label="Blue Plastic Body")
    #     axes[0].scatter(w_d, w_i, alpha=0.2, color="gray", s=12, label="White Tape")
    #     axes[0].plot(d_grid, power_decay_decay(d_grid, *fits["Blue Body"]), color="navy", lw=2.5)
    #     axes[0].plot(d_grid, power_decay_decay(d_grid, *fits["White Tape"]), color="black", lw=2.5, linestyle="--")
    #     axes[0].set_xlabel("Distance (m)")
    #     axes[0].set_ylabel("Raw Intensity (0-255)")
    #     axes[0].set_title("Raw Intensity: Blue Body vs. White Tape")
    #     axes[0].legend()

    #     sns.kdeplot(b_i_norm, ax=axes[1], color="tab:blue", fill=True, label="Blue Body Norm")
    #     sns.kdeplot(w_i_norm, ax=axes[1], color="gray", fill=True, label="White Tape Norm")
    #     axes[1].axvline(x=140, color="red", linestyle="--", label="Threshold = 140")
    #     axes[1].set_xlabel(f"Normalized Intensity at {ref_distance}m (0-255)")
    #     axes[1].set_ylabel("Density")
    #     axes[1].set_title("Normalized Distribution: Blue vs. White")
    #     axes[1].legend()

    #     plt.tight_layout()
    #     fig.savefig(self.output_dir / "blue_vs_white_intensity.png", dpi=300)
    #     plt.close()

    #     print(f"   ✅ Unmixing complete! Generated 3 figures in: {self.output_dir}")

    from sklearn.mixture import GaussianMixture

    # def analyze_and_plot_cone_intensity_unmixing(
    #     self,
    #     records_color: List[Dict],
    #     ref_distance: float = INTENSITY_REF_DISTANCE,
    # ):
    #     """
    #     Aggregate all points from all orange and blue clusters first.

    #     Then, per colour:
    #     - Compute ONE global 80th-percentile intensity threshold.
    #     - Points >= that threshold are treated as white tape.
    #     - Points below it are treated as coloured plastic body.

    #     The white-tape pool combines white candidates from orange and blue cones.
    #     """
    #     print("\nAggregating all Orange and Blue clusters before white/body separation...")

    #     # First pass: aggregate every point from every labelled cluster.
    #     all_orange_d, all_orange_i = [], []
    #     all_blue_d, all_blue_i = [], []

    #     for r in tqdm(records_color, desc="Aggregating labelled cone points"):
    #         color = r["color"]

    #         if color not in {"orange", "blue"}:
    #             continue

    #         pcd_path = r.get("pcd_path")
    #         if not pcd_path or not pcd_path.exists():
    #             continue

    #         pts = load_pcd_binary(pcd_path)
    #         if pts is None or len(pts) < 3:
    #             continue

    #         distance = float(r["distance"])
    #         intensities = np.clip(pts[:, 3], 0, 255)

    #         if color == "orange":
    #             all_orange_d.extend([distance] * len(intensities))
    #             all_orange_i.extend(intensities)
    #         elif color == "blue":
    #             all_blue_d.extend([distance] * len(intensities))
    #             all_blue_i.extend(intensities)

    #     all_orange_d = np.asarray(all_orange_d, dtype=np.float64)
    #     all_orange_i = np.asarray(all_orange_i, dtype=np.float64)
    #     all_blue_d = np.asarray(all_blue_d, dtype=np.float64)
    #     all_blue_i = np.asarray(all_blue_i, dtype=np.float64)

    #     if len(all_orange_i) == 0 or len(all_blue_i) == 0:
    #         print("Not enough orange and blue points for global aggregation.")
    #         return

    #     # Second pass: one global threshold for every orange point and every blue point.
    #     orange_tape_threshold = float(np.percentile(all_orange_i, 80))
    #     blue_tape_threshold = float(np.percentile(all_blue_i, 80))

    #     orange_tape_mask = all_orange_i >= orange_tape_threshold
    #     blue_tape_mask = all_blue_i >= blue_tape_threshold

    #     # Orange body: lower 80% of ALL pooled orange points.
    #     o_d = all_orange_d[~orange_tape_mask]
    #     o_i = all_orange_i[~orange_tape_mask]

    #     # Blue body: lower 80% of ALL pooled blue points.
    #     b_d = all_blue_d[~blue_tape_mask]
    #     b_i = all_blue_i[~blue_tape_mask]

    #     # White tape: upper 20% candidates from BOTH globally pooled colour sets.
    #     w_d = np.concatenate([
    #         all_orange_d[orange_tape_mask],
    #         all_blue_d[blue_tape_mask],
    #     ])
    #     w_i = np.concatenate([
    #         all_orange_i[orange_tape_mask],
    #         all_blue_i[blue_tape_mask],
    #     ])

    #     print(f"Orange pooled points: {len(all_orange_i)}")
    #     print(f"Blue pooled points:   {len(all_blue_i)}")
    #     print(f"Orange global tape threshold: {orange_tape_threshold:.2f}")
    #     print(f"Blue global tape threshold:   {blue_tape_threshold:.2f}")
    #     print(f"Orange body points: {len(o_i)}")
    #     print(f"Blue body points:   {len(b_i)}")
    #     print(f"White tape points:  {len(w_i)}")

    #     def power_decay(d: np.ndarray, a: float, b: float) -> np.ndarray:
    #         return a * np.power(np.maximum(d, 1e-6), -b)

    #     fits = {}
    #     groups = [
    #         ("Orange Body", o_d, o_i),
    #         ("Blue Body", b_d, b_i),
    #         ("White Tape", w_d, w_i),
    #     ]

    #     for name, d_arr, i_arr in groups:
    #         valid = (
    #             np.isfinite(d_arr)
    #             & np.isfinite(i_arr)
    #             & (d_arr > 0.0)
    #             & (i_arr >= 0.0)
    #         )
    #         d_fit = d_arr[valid]
    #         i_fit = i_arr[valid]

    #         if len(d_fit) < 6:
    #             print(f"Warning: insufficient points for {name} fit.")
    #             fits[name] = np.array([200.0, 1.0])
    #             continue

    #         try:
    #             popt, _ = curve_fit(
    #                 power_decay,
    #                 d_fit,
    #                 i_fit,
    #                 p0=[200.0, 1.0],
    #                 maxfev=10000,
    #             )
    #             fits[name] = popt
    #         except Exception as exc:
    #             print(f"Warning: fit failed for {name}: {exc}")
    #             fits[name] = np.array([200.0, 1.0])

    #     # Normalize each category using its independently fitted distance exponent.
    #     o_i_norm = np.clip(
    #         o_i * np.power(o_d / ref_distance, fits["Orange Body"][1]),
    #         0,
    #         255,
    #     )
    #     b_i_norm = np.clip(
    #         b_i * np.power(b_d / ref_distance, fits["Blue Body"][1]),
    #         0,
    #         255,
    #     )
    #     w_i_norm = np.clip(
    #         w_i * np.power(w_d / ref_distance, fits["White Tape"][1]),
    #         0,
    #         255,
    #     )

    #     max_distance = max(
    #         25.0,
    #         float(np.max(o_d)),
    #         float(np.max(b_d)),
    #         float(np.max(w_d)),
    #     )
    #     d_grid = np.linspace(1.0, max_distance, 200)

    #     # Graph 1: Combined raw and normalized data.
    #     fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    #     axes[0].scatter(
    #         o_d, o_i, alpha=0.12, color="tab:orange", s=10,
    #         label="Orange Body Points",
    #     )
    #     axes[0].scatter(
    #         b_d, b_i, alpha=0.12, color="tab:blue", s=10,
    #         label="Blue Body Points",
    #     )
    #     axes[0].scatter(
    #         w_d, w_i, alpha=0.12, color="gray", s=10,
    #         label="White Tape Candidates",
    #     )

    #     axes[0].plot(
    #         d_grid,
    #         power_decay(d_grid, *fits["Orange Body"]),
    #         color="darkorange",
    #         lw=2.5,
    #         label="Orange Body Fit",
    #     )
    #     axes[0].plot(
    #         d_grid,
    #         power_decay(d_grid, *fits["Blue Body"]),
    #         color="navy",
    #         lw=2.5,
    #         label="Blue Body Fit",
    #     )
    #     axes[0].plot(
    #         d_grid,
    #         power_decay(d_grid, *fits["White Tape"]),
    #         color="black",
    #         lw=2.5,
    #         linestyle="--",
    #         label="White Tape Fit",
    #     )

    #     axes[0].set_ylabel("Raw Intensity (0-255)")
    #     axes[0].set_title("Globally Aggregated LiDAR Intensity Decay")
    #     axes[0].legend(loc="upper right")

    #     axes[1].scatter(
    #         o_d, o_i_norm, alpha=0.12, color="tab:orange", s=10,
    #         label="Orange Body",
    #     )
    #     axes[1].scatter(
    #         b_d, b_i_norm, alpha=0.12, color="tab:blue", s=10,
    #         label="Blue Body",
    #     )
    #     axes[1].scatter(
    #         w_d, w_i_norm, alpha=0.12, color="gray", s=10,
    #         label="White Tape",
    #     )

    #     axes[1].axhline(
    #         y=140,
    #         color="red",
    #         linestyle=":",
    #         lw=2,
    #         label="Candidate threshold = 140",
    #     )
    #     axes[1].set_xlabel("Distance from LiDAR (m)")
    #     axes[1].set_ylabel(f"Normalized intensity at {ref_distance} m (0-255)")
    #     axes[1].set_title("Globally Aggregated Distance-Normalized Intensity")
    #     axes[1].legend(loc="upper right")

    #     plt.tight_layout()
    #     fig.savefig(
    #         self.output_dir / "combined_intensity_decay_and_normalized.png",
    #         dpi=300,
    #     )
    #     plt.close(fig)

    #     # Graph 2: Orange body vs pooled white tape.
    #     fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    #     axes[0].scatter(
    #         o_d, o_i, alpha=0.15, color="tab:orange", s=12,
    #         label="Orange Plastic Body",
    #     )
    #     axes[0].scatter(
    #         w_d, w_i, alpha=0.15, color="gray", s=12,
    #         label="White Tape",
    #     )
    #     axes[0].plot(
    #         d_grid,
    #         power_decay(d_grid, *fits["Orange Body"]),
    #         color="darkorange",
    #         lw=2.5,
    #     )
    #     axes[0].plot(
    #         d_grid,
    #         power_decay(d_grid, *fits["White Tape"]),
    #         color="black",
    #         lw=2.5,
    #         linestyle="--",
    #     )
    #     axes[0].set_xlabel("Distance (m)")
    #     axes[0].set_ylabel("Raw Intensity (0-255)")
    #     axes[0].set_title("Global: Orange Body vs. White Tape")
    #     axes[0].legend()

    #     sns.kdeplot(
    #         o_i_norm,
    #         ax=axes[1],
    #         color="tab:orange",
    #         fill=True,
    #         label="Orange Body",
    #     )
    #     sns.kdeplot(
    #         w_i_norm,
    #         ax=axes[1],
    #         color="gray",
    #         fill=True,
    #         label="White Tape",
    #     )
    #     axes[1].axvline(
    #         x=140,
    #         color="red",
    #         linestyle="--",
    #         label="Threshold = 140",
    #     )
    #     axes[1].set_xlabel(f"Normalized intensity at {ref_distance} m (0-255)")
    #     axes[1].set_ylabel("Density")
    #     axes[1].set_title("Normalized Distribution: Orange vs. White")
    #     axes[1].legend()

    #     plt.tight_layout()
    #     fig.savefig(self.output_dir / "orange_vs_white_intensity.png", dpi=300)
    #     plt.close(fig)

    #     # Graph 3: Blue body vs pooled white tape.
    #     fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    #     axes[0].scatter(
    #         b_d, b_i, alpha=0.15, color="tab:blue", s=12,
    #         label="Blue Plastic Body",
    #     )
    #     axes[0].scatter(
    #         w_d, w_i, alpha=0.15, color="gray", s=12,
    #         label="White Tape",
    #     )
    #     axes[0].plot(
    #         d_grid,
    #         power_decay(d_grid, *fits["Blue Body"]),
    #         color="navy",
    #         lw=2.5,
    #     )
    #     axes[0].plot(
    #         d_grid,
    #         power_decay(d_grid, *fits["White Tape"]),
    #         color="black",
    #         lw=2.5,
    #         linestyle="--",
    #     )
    #     axes[0].set_xlabel("Distance (m)")
    #     axes[0].set_ylabel("Raw Intensity (0-255)")
    #     axes[0].set_title("Global: Blue Body vs. White Tape")
    #     axes[0].legend()

    #     sns.kdeplot(
    #         b_i_norm,
    #         ax=axes[1],
    #         color="tab:blue",
    #         fill=True,
    #         label="Blue Body",
    #     )
    #     sns.kdeplot(
    #         w_i_norm,
    #         ax=axes[1],
    #         color="gray",
    #         fill=True,
    #         label="White Tape",
    #     )
    #     axes[1].axvline(
    #         x=140,
    #         color="red",
    #         linestyle="--",
    #         label="Threshold = 140",
    #     )
    #     axes[1].set_xlabel(f"Normalized intensity at {ref_distance} m (0-255)")
    #     axes[1].set_ylabel("Density")
    #     axes[1].set_title("Normalized Distribution: Blue vs. White")
    #     axes[1].legend()

    #     plt.tight_layout()
    #     fig.savefig(self.output_dir / "blue_vs_white_intensity.png", dpi=300)
    #     plt.close(fig)

    #     print(f"Global unmixing complete. Figures saved in: {self.output_dir}")

    # def analyze_and_plot_cone_intensity_unmixing(
    #     self,
    #     records_color: List[Dict],
    #     ref_distance: float = INTENSITY_REF_DISTANCE,
    # ):
    #     """
    #     Aggregate all orange and blue cone points separately.

    #     A separate two-component GMM is fitted to:
    #     - all orange cone point intensities
    #     - all blue cone point intensities

    #     For each colour, the lower-intensity component is the plastic body and
    #     the higher-intensity component is white-tape candidate points.

    #     Orange tape and blue tape are combined only after their respective
    #     colour-specific classifications.
    #     """
    #     print("\nAggregating Orange and Blue clusters for separate GMM body/tape separation...")

    #     all_orange_d, all_orange_i = [], []
    #     all_blue_d, all_blue_i = [], []

    #     for r in tqdm(records_color, desc="Aggregating labelled cone points"):
    #         color = r["color"]

    #         if color not in {"orange", "blue"}:
    #             continue

    #         pcd_path = r.get("pcd_path")
    #         if not pcd_path or not pcd_path.exists():
    #             continue

    #         pts = load_pcd_binary(pcd_path)
    #         if pts is None or len(pts) < 3:
    #             continue

    #         distance = float(r["distance"])
    #         intensities = np.clip(pts[:, 3], 0, 255).astype(np.float64)

    #         if color == "orange":
    #             all_orange_d.extend([distance] * len(intensities))
    #             all_orange_i.extend(intensities)
    #         else:
    #             all_blue_d.extend([distance] * len(intensities))
    #             all_blue_i.extend(intensities)

    #     all_orange_d = np.asarray(all_orange_d, dtype=np.float64)
    #     all_orange_i = np.asarray(all_orange_i, dtype=np.float64)

    #     all_blue_d = np.asarray(all_blue_d, dtype=np.float64)
    #     all_blue_i = np.asarray(all_blue_i, dtype=np.float64)

    #     if len(all_orange_i) == 0 or len(all_blue_i) == 0:
    #         print("Not enough orange and blue points for separate GMM fitting.")
    #         return

    #     def weighted_gaussian_pdf(
    #         x: np.ndarray,
    #         mean: float,
    #         std: float,
    #         weight: float,
    #     ) -> np.ndarray:
    #         std = max(float(std), 1e-6)
    #         return (
    #             weight
    #             * np.exp(-0.5 * ((x - mean) / std) ** 2)
    #             / (std * np.sqrt(2.0 * np.pi))
    #         )

    #     def fit_color_gmm(
    #         intensities: np.ndarray,
    #         color_name: str,
    #     ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
    #         """
    #         Fit a 2-component GMM to a single colour's aggregate intensities.

    #         Returns:
    #             body_mask
    #             tape_mask
    #             GMM component intersection threshold
    #             fitted component information
    #         """
    #         if len(intensities) < 10:
    #             raise RuntimeError(
    #                 f"Not enough {color_name} points for two-component GMM fitting."
    #             )

    #         gmm = GaussianMixture(
    #             n_components=2,
    #             covariance_type="full",
    #             n_init=20,
    #             random_state=42,
    #             reg_covar=1e-4,
    #         )
    #         gmm.fit(intensities.reshape(-1, 1))

    #         means = gmm.means_.flatten()
    #         variances = gmm.covariances_.reshape(-1)
    #         stds = np.sqrt(np.maximum(variances, 1e-12))
    #         weights = gmm.weights_.flatten()

    #         component_order = np.argsort(means)
    #         body_component = int(component_order[0])
    #         tape_component = int(component_order[1])

    #         body_mean = float(means[body_component])
    #         body_std = float(stds[body_component])
    #         body_weight = float(weights[body_component])

    #         tape_mean = float(means[tape_component])
    #         tape_std = float(stds[tape_component])
    #         tape_weight = float(weights[tape_component])

    #         # The decision boundary is the weighted-Gaussian crossing between
    #         # the learned body and tape components, not an intensity percentile.
    #         x_search = np.linspace(body_mean, tape_mean, 10000)

    #         body_pdf = weighted_gaussian_pdf(
    #             x_search,
    #             body_mean,
    #             body_std,
    #             body_weight,
    #         )
    #         tape_pdf = weighted_gaussian_pdf(
    #             x_search,
    #             tape_mean,
    #             tape_std,
    #             tape_weight,
    #         )

    #         separation_threshold = float(
    #             x_search[np.argmin(np.abs(body_pdf - tape_pdf))]
    #         )

    #         posteriors = gmm.predict_proba(intensities.reshape(-1, 1))
    #         tape_probability = posteriors[:, tape_component]

    #         # Require both high-component membership and a value above the
    #         # learned intersection boundary.
    #         tape_mask = (
    #             (tape_probability >= 0.5)
    #             & (intensities >= separation_threshold)
    #         )
    #         body_mask = ~tape_mask

    #         info = {
    #             "body_mean": body_mean,
    #             "body_std": body_std,
    #             "body_weight": body_weight,
    #             "tape_mean": tape_mean,
    #             "tape_std": tape_std,
    #             "tape_weight": tape_weight,
    #             "threshold": separation_threshold,
    #         }

    #         print(f"\n--- {color_name.capitalize()} GMM Body/Tape Separation ---")
    #         print(
    #             f"{color_name.capitalize()} body component: "
    #             f"mean={body_mean:.2f}, std={body_std:.2f}, "
    #             f"weight={body_weight:.3f}"
    #         )
    #         print(
    #             f"{color_name.capitalize()} tape component: "
    #             f"mean={tape_mean:.2f}, std={tape_std:.2f}, "
    #             f"weight={tape_weight:.3f}"
    #         )
    #         print(
    #             f"{color_name.capitalize()} GMM separation threshold: "
    #             f"{separation_threshold:.2f}"
    #         )
    #         print(f"{color_name.capitalize()} body points: {np.sum(body_mask)}")
    #         print(f"{color_name.capitalize()} tape points: {np.sum(tape_mask)}")

    #         return body_mask, tape_mask, separation_threshold, info

    #     orange_body_mask, orange_tape_mask, orange_threshold, orange_gmm_info = (
    #         fit_color_gmm(all_orange_i, "orange")
    #     )

    #     blue_body_mask, blue_tape_mask, blue_threshold, blue_gmm_info = (
    #         fit_color_gmm(all_blue_i, "blue")
    #     )

    #     # Body points remain colour-specific.
    #     o_d = all_orange_d[orange_body_mask]
    #     o_i = all_orange_i[orange_body_mask]

    #     b_d = all_blue_d[blue_body_mask]
    #     b_i = all_blue_i[blue_body_mask]

    #     # White tape is pooled only after independent orange and blue GMM splits.
    #     orange_tape_d = all_orange_d[orange_tape_mask]
    #     orange_tape_i = all_orange_i[orange_tape_mask]

    #     blue_tape_d = all_blue_d[blue_tape_mask]
    #     blue_tape_i = all_blue_i[blue_tape_mask]

    #     w_d = np.concatenate([orange_tape_d, blue_tape_d])
    #     w_i = np.concatenate([orange_tape_i, blue_tape_i])

    #     if len(o_i) == 0 or len(b_i) == 0 or len(w_i) == 0:
    #         print("Separate GMM separation produced an empty group; cannot continue.")
    #         return

    #     print("\n--- Separate Orange / Blue GMM Summary ---")
    #     print(f"Orange pooled points: {len(all_orange_i)}")
    #     print(f"Blue pooled points:   {len(all_blue_i)}")
    #     print(f"Orange GMM threshold: {orange_threshold:.2f}")
    #     print(f"Blue GMM threshold:   {blue_threshold:.2f}")
    #     print(f"Orange body points:   {len(o_i)}")
    #     print(f"Orange tape points:   {len(orange_tape_i)}")
    #     print(f"Blue body points:     {len(b_i)}")
    #     print(f"Blue tape points:     {len(blue_tape_i)}")
    #     print(f"Combined white tape:  {len(w_i)}")

    #     # -------------------------------------------------------------------------
    #     # Distance decay fit.
    #     # The b >= 0 bound prevents an unphysical increasing intensity model.
    #     # -------------------------------------------------------------------------
    #     def power_decay(d: np.ndarray, a: float, b: float) -> np.ndarray:
    #         return a * np.power(np.maximum(d, 1e-6), -b)

    #     fits = {}
    #     groups = [
    #         ("Orange Body", o_d, o_i),
    #         ("Blue Body", b_d, b_i),
    #         ("White Tape", w_d, w_i),
    #     ]

    #     for name, d_arr, i_arr in groups:
    #         valid = (
    #             np.isfinite(d_arr)
    #             & np.isfinite(i_arr)
    #             & (d_arr > 0.0)
    #             & (i_arr >= 0.0)
    #         )

    #         d_fit = d_arr[valid]
    #         i_fit = i_arr[valid]

    #         if len(d_fit) < 6:
    #             print(f"Warning: insufficient points for {name} fit.")
    #             fits[name] = np.array([float(np.mean(i_fit)), 0.0])
    #             continue

    #         try:
    #             popt, _ = curve_fit(
    #                 power_decay,
    #                 d_fit,
    #                 i_fit,
    #                 p0=[max(float(np.mean(i_fit)), 1.0), 0.5],
    #                 bounds=([0.0, 0.0], [255.0, 5.0]),
    #                 maxfev=20000,
    #             )
    #             fits[name] = popt
    #         except Exception as exc:
    #             print(f"Warning: fit failed for {name}: {exc}")
    #             fits[name] = np.array([float(np.mean(i_fit)), 0.0])

    #     # Normalize every group to the reference distance.
    #     # Given I(d) = a*d^(-b), I(ref) = I(d)*(d/ref)^b.
    #     o_i_norm = np.clip(
    #         o_i * np.power(o_d / ref_distance, fits["Orange Body"][1]),
    #         0,
    #         255,
    #     )
    #     b_i_norm = np.clip(
    #         b_i * np.power(b_d / ref_distance, fits["Blue Body"][1]),
    #         0,
    #         255,
    #     )
    #     w_i_norm = np.clip(
    #         w_i * np.power(w_d / ref_distance, fits["White Tape"][1]),
    #         0,
    #         255,
    #     )

    #     max_distance = max(
    #         25.0,
    #         float(np.max(o_d)),
    #         float(np.max(b_d)),
    #         float(np.max(w_d)),
    #     )
    #     d_grid = np.linspace(1.0, max_distance, 200)

    #     # -------------------------------------------------------------------------
    #     # Graph 1: combined raw and normalized data.
    #     # -------------------------------------------------------------------------
    #     fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    #     axes[0].scatter(
    #         o_d,
    #         o_i,
    #         alpha=0.12,
    #         color="tab:orange",
    #         s=10,
    #         label="Orange Body Points",
    #     )
    #     axes[0].scatter(
    #         b_d,
    #         b_i,
    #         alpha=0.12,
    #         color="tab:blue",
    #         s=10,
    #         label="Blue Body Points",
    #     )
    #     axes[0].scatter(
    #         w_d,
    #         w_i,
    #         alpha=0.12,
    #         color="gray",
    #         s=10,
    #         label="White Tape Points",
    #     )

    #     axes[0].plot(
    #         d_grid,
    #         power_decay(d_grid, *fits["Orange Body"]),
    #         color="darkorange",
    #         lw=2.5,
    #         label="Orange Body Fit",
    #     )
    #     axes[0].plot(
    #         d_grid,
    #         power_decay(d_grid, *fits["Blue Body"]),
    #         color="navy",
    #         lw=2.5,
    #         label="Blue Body Fit",
    #     )
    #     axes[0].plot(
    #         d_grid,
    #         power_decay(d_grid, *fits["White Tape"]),
    #         color="black",
    #         lw=2.5,
    #         linestyle="--",
    #         label="White Tape Fit",
    #     )

    #     axes[0].axhline(
    #         orange_threshold,
    #         color="darkorange",
    #         linestyle=":",
    #         lw=2.0,
    #         label=f"Orange GMM Threshold = {orange_threshold:.1f}",
    #     )
    #     axes[0].axhline(
    #         blue_threshold,
    #         color="navy",
    #         linestyle=":",
    #         lw=2.0,
    #         label=f"Blue GMM Threshold = {blue_threshold:.1f}",
    #     )

    #     axes[0].set_ylabel("Raw Intensity (0-255)")
    #     axes[0].set_title("Globally Aggregated LiDAR Intensity: Separate Orange/Blue GMM")
    #     axes[0].legend(loc="upper right")

    #     axes[1].scatter(
    #         o_d,
    #         o_i_norm,
    #         alpha=0.12,
    #         color="tab:orange",
    #         s=10,
    #         label="Orange Body",
    #     )
    #     axes[1].scatter(
    #         b_d,
    #         b_i_norm,
    #         alpha=0.12,
    #         color="tab:blue",
    #         s=10,
    #         label="Blue Body",
    #     )
    #     axes[1].scatter(
    #         w_d,
    #         w_i_norm,
    #         alpha=0.12,
    #         color="gray",
    #         s=10,
    #         label="White Tape",
    #     )

    #     axes[1].set_xlabel("Distance from LiDAR (m)")
    #     axes[1].set_ylabel(f"Normalized Intensity at {ref_distance} m (0-255)")
    #     axes[1].set_title("Distance-Normalized Intensity After Separate GMM Separation")
    #     axes[1].legend(loc="upper right")

    #     plt.tight_layout()
    #     fig.savefig(
    #         self.output_dir / "combined_intensity_decay_and_normalized.png",
    #         dpi=300,
    #     )
    #     plt.close(fig)

    #     # -------------------------------------------------------------------------
    #     # Graph 2: orange body vs orange-derived white tape.
    #     # -------------------------------------------------------------------------
    #     fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    #     axes[0].scatter(
    #         o_d,
    #         o_i,
    #         alpha=0.15,
    #         color="tab:orange",
    #         s=12,
    #         label="Orange Plastic Body",
    #     )
    #     axes[0].scatter(
    #         orange_tape_d,
    #         orange_tape_i,
    #         alpha=0.15,
    #         color="gray",
    #         s=12,
    #         label="Orange-Derived White Tape",
    #     )
    #     axes[0].plot(
    #         d_grid,
    #         power_decay(d_grid, *fits["Orange Body"]),
    #         color="darkorange",
    #         lw=2.5,
    #     )
    #     axes[0].axhline(
    #         orange_threshold,
    #         color="darkorange",
    #         linestyle="--",
    #         lw=2.0,
    #         label=f"Orange GMM Threshold = {orange_threshold:.1f}",
    #     )
    #     axes[0].set_xlabel("Distance (m)")
    #     axes[0].set_ylabel("Raw Intensity (0-255)")
    #     axes[0].set_title("Orange GMM: Orange Body vs. White Tape")
    #     axes[0].legend()

    #     orange_tape_i_norm = np.clip(
    #         orange_tape_i
    #         * np.power(
    #             orange_tape_d / ref_distance,
    #             fits["White Tape"][1],
    #         ),
    #         0,
    #         255,
    #     )

    #     sns.kdeplot(
    #         o_i_norm,
    #         ax=axes[1],
    #         color="tab:orange",
    #         fill=True,
    #         label="Orange Body",
    #     )
    #     sns.kdeplot(
    #         orange_tape_i_norm,
    #         ax=axes[1],
    #         color="gray",
    #         fill=True,
    #         label="Orange-Derived White Tape",
    #     )
    #     axes[1].set_xlabel(f"Normalized Intensity at {ref_distance} m (0-255)")
    #     axes[1].set_ylabel("Density")
    #     axes[1].set_title("Normalized Distribution: Orange vs. White")
    #     axes[1].legend()

    #     plt.tight_layout()
    #     fig.savefig(self.output_dir / "orange_vs_white_intensity.png", dpi=300)
    #     plt.close(fig)

    #     # -------------------------------------------------------------------------
    #     # Graph 3: blue body vs blue-derived white tape.
    #     # -------------------------------------------------------------------------
    #     fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    #     axes[0].scatter(
    #         b_d,
    #         b_i,
    #         alpha=0.15,
    #         color="tab:blue",
    #         s=12,
    #         label="Blue Plastic Body",
    #     )
    #     axes[0].scatter(
    #         blue_tape_d,
    #         blue_tape_i,
    #         alpha=0.15,
    #         color="gray",
    #         s=12,
    #         label="Blue-Derived White Tape",
    #     )
    #     axes[0].plot(
    #         d_grid,
    #         power_decay(d_grid, *fits["Blue Body"]),
    #         color="navy",
    #         lw=2.5,
    #     )
    #     axes[0].axhline(
    #         blue_threshold,
    #         color="navy",
    #         linestyle="--",
    #         lw=2.0,
    #         label=f"Blue GMM Threshold = {blue_threshold:.1f}",
    #     )
    #     axes[0].set_xlabel("Distance (m)")
    #     axes[0].set_ylabel("Raw Intensity (0-255)")
    #     axes[0].set_title("Blue GMM: Blue Body vs. White Tape")
    #     axes[0].legend()

    #     blue_tape_i_norm = np.clip(
    #         blue_tape_i
    #         * np.power(
    #             blue_tape_d / ref_distance,
    #             fits["White Tape"][1],
    #         ),
    #         0,
    #         255,
    #     )

    #     sns.kdeplot(
    #         b_i_norm,
    #         ax=axes[1],
    #         color="tab:blue",
    #         fill=True,
    #         label="Blue Body",
    #     )
    #     sns.kdeplot(
    #         blue_tape_i_norm,
    #         ax=axes[1],
    #         color="gray",
    #         fill=True,
    #         label="Blue-Derived White Tape",
    #     )
    #     axes[1].set_xlabel(f"Normalized Intensity at {ref_distance} m (0-255)")
    #     axes[1].set_ylabel("Density")
    #     axes[1].set_title("Normalized Distribution: Blue vs. White")
    #     axes[1].legend()

    #     plt.tight_layout()
    #     fig.savefig(self.output_dir / "blue_vs_white_intensity.png", dpi=300)
    #     plt.close(fig)

    #     print(f"Separate orange/blue GMM unmixing complete. Figures saved in: {self.output_dir}")

    

    def fit_and_plot_intensity_decay(self, records: List[Dict[str, float]]) -> Dict[str, Dict]:
        print("\n⚡ Modeling Intensity Decay (via AIC)...")
        decay_params = {}

        colors = sorted(list({r["color"] for r in records}))
        colors_map = {"orange": "tab:orange", "blue": "tab:blue", "yellow": "gold"}

        plt.figure(figsize=(11, 7))

        for color in colors:
            color_records = [r for r in records if r["color"] == color]
            d = np.array([r["distance"] for r in color_records], dtype=np.float64)
            i_avg = np.array([r["avg_intensity"] for r in color_records], dtype=np.float64)

            mask = (d > 0.5) & (d < 30.0) & (i_avg > 0)
            d_c, i_c = d[mask], i_avg[mask]

            if len(d_c) < 5:
                continue

            name, popt, aic, r2, func = fit_best_model(
                d_c, i_c,
                models={k: v for k, v in CANDIDATE_MODELS.items() if k != "inverse_square"},
            )
            decay_params[color] = {"model": name, "params": popt, "aic": aic, "r2": r2, "func": func}

            label = CANDIDATE_MODELS[name]["label"](popt).replace("N(d)", "I(d)")
            print(f"   Color [{color.upper()}]: best model = {name} (AIC={aic:.2f}, R^2={r2:.4f})  {label}")

            plt.scatter(d_c, i_c, alpha=0.25, color=colors_map.get(color, "gray"))

            d_grid = np.linspace(d_c.min(), d_c.max(), 150)
            plt.plot(d_grid, func(d_grid, *popt), color=colors_map.get(color, "black"), lw=2.5,
                     label=f"{color.capitalize()} ({name})")

        plt.title("LiDAR Intensity Decay vs. Distance")
        plt.xlabel("Distance from LiDAR (m)")
        plt.ylabel("Raw Intensity")
        plt.legend()
        plt.tight_layout()

        plot_path = self.output_dir / "distance_vs_intensity_decay.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"   Saved plot: {plot_path}")

        return decay_params

    def generate_color_feature_stats(self, records: List[Dict[str, float]]):
        print("\n📊 Generating Per-Color Statistical Summaries (including Vertical Contrast)...")

        features_to_analyze = [
            "height", "width", "aspect_ratio", "volume_approx",
            "avg_intensity", "std_intensity", "mid_intensity", "reflective_pct",
            "contrast", "contrast_bot_mid", "contrast_mid_top", "contrast_bot_top"
        ]

        by_color = {}
        for r in records:
            c = r["color"]
            if c not in by_color:
                by_color[c] = []
            by_color[c].append(r)

        summary_rows = []

        print("\n" + "=" * 95)
        print(f"{'Color':<10} | {'Feature':<24} | {'Mean':<8} | {'Std':<8} | {'Median':<8} | {'IQR':<8} | {'Min':<8} | {'Max':<8}")
        print("=" * 95)

        for color, color_records in sorted(by_color.items()):
            for feat in features_to_analyze:
                vals = np.array([r[feat] for r in color_records if feat in r and np.isfinite(r[feat])], dtype=np.float64)
                if vals.size == 0:
                    continue
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals))
                med_val = float(np.median(vals))
                iqr_val = float(stats.iqr(vals))
                min_val = float(np.min(vals))
                max_val = float(np.max(vals))

                summary_rows.append({
                    "Color": color,
                    "Feature": feat,
                    "Mean": mean_val,
                    "Std": std_val,
                    "Median": med_val,
                    "IQR": iqr_val,
                    "Min": min_val,
                    "Max": max_val
                })

                print(f"{color:<10} | {feat:<24} | {mean_val:<8.3f} | {std_val:<8.3f} | {med_val:<8.3f} | {iqr_val:<8.3f} | {min_val:<8.3f} | {max_val:<8.3f}")

        print("=" * 95 + "\n")

        csv_path = self.output_dir / "color_feature_statistics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Color", "Feature", "Mean", "Std", "Median", "IQR", "Min", "Max"])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"   Saved CSV table: {csv_path}")

        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        plot_feats = [
            "height", "width", "avg_intensity",
            "std_intensity", "mid_intensity", "reflective_pct",
            "contrast", "contrast_bot_mid", "contrast_bot_top"
        ]

        colors = sorted(list(by_color.keys()))
        color_map = {"orange": "tab:orange", "blue": "tab:blue", "yellow": "gold"}

        for ax, feat in zip(axes.flat, plot_feats):
            data_to_plot = [[r[feat] for r in by_color[c] if feat in r and np.isfinite(r[feat])] for c in colors]
            bplot = ax.boxplot(data_to_plot, labels=[c.capitalize() for c in colors], patch_artist=True)
            
            for patch, c in zip(bplot['boxes'], colors):
                patch.set_facecolor(color_map.get(c, "gray"))
                patch.set_alpha(0.7)

            ax.set_title(f"Distribution of {feat.replace('_', ' ').capitalize()}", fontsize=11, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.output_dir / "color_feature_distributions.png", dpi=300)
        plt.close()
        print(f"   Saved distribution plot: {self.output_dir / 'color_feature_distributions.png'}")


def main():
    print(f"\n🚀 Running Cone Statistical Analysis on Project Root: {REPO_ROOT}")
    analyzer = ConeDatasetAnalyzer(REPO_ROOT / "Dataset")

    # Step 1: Distance vs Point Count -- automatic best-fit model search (AIC)
    records_all = analyzer.collect_all_pcds_for_point_estimation()
    if records_all:
        analyzer.fit_distance_point_estimator(records_all)
    else:
        print("❌ No dataset PCD files found for point estimator modeling.")
        return

    # Step 2: Color Dataset Collection
    records_color = analyzer.collect_labeled_color_pcds()
    if records_color:
        # Step 3: Intensity Unmixing & Distribution Plots (White vs Orange vs Blue)
        analyzer.analyze_and_plot_cone_intensity_unmixing(records_color)

        # Step 4: Cluster-Level Intensity Decay Modeling
        analyzer.fit_and_plot_intensity_decay(records_color)

        # Step 5: Summary Feature Statistics
        analyzer.generate_color_feature_stats(records_color)
    else:
        print("⚠️ No labeled color files found in Processed/Color or raw.")

    print("\n✅ Analysis Complete! All figures and statistics saved to:")
    print(f"   {REPO_ROOT / 'figures' / 'statistical_analysis'}\n")


if __name__ == "__main__":
    main()