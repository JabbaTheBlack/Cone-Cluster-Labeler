#!/usr/bin/env python3
"""
Statistical Analysis of LiDAR Cone Clusters without Pandas Dependency.

Key Capabilities:
1. Distance-based Point Count Estimator: automatic best-fit model search
   (power law, inverse-square, exponential decay, rational decay) across all
   Dataset PCDs, selected by AIC (Akaike Information Criterion) so that
   models with more free parameters aren't unfairly favored over simpler
   ones just because they achieve a marginally higher R^2.
2. Distance-based Intensity Loss Compensation: models baseline average
   intensity decay per color (also selected via AIC) and produces a
   distance-compensated average intensity feature normalized to a reference
   distance.
3. Pure NumPy & Standard Library implementation (compatible with NumPy 1.21.5).
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
# All compensated intensities are rescaled to "what the intensity would be at
# this distance" using the fitted per-color decay model.
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

    # Three-band vertical intensity profiles (bottom / mid / top) are still
    # extracted here since they are cheap and may be useful for other
    # purposes, but the derived contrast_* ratios are NO LONGER used in the
    # statistical summary or plots: with low point counts per cluster (esp.
    # at range) the per-band split becomes noisy/unreliable, so they are
    # excluded from `generate_color_feature_stats`.
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
        # Kept in the raw record dict (cheap, may be useful elsewhere) but
        # deliberately excluded from generate_color_feature_stats() below.
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
    """
    Compute AIC for a least-squares fit, assuming normally distributed
    residuals:   AIC = n * ln(SS_res / n) + 2k
    Lower AIC indicates a better trade-off between fit quality and model
    complexity (number of free parameters k). This is what should be used to
    pick between models with different numbers of parameters (e.g. the
    2-parameter power law vs. the 3-parameter exponential/rational decay),
    since raw R^2 mechanically favors models with more free parameters even
    when the extra flexibility is just fitting noise.
    """
    n = len(y_true)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_res = max(ss_res, 1e-12)
    return n * np.log(ss_res / n) + 2 * k


def fit_best_model(d: np.ndarray, y: np.ndarray, models: Dict[str, Dict] = CANDIDATE_MODELS,
                    maxfev: int = 10000) -> Tuple[str, np.ndarray, float, float, Callable]:
    """
    Try all candidate models, return the one with the LOWEST AIC (best
    complexity-penalized fit), not simply the highest R^2. Returns
    (best_name, best_params, best_aic, best_r2, best_func).
    """
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
        """Collect distance & point count metrics from ALL PCD files into a list of dicts."""
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

    def collect_labeled_color_pcds(self) -> List[Dict[str, float]]:
        """Collect features strictly from Processed/Color and raw folders with labels."""
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
                    records.append(metrics)

        print(f"   Successfully built color dataset: {len(records)} samples.")
        return records

    def fit_distance_point_estimator(
        self, records: List[Dict[str, float]]
    ) -> Tuple[str, np.ndarray, float]:
        """
        Fits pure decay candidate models (Power Law via Log-Log Pearson,
        Exponential via Log-Linear Pearson, Inverse-Square, and Rational Decay)
        without additive (+c) constants. Evaluates fits in log-space using AIC
        and returns (best_name, best_params, best_r2).
        """
        print("\n📈 Searching for Best-Fit Point Count Estimator Model...")
        d = np.array([r["distance"] for r in records], dtype=np.float64)
        n = np.array([r["num_points"] for r in records], dtype=np.float64)

        valid_mask = (d > 0.5) & (d < 30.0) & (n >= 3)
        d_clean, n_clean = d[valid_mask], n[valid_mask]
        log_d = np.log(d_clean)
        log_n = np.log(n_clean)

        candidates = {}

        # 1. Power Law via Log-Log Pearson Regression: ln(N) = ln(a) - b * ln(d)
        r_loglog = float(np.corrcoef(log_d, log_n)[0, 1])
        slope_pl, intercept_pl = np.polyfit(log_d, log_n, 1)
        a_pl, b_pl = np.exp(intercept_pl), -slope_pl
        params_pl = np.array([a_pl, b_pl], dtype=np.float64)
        func_pl = lambda x, a, b: a * (x ** (-b))
        candidates["power_law_pearson"] = {
            "params": params_pl,
            "func": func_pl,
            "n_params": 2,
            "label": f"N(d) = {a_pl:.2f} * d^(-{b_pl:.3f})]"
        }

        # 2. Pure Exponential Decay via Log-Linear Pearson Regression: ln(N) = ln(a) - b * d
        r_loglin = float(np.corrcoef(d_clean, log_n)[0, 1])
        slope_exp, intercept_exp = np.polyfit(d_clean, log_n, 1)
        a_exp, b_exp = np.exp(intercept_exp), -slope_exp
        params_exp = np.array([a_exp, b_exp], dtype=np.float64)
        func_exp = lambda x, a, b: a * np.exp(-b * x)
        candidates["pure_exponential"] = {
            "params": params_exp,
            "func": func_exp,
            "n_params": 2,
            "label": f"N(d) = {a_exp:.2f} * exp(-{b_exp:.3f} d) [r_loglin={r_loglin:.3f}]"
        }

        # 3. Pure Inverse-Square Law: N(d) = a / d^2
        a_inv2 = float(np.exp(np.mean(log_n + 2.0 * log_d)))
        params_inv2 = np.array([a_inv2], dtype=np.float64)
        func_inv2 = lambda x, a: a / (x ** 2)
        candidates["inverse_square"] = {
            "params": params_inv2,
            "func": func_inv2,
            "n_params": 1,
            "label": f"N(d) = {a_inv2:.2f} / d^2"
        }

        # 4. Rational Decay: N(d) = a / (1 + b * d^2)
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

        # Evaluate all models in log-space to select the best via AIC
        results = []
        for name, spec in candidates.items():
            y_pred = spec["func"](d_clean, *spec["params"])
            log_y_pred = np.log(np.maximum(y_pred, 1e-6))

            aic = akaike_information_criterion(log_n, log_y_pred, spec["n_params"])
            r2 = r_squared(log_n, log_y_pred)
            results.append((name, spec["params"], aic, r2, spec["func"], spec["label"]))

        # Sort candidates by AIC (lowest is best)
        results.sort(key=lambda x: x[2])
        best_name, best_params, best_aic, best_r2, best_func, best_label = results[0]

        print("   --- Non-Linear Pure Decay Models (ranked by Log-Space AIC) ---")
        for name, params, aic, r2, func, label in results:
            marker = " <== BEST (lowest AIC)" if name == best_name else ""
            print(f"   [{name:<20}] AIC={aic:9.2f}  R^2={r2:.4f}  {label}{marker}")

        print(f"\n   ✅ Best model: {best_name}  (AIC={best_aic:.2f}, R^2={best_r2:.4f})")
        print(f"   {best_label}")

        # Plot raw scatter + winning pure-decay fit
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

    def fit_and_plot_intensity_decay(self, records: List[Dict[str, float]]) -> Dict[str, Dict]:
        """
        Fit the best-available decay model (selected via AIC) for
        avg_intensity vs distance, per color, and plot the decay curves.
        """
        print("\n⚡ Modeling Intensity Decay (via AIC)...")
        decay_params = {}

        colors = sorted(list({r["color"] for r in records}))
        # Standard matplotlib colors for consistency
        colors_map = {"orange": "tab:orange", "blue": "tab:blue", "yellow": "gold"}

        plt.figure(figsize=(11, 7))

        for color in colors:
            color_records = [r for r in records if r["color"] == color]
            d = np.array([r["distance"] for r in color_records], dtype=np.float64)
            i_avg = np.array([r["avg_intensity"] for r in color_records], dtype=np.float64)

            # Filter out extreme distances or invalid intensities
            mask = (d > 0.5) & (d < 30.0) & (i_avg > 0)
            d_c, i_c = d[mask], i_avg[mask]

            # Require at least 5 points to attempt a curve fit
            if len(d_c) < 5:
                continue

            # Fit the data
            name, popt, aic, r2, func = fit_best_model(
                d_c, i_c,
                models={k: v for k, v in CANDIDATE_MODELS.items() if k != "inverse_square"},
            )
            decay_params[color] = {"model": name, "params": popt, "aic": aic, "r2": r2, "func": func}

            label = CANDIDATE_MODELS[name]["label"](popt).replace("N(d)", "I(d)")
            print(f"   Color [{color.upper()}]: best model = {name} (AIC={aic:.2f}, R^2={r2:.4f})  {label}")

            # Plot raw scatter points
            plt.scatter(d_c, i_c, alpha=0.25, color=colors_map.get(color, "gray"))
            
            # Plot the best-fit decay curve
            d_grid = np.linspace(d_c.min(), d_c.max(), 150)
            plt.plot(d_grid, func(d_grid, *popt), color=colors_map.get(color, "black"), lw=2.5,
                     label=f"{color.capitalize()} ({name})")

        plt.title("LiDAR Intensity Decay vs. Distance")
        plt.xlabel("Distance from LiDAR (m)")
        plt.ylabel("Raw Intensity")
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.output_dir / "distance_vs_intensity_decay.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"   Saved plot: {plot_path}")

        return decay_params

    # def generate_color_feature_stats(self, records: List[Dict[str, float]]):
    #     """
    #     Compute summary statistics for features grouped by cone color without pandas.
    #     Generates clean summary CSV and boxplot distributions per feature.
    #     """
    #     print("\n📊 Generating Per-Color Statistical Summaries...")
        
    #     # Replaced 'avg_intensity_compensated' with 'std_intensity' since compensation logic was dropped
    #     features_to_analyze = [
    #         "height", "width", "aspect_ratio", "volume_approx",
    #         "avg_intensity", "std_intensity", "mid_intensity", "reflective_pct",
    #     ]

    #     by_color = {}
    #     for r in records:
    #         c = r["color"]
    #         if c not in by_color:
    #             by_color[c] = []
    #         by_color[c].append(r)

    #     summary_rows = []

    #     print("\n" + "=" * 95)
    #     print(f"{'Color':<10} | {'Feature':<24} | {'Mean':<8} | {'Std':<8} | {'Median':<8} | {'IQR':<8} | {'Min':<8} | {'Max':<8}")
    #     print("=" * 95)

    #     for color, color_records in sorted(by_color.items()):
    #         for feat in features_to_analyze:
    #             vals = np.array([r[feat] for r in color_records if feat in r], dtype=np.float64)
    #             if vals.size == 0:
    #                 continue
    #             mean_val = float(np.mean(vals))
    #             std_val = float(np.std(vals))
    #             med_val = float(np.median(vals))
    #             iqr_val = float(stats.iqr(vals))
    #             min_val = float(np.min(vals))
    #             max_val = float(np.max(vals))

    #             summary_rows.append({
    #                 "Color": color,
    #                 "Feature": feat,
    #                 "Mean": mean_val,
    #                 "Std": std_val,
    #                 "Median": med_val,
    #                 "IQR": iqr_val,
    #                 "Min": min_val,
    #                 "Max": max_val
    #             })

    #             print(f"{color:<10} | {feat:<24} | {mean_val:<8.3f} | {std_val:<8.3f} | {med_val:<8.3f} | {iqr_val:<8.3f} | {min_val:<8.3f} | {max_val:<8.3f}")

    #     print("=" * 95 + "\n")

    #     # Export CSV via stdlib csv.DictWriter
    #     csv_path = self.output_dir / "color_feature_statistics.csv"
    #     with open(csv_path, "w", newline="") as f:
    #         writer = csv.DictWriter(f, fieldnames=["Color", "Feature", "Mean", "Std", "Median", "IQR", "Min", "Max"])
    #         writer.writeheader()
    #         writer.writerows(summary_rows)
    #     print(f"   Saved CSV table: {csv_path}")

    #     # Visual distributions
    #     fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    #     plot_feats = ["height", "width", "avg_intensity", "std_intensity", "mid_intensity", "reflective_pct"]

    #     colors = sorted(list(by_color.keys()))
        
    #     # Map color names explicitly to prevent label-color mismatch
    #     color_map = {
    #         "orange": "tab:orange",
    #         "blue": "tab:blue",
    #         "yellow": "gold"
    #     }

    #     for ax, feat in zip(axes.flat, plot_feats):
    #         data_to_plot = [[r[feat] for r in by_color[c] if feat in r] for c in colors]
    #         bplot = ax.boxplot(data_to_plot, labels=[c.capitalize() for c in colors], patch_artist=True)
            
    #         # Explicitly set fill color based on the current label key
    #         for patch, c in zip(bplot['boxes'], colors):
    #             patch.set_facecolor(color_map.get(c, "gray"))
    #             patch.set_alpha(0.7)
                
    #         ax.set_title(f"Distribution of {feat.replace('_', ' ').capitalize()}", fontsize=12, fontweight="bold")
    #         ax.grid(True, linestyle="--", alpha=0.5)

    #     plt.tight_layout()
    #     plt.savefig(self.output_dir / "color_feature_distributions.png", dpi=300)
    #     plt.close()
    #     print(f"   Saved distribution plot: {self.output_dir / 'color_feature_distributions.png'}")
    
    def generate_color_feature_stats(self, records: List[Dict[str, float]]):
        """
        Compute summary statistics for features grouped by cone color without pandas.
        Generates clean summary CSV and boxplot distributions per feature (including contrast metrics).
        """
        print("\n📊 Generating Per-Color Statistical Summaries (including Vertical Contrast)...")
        
        # Re-included 'contrast' and vertical band contrast ratios
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

        # Export CSV via stdlib csv.DictWriter
        csv_path = self.output_dir / "color_feature_statistics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Color", "Feature", "Mean", "Std", "Median", "IQR", "Min", "Max"])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"   Saved CSV table: {csv_path}")

        # Visual distributions (3x3 grid to fit general geometry, intensity, and contrast)
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        plot_feats = [
            "height", "width", "avg_intensity",
            "std_intensity", "mid_intensity", "reflective_pct",
            "contrast", "contrast_bot_mid", "contrast_bot_top"
        ]

        colors = sorted(list(by_color.keys()))
        
        color_map = {
            "orange": "tab:orange",
            "blue": "tab:blue",
            "yellow": "gold"
        }

        for ax, feat in zip(axes.flat, plot_feats):
            data_to_plot = [[r[feat] for r in by_color[c] if feat in r and np.isfinite(r[feat])] for c in colors]
            bplot = ax.boxplot(data_to_plot, tick_labels=[c.capitalize() for c in colors], patch_artist=True)
            
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

    # Step 2 & 3: Color Statistics & Distance-Compensated Intensity
    records_color = analyzer.collect_labeled_color_pcds()
    if records_color:
        analyzer.fit_and_plot_intensity_decay(records_color)
        analyzer.generate_color_feature_stats(records_color)
    else:
        print("⚠️ No labeled color files found in Processed/Color or raw.")

    print("\n✅ Analysis Complete! All figures and statistics saved to:")
    print(f"   {REPO_ROOT / 'figures' / 'statistical_analysis'}\n")


if __name__ == "__main__":
    main()
