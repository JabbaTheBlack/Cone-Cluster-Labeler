#!/usr/bin/env python3
"""
Statistical Analysis of LiDAR Cone Clusters without Pandas Dependency.

Key Capabilities:
1. Distance-based Point Count Estimator: N(d) = a * d^b fitting across all Dataset PCDs.
2. Distance-based Intensity Loss Compensation: Models both baseline total intensity
   and 3-band vertical intensity profiles (accounting for center black/white strips).
3. Pure NumPy & Standard Library implementation (compatible with NumPy 1.21.5).
"""

import csv
import json
import os
import re
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

    # Three-band vertical intensity profiles (bottom / mid / top)
    # Critical for distinguishing cones with white reflective or black center strips
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


def power_law(d: np.ndarray, a: float, b: float) -> np.ndarray:
    """Power-law curve model: f(d) = a * d^b"""
    return a * np.power(d, b)


def inverse_square_law(d: np.ndarray, a: float) -> np.ndarray:
    """Theoretical LiDAR density decay model: f(d) = a / d^2"""
    return a / (d ** 2)


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

    def fit_distance_point_estimator(self, records: List[Dict[str, float]]) -> Tuple[float, float]:
        """Fit point count estimation formula using NumPy arrays directly."""
        print("\n📈 Fitting Point Count Estimator Model...")
        d = np.array([r["distance"] for r in records], dtype=np.float64)
        n = np.array([r["num_points"] for r in records], dtype=np.float64)

        valid_mask = (d > 0.5) & (d < 30.0) & (n >= 3)
        d_clean, n_clean = d[valid_mask], n[valid_mask]

        popt_power, _ = curve_fit(power_law, d_clean, n_clean, p0=[500.0, -2.0])
        popt_inv, _ = curve_fit(inverse_square_law, d_clean, n_clean, p0=[500.0])

        a_pow, b_pow = popt_power
        a_inv = popt_inv[0]

        print("   --- Fit Results ---")
        print(f"   [Empirical Power Law]  N(d) = {a_pow:.2f} * d^({b_pow:.3f})")
        print(f"   [Theoretical Model]    N(d) = {a_inv:.2f} / d^2")

        # Visualizing with matplotlib
        plt.figure(figsize=(10, 6))
        plt.scatter(d, n, alpha=0.2, color="gray", label="Observed Cones")

        d_grid = np.linspace(d_clean.min(), d_clean.max(), 200)
        plt.plot(d_grid, power_law(d_grid, a_pow, b_pow), 'r-', lw=2.5, label=f"Fit: N(d) = {a_pow:.1f} · d^({b_pow:.2f})")
        plt.plot(d_grid, inverse_square_law(d_grid, a_inv), 'b--', lw=2.0, label=f"Theoretical: N(d) = {a_inv:.1f} / d²")

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

        return a_pow, b_pow

    def fit_intensity_decay(self, records: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        """Fit intensity decay curves for overall baseline and center strip band."""
        print("\n⚡ Modeling Intensity Decay Loss over Distance...")
        decay_params = {}

        colors = sorted(list({r["color"] for r in records}))
        colors_map = {"orange": "tab:orange", "blue": "tab:blue", "yellow": "gold"}

        plt.figure(figsize=(11, 7))

        for color in colors:
            color_records = [r for r in records if r["color"] == color]
            d = np.array([r["distance"] for r in color_records], dtype=np.float64)
            i_avg = np.array([r["avg_intensity"] for r in color_records], dtype=np.float64)
            i_mid = np.array([r["mid_intensity"] for r in color_records], dtype=np.float64)

            mask = (d > 0.5) & (d < 30.0) & (i_avg > 0)
            d_c, i_c, i_mid_c = d[mask], i_avg[mask], i_mid[mask]

            if len(d_c) < 5:
                continue

            # Fit baseline average intensity
            popt_avg, _ = curve_fit(power_law, d_c, i_c, p0=[float(i_c.max()), -1.5], maxfev=5000)
            # Fit mid-band intensity (strip area)
            popt_mid, _ = curve_fit(power_law, d_c, i_mid_c, p0=[float(i_mid_c.max()), -1.5], maxfev=5000)

            decay_params[f"{color}_avg"] = popt_avg
            decay_params[f"{color}_mid"] = popt_mid

            print(f"   Color [{color.upper()} Baseline Avg]: I(d) = {popt_avg[0]:.2f} * d^({popt_avg[1]:.3f})")
            print(f"   Color [{color.upper()} Mid-Band Strip]: I(d) = {popt_mid[0]:.2f} * d^({popt_mid[1]:.3f})")

            # Plot Baseline Average
            plt.scatter(d_c, i_c, alpha=0.25, color=colors_map.get(color, "gray"))
            d_grid = np.linspace(d_c.min(), d_c.max(), 150)
            plt.plot(d_grid, power_law(d_grid, popt_avg[0], popt_avg[1]), color=colors_map.get(color, "black"), lw=2.5,
                     label=f"{color.capitalize()} Avg Fit (b={popt_avg[1]:.2f})")

            # Plot Mid-band (Strip) fit as dotted line
            plt.plot(d_grid, power_law(d_grid, popt_mid[0], popt_mid[1]), color=colors_map.get(color, "black"), ls="--", lw=1.5,
                     label=f"{color.capitalize()} Strip Fit (b={popt_mid[1]:.2f})")

        plt.title("LiDAR Intensity Loss Decay: Baseline Avg vs. Center Strip")
        plt.xlabel("Distance from LiDAR (m)")
        plt.ylabel("Raw Intensity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "distance_vs_intensity_decay.png", dpi=300)
        plt.close()
        print(f"   Saved plot: {self.output_dir / 'distance_vs_intensity_decay.png'}")

        return decay_params

    def generate_color_feature_stats(self, records: List[Dict[str, float]]):
        """Compute summary statistics for features grouped by cone color without pandas."""
        print("\n📊 Generating Per-Color Statistical Summaries...")
        features_to_analyze = [
            "height", "width", "aspect_ratio", "volume_approx",
            "avg_intensity", "std_intensity", "mid_intensity", "reflective_pct",
            "contrast_bot_mid", "contrast_mid_top", "contrast_bot_top"
        ]

        by_color = {}
        for r in records:
            c = r["color"]
            if c not in by_color:
                by_color[c] = []
            by_color[c].append(r)

        summary_rows = []

        print("\n" + "=" * 95)
        print(f"{'Color':<10} | {'Feature':<18} | {'Mean':<8} | {'Std':<8} | {'Median':<8} | {'IQR':<8} | {'Min':<8} | {'Max':<8}")
        print("=" * 95)

        for color, color_records in sorted(by_color.items()):
            for feat in features_to_analyze:
                vals = np.array([r[feat] for r in color_records], dtype=np.float64)
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

                print(f"{color:<10} | {feat:<18} | {mean_val:<8.3f} | {std_val:<8.3f} | {med_val:<8.3f} | {iqr_val:<8.3f} | {min_val:<8.3f} | {max_val:<8.3f}")

        print("=" * 95 + "\n")

        # Export CSV via stdlib csv.DictWriter
        csv_path = self.output_dir / "color_feature_statistics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Color", "Feature", "Mean", "Std", "Median", "IQR", "Min", "Max"])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"   Saved CSV table: {csv_path}")

        # Visual distributions
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        plot_feats = ["height", "width", "avg_intensity", "mid_intensity", "contrast_bot_top", "reflective_pct"]

        colors = sorted(list(by_color.keys()))
        palette = ["orange", "blue", "yellow"]

        for ax, feat in zip(axes.flat, plot_feats):
            data_to_plot = [[r[feat] for r in by_color[c]] for c in colors]
            bplot = ax.boxplot(data_to_plot, labels=colors, patch_artist=True)
            for patch, col in zip(bplot['boxes'], palette[:len(colors)]):
                patch.set_facecolor(col)
                patch.set_alpha(0.7)
            ax.set_title(f"Distribution of {feat}")

        plt.tight_layout()
        plt.savefig(self.output_dir / "color_feature_distributions.png", dpi=300)
        plt.close()
        print(f"   Saved distribution plot: {self.output_dir / 'color_feature_distributions.png'}")


def main():
    print(f"\n🚀 Running Cone Statistical Analysis on Project Root: {REPO_ROOT}")
    analyzer = ConeDatasetAnalyzer(REPO_ROOT / "Dataset")

    # Step 1: Distance vs Point Count Model across whole Dataset
    records_all = analyzer.collect_all_pcds_for_point_estimation()
    if records_all:
        a_pow, b_pow = analyzer.fit_distance_point_estimator(records_all)
    else:
        print("❌ No dataset PCD files found for point estimator modeling.")
        return

    # Step 2 & 3: Color Statistics & Intensity Loss Compensation
    records_color = analyzer.collect_labeled_color_pcds()
    if records_color:
        analyzer.fit_intensity_decay(records_color)
        analyzer.generate_color_feature_stats(records_color)
    else:
        print("⚠️ No labeled color files found in Processed/Color or raw.")

    print("\n✅ Analysis Complete! All figures and statistics saved to:")
    print(f"   {REPO_ROOT / 'figures' / 'statistical_analysis'}\n")


if __name__ == "__main__":
    main()