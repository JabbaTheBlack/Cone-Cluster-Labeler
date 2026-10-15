"""Vertical (Z) intensity profile of a cone cluster, built from LiDAR scan rings.

Points of one cluster lie on a few discrete elevation angles (one per laser), so each
ring gives one robust intensity sample at one height. Edge points of a ring hit the cone
silhouette at grazing incidence and read much darker, so they are trimmed.
"""
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre

from features import FEATURE_NAMES as BASELINE17_NAMES, extract_extended_features

RING_GAP_DEG = 0.4          # elevation gap separating two lasers (observed spacing 0.7–1.4 deg)
CONE_HEIGHT_M = 0.325       # small FS cone
PROFILE_BINS = 6
HEIGHT_MODES = ('minmax', 'apex')


@dataclass
class RingProfile:
    h: np.ndarray            # (R,) normalised ring height, 0 = bottom, 1 = top
    y: np.ndarray            # (R,) log1p(median intensity) of the trimmed ring
    n: np.ndarray            # (R,) points per ring
    h_points: np.ndarray     # (N,) normalised height of every point
    y_points: np.ndarray     # (N,) log1p(intensity) of every point
    distance: float          # horizontal range of the centroid [m]


def split_rings(points: np.ndarray, gap_deg: float = RING_GAP_DEG) -> list[np.ndarray]:
    xyz = points[:, :3].astype(np.float64)
    elevation = np.degrees(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])))
    order = np.argsort(elevation)
    cuts = np.flatnonzero(np.diff(elevation[order]) > gap_deg) + 1
    return np.split(order, cuts)


def normalise_height(z: np.ndarray, z_all: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'minmax':
        span = z_all.max() - z_all.min()
        if span < 1e-3:
            return np.full_like(z, 0.5)
        return (z - z_all.min()) / span
    if mode == 'apex':
        # Ground removal cuts the cone bottom unpredictably; the top is a steadier anchor.
        return 1.0 - (z_all.max() - z) / CONE_HEIGHT_M
    raise ValueError(f'Unknown height mode: {mode}')


def ring_profile(points: np.ndarray, height_mode: str = 'minmax') -> RingProfile:
    xyz = points[:, :3].astype(np.float64)
    intensity = points[:, 3].astype(np.float64)
    z_ring, y_ring, n_ring = [], [], []
    for idx in split_rings(points):
        if len(idx) >= 5:
            azimuth = np.arctan2(xyz[idx, 1], xyz[idx, 0])
            idx = idx[np.argsort(azimuth)][1:-1]
        z_ring.append(xyz[idx, 2].mean())
        y_ring.append(np.log1p(np.median(intensity[idx])))
        n_ring.append(len(idx))
    z_ring = np.asarray(z_ring)
    centroid = xyz.mean(axis=0)
    return RingProfile(
        h=normalise_height(z_ring, xyz[:, 2], height_mode),
        y=np.asarray(y_ring),
        n=np.asarray(n_ring),
        h_points=normalise_height(xyz[:, 2], xyz[:, 2], height_mode),
        y_points=np.log1p(np.clip(intensity, 0.0, None)),
        distance=float(np.hypot(centroid[0], centroid[1])),
    )


# ---------------------------------------------------------------------------
# Static (training-free) feature groups
# ---------------------------------------------------------------------------

def _otsu_split(values: np.ndarray) -> tuple[float, float, float]:
    """Best two-level split of sorted values: (low mean, high mean, high fraction)."""
    v = np.sort(values)
    n = len(v)
    if n < 4 or v[-1] - v[0] < 1e-6:
        return float(v.mean()), float(v.mean()), 0.0
    csum = np.cumsum(v)
    k = np.arange(1, n)
    mean_lo = csum[:-1] / k
    mean_hi = (csum[-1] - csum[:-1]) / (n - k)
    between = k * (n - k) * (mean_hi - mean_lo) ** 2
    best = int(np.argmax(between))
    return float(mean_lo[best]), float(mean_hi[best]), float((n - k[best]) / n)


def group_range(points, prof, ctx):
    return [prof.distance, np.log(len(points))]


def group_geometry(points, prof, ctx):
    xyz = points[:, :3]
    height = float(np.ptp(xyz[:, 2]))
    width = float(max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), 1e-3))
    return [height, width, height / width]


def group_profile(points, prof, ctx):
    level = float(np.median(prof.y_points))
    bins = np.clip((prof.h_points * PROFILE_BINS).astype(int), 0, PROFILE_BINS - 1)
    shape = [float(np.median(prof.y_points[bins == b])) - level if np.any(bins == b) else np.nan
             for b in range(PROFILE_BINS)]
    return [level, *shape]


def group_ring(points, prof, ctx):
    y = prof.y
    top, bottom = y[np.argmax(prof.h)], y[np.argmin(prof.h)]
    return [len(y), y.max(), y.min(), y.max() - y.min(), top, bottom, float(prof.h[np.argmax(y)])]


def group_bimodal(points, prof, ctx):
    lo, hi, frac = _otsu_split(prof.y_points)
    return [lo, hi, frac, hi - lo]


def group_baseline17(points, prof, ctx):
    return extract_extended_features(points).tolist()


def _relative_to(prefix):
    """Cluster intensity relative to reference clusters of the same scan (cancels per-recording gain)."""
    def group(points, prof, ctx):
        level = float(np.median(prof.y_points))
        ref = ctx.get(f'{prefix}_level', np.nan)
        ref_near = ctx.get(f'{prefix}_level_near', np.nan)
        return [level - ref, level - ref_near, prof.y.max() - ref, ctx.get(f'{prefix}_n', 0)]
    return group


STATIC_GROUPS = {
    'range': (group_range, ['distance', 'log_n_points']),
    'geometry': (group_geometry, ['height', 'width', 'aspect']),
    'profile': (group_profile, ['level'] + [f'shape_bin{b}' for b in range(PROFILE_BINS)]),
    'ring': (group_ring, ['n_rings', 'ring_max', 'ring_min', 'ring_contrast',
                          'ring_top', 'ring_bottom', 'ring_argmax_h']),
    'bimodal': (group_bimodal, ['otsu_low', 'otsu_high', 'otsu_high_frac', 'otsu_gap']),
    'baseline17': (group_baseline17, [f'b17_{n}' for n in BASELINE17_NAMES]),
    'scene': (_relative_to('scene'), ['scene_rel_level', 'scene_rel_level_near', 'scene_rel_ring_max', 'scene_n']),
    'yref': (_relative_to('yellow'), ['yref_rel_level', 'yref_rel_level_near', 'yref_rel_ring_max', 'yref_n']),
}


def static_features(points, prof, groups, ctx=None) -> np.ndarray:
    values = []
    for g in groups:
        values.extend(STATIC_GROUPS[g][0](points, prof, ctx or {}))
    return np.asarray(values, dtype=np.float64)


def static_feature_names(groups) -> list[str]:
    return [name for g in groups for name in STATIC_GROUPS[g][1]]


# ---------------------------------------------------------------------------
# Per-cluster stripe fit: log I(h) = b + a * exp(-(h - mu)^2 / 2 s^2)
# ---------------------------------------------------------------------------

class StripeFit:
    """Stripe position/width (mu, s) are shared and learned from training rings without labels;
    each cluster then only needs a 2-parameter linear fit (body level b, stripe contrast a)."""

    MU_GRID = np.linspace(0.2, 0.9, 15)
    S_GRID = np.array([0.06, 0.1, 0.15, 0.2, 0.3])
    feature_names = ['stripe_b', 'stripe_a', 'stripe_rmse']

    def __init__(self, min_rings_fit: int = 3):
        self.min_rings_fit = min_rings_fit
        self.mu = 0.5
        self.s = 0.15

    @staticmethod
    def _solve(h, y, mu, s):
        g = np.exp(-0.5 * ((h - mu) / s) ** 2)
        A = np.column_stack([np.ones_like(g), g])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        return coef, float(np.sqrt(np.mean(resid ** 2)))

    def fit(self, profiles):
        usable = [p for p in profiles if len(p.y) >= self.min_rings_fit]
        h = np.clip(np.concatenate([p.h for p in usable]), 0, 1)
        y = np.concatenate([p.y for p in usable])
        owner = np.concatenate([np.full(len(p.y), i) for i, p in enumerate(usable)])
        n = np.bincount(owner).astype(np.float64)
        sy, syy = np.bincount(owner, y), np.bincount(owner, y * y)
        best = (np.inf, self.mu, self.s)
        for mu in self.MU_GRID:
            for s in self.S_GRID:
                # Closed-form per-cluster least squares of y = b + a g from sufficient statistics.
                g = np.exp(-0.5 * ((h - mu) / s) ** 2)
                sg, sgg, sgy = np.bincount(owner, g), np.bincount(owner, g * g), np.bincount(owner, g * y)
                det = n * sgg - sg ** 2
                ok = det > 1e-9
                a = np.where(ok, (n * sgy - sg * sy) / np.where(ok, det, 1.0), 0.0)
                b = (sy - a * sg) / n
                sse = float(np.sum(syy - b * sy - a * sgy))
                if sse < best[0]:
                    best = (sse, float(mu), float(s))
        _, self.mu, self.s = best
        return self

    def transform_one(self, prof) -> list[float]:
        if len(prof.y) < 2:
            return [float(prof.y.mean()), np.nan, np.nan]
        (b, a), rmse = self._solve(np.clip(prof.h, 0, 1), prof.y, self.mu, self.s)
        return [float(b), float(a), rmse]

    def transform(self, profiles) -> np.ndarray:
        return np.array([self.transform_one(p) for p in profiles], dtype=np.float64)


# ---------------------------------------------------------------------------
# Class template: learn f_c(h, d) per colour, fit the observed ring profile to each
# ---------------------------------------------------------------------------

class ZProfileTemplateClassifier:
    """Generative per-class model of the ring profile:

        y_j = sum_k beta_ck P_k(2h_j - 1) + gamma_c0 + gamma_c1 log d + gamma_c2 (log d)^2 + u + e_j
        u ~ N(0, tau_c^2)   per-cluster offset (incidence angle, dirt, per-laser gain)
        e_j ~ N(0, sigma_c^2)

    Classification = marginal likelihood ratio of the observed rings under both classes.
    offset_mode: 'learned' (tau_c estimated from training clusters), 'shape' (tau -> inf,
    only the shape of the profile counts), 'none' (tau = 0, absolute level counts fully).
    """

    def __init__(self, offset_mode='learned', degree_h=4, degree_d=2, ridge=1e-2, prior='balanced'):
        self.offset_mode = offset_mode
        self.degree_h = degree_h
        self.degree_d = degree_d
        self.ridge = ridge
        self.prior = prior

    def _design(self, h, d):
        x = 2.0 * np.clip(h, 0.0, 1.0) - 1.0
        shape = legendre.legvander(x, self.degree_h)[:, 1:]
        logd = np.log(np.maximum(d, 0.5))
        dist = np.column_stack([logd ** p for p in range(self.degree_d + 1)])
        return np.hstack([shape, dist])

    def _profile_arrays(self, profiles):
        h = np.concatenate([p.h for p in profiles])
        y = np.concatenate([p.y for p in profiles])
        d = np.concatenate([np.full(len(p.y), p.distance) for p in profiles])
        owner = np.concatenate([np.full(len(p.y), i) for i, p in enumerate(profiles)])
        return h, y, d, owner

    def fit(self, profiles, y):
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.params_ = {}
        for c in self.classes_:
            profs = [p for p, lab in zip(profiles, y) if lab == c]
            h, t, d, owner = self._profile_arrays(profs)
            A = self._design(h, d)
            coef = np.linalg.solve(A.T @ A + self.ridge * np.eye(A.shape[1]), A.T @ t)
            resid = t - A @ coef
            counts = np.bincount(owner)
            u_mean = np.bincount(owner, weights=resid) / counts
            within = resid - u_mean[owner]
            multi = counts >= 2
            sigma2 = float(np.sum(within[multi[owner]] ** 2) / max(np.sum(counts[multi] - 1), 1))
            tau2 = float(max(np.var(u_mean) - np.mean(sigma2 / counts), 1e-4))
            self.params_[int(c)] = {'coef': coef, 'sigma2': max(sigma2, 1e-4), 'tau2': tau2,
                                    'n_clusters': len(profs), 'n_rings': len(t)}
        n1 = int(np.sum(y == 1))
        self.log_prior_ratio_ = 0.0 if self.prior == 'balanced' else float(np.log(n1 / (len(y) - n1)))
        return self

    def _loglik(self, prof, c):
        p = self.params_[c]
        r = prof.y - self._design(prof.h, np.full(len(prof.y), prof.distance)) @ p['coef']
        R, s2 = len(r), p['sigma2']
        if self.offset_mode == 'shape':
            quad = (r @ r - r.sum() ** 2 / R) / s2
            return -0.5 * (quad + (R - 1) * np.log(s2))
        t2 = p['tau2'] if self.offset_mode == 'learned' else 0.0
        quad = (r @ r - t2 / (s2 + R * t2) * r.sum() ** 2) / s2
        logdet = R * np.log(s2) + np.log1p(R * t2 / s2)
        return -0.5 * (quad + logdet)

    def decision_function(self, profiles):
        return np.array([self._loglik(p, 1) - self._loglik(p, 0) + self.log_prior_ratio_ for p in profiles])

    def predict_proba(self, profiles):
        score = np.clip(self.decision_function(profiles), -50, 50)
        p1 = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1 - p1, p1])

    def template_curve(self, c, h, distance):
        return self._design(h, np.full(len(h), distance)) @ self.params_[c]['coef']

    def describe(self) -> dict:
        return {
            'offset_mode': self.offset_mode, 'degree_h': self.degree_h, 'degree_d': self.degree_d,
            'ridge': self.ridge, 'prior': self.prior, 'log_prior_ratio': self.log_prior_ratio_,
            'basis': 'Legendre P_1..P_K(2h-1) + [1, log d, (log d)^2 ...]',
            'classes': {c: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
                        for c, p in self.params_.items()},
        }
