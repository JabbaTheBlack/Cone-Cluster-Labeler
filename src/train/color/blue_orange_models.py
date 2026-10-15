"""Blue-vs-orange approaches behind one interface: featurize(points) -> fit -> predict_proba."""
from dataclasses import replace
from typing import NamedTuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from zprofile import (RingProfile, StripeFit, ZProfileTemplateClassifier, ring_profile,
                      static_feature_names, static_features)


class ClusterRep(NamedTuple):
    static: np.ndarray
    prof: RingProfile
    context: dict           # scan-level references (see cone_dataset.scene_context)


def make_estimator(name: str, seed: int):
    if name == 'hgb':
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10,
            l2_regularization=1.0, class_weight='balanced', random_state=seed)
    if name == 'rf':
        return Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('rf', RandomForestClassifier(
                n_estimators=400, min_samples_leaf=3, max_features='sqrt',
                class_weight='balanced_subsample', n_jobs=-1, random_state=seed)),
        ])
    if name == 'logreg':
        return Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('lr', LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000)),
        ])
    if name == 'xgb':
        from xgboost import XGBClassifier
        # Fallback params of xgb_model.py; class balance is set in FeatureApproach.fit (scale_pos_weight).
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            tree_method='hist', random_state=seed, n_jobs=1)
    raise ValueError(f'Unknown model: {name}')


def _single_threaded(estimator):
    """Per-cluster latency must not include thread-pool start-up."""
    for key in estimator.get_params():
        if key == 'n_jobs' or key.endswith('__n_jobs'):
            estimator.set_params(**{key: 1})


class FeatureApproach:
    """Hand-crafted feature groups (+ optional per-cluster stripe fit) -> sklearn classifier."""

    def __init__(self, groups, model, height_mode='minmax', seed=42):
        self.groups = list(groups)
        self.static_groups = [g for g in self.groups if g != 'stripe']
        self.use_stripe = 'stripe' in self.groups
        self.model = model
        self.height_mode = height_mode
        self.seed = seed

    @property
    def cache_key(self):
        return (self.height_mode, tuple(self.static_groups))

    def featurize(self, points, context=None) -> ClusterRep:
        prof = ring_profile(points, self.height_mode)
        return ClusterRep(static_features(points, prof, self.static_groups, context), prof, context or {})

    @property
    def feature_names(self):
        return static_feature_names(self.static_groups) + (StripeFit.feature_names if self.use_stripe else [])

    def _matrix(self, reps):
        X = np.vstack([r.static for r in reps]) if self.static_groups else np.empty((len(reps), 0))
        if self.use_stripe:
            X = np.hstack([X, self.stripe_.transform([r.prof for r in reps])])
        return X

    def fit(self, reps, y):
        if self.use_stripe:
            self.stripe_ = StripeFit().fit([r.prof for r in reps])
        self.estimator_ = make_estimator(self.model, self.seed)
        if self.model == 'xgb':
            y_arr = np.asarray(y)
            self.estimator_.set_params(scale_pos_weight=float((y_arr == 0).sum() / max((y_arr == 1).sum(), 1)))
        self.estimator_.fit(self._matrix(reps), y)
        return self

    def predict_proba(self, reps) -> np.ndarray:
        return self.estimator_.predict_proba(self._matrix(reps))[:, 1]

    def prepare_for_latency(self):
        _single_threaded(self.estimator_)

    def describe(self) -> dict:
        info = {'feature_names': self.feature_names,
                'estimator': {k: repr(v) for k, v in self.estimator_.get_params(deep=True).items()
                              if not hasattr(v, 'fit')}}
        if self.use_stripe:
            info['stripe'] = {'mu': self.stripe_.mu, 's': self.stripe_.s}
        final = self.estimator_[-1] if isinstance(self.estimator_, Pipeline) else self.estimator_
        if hasattr(final, 'feature_importances_'):
            info['feature_importances'] = dict(zip(self.feature_names, final.feature_importances_.tolist()))
        if hasattr(final, 'coef_'):
            info['standardised_coefficients'] = dict(zip(self.feature_names, final.coef_[0].tolist()))
        return info


class TemplateApproach:
    """Z-profile template fit (per-class learned intensity function of height and distance)."""

    def __init__(self, offset_mode='learned', height_mode='minmax', degree_h=4, degree_d=2,
                 prior='balanced', reference=None, seed=42):
        self.offset_mode = offset_mode
        self.height_mode = height_mode
        self.degree_h = degree_h
        self.degree_d = degree_d
        self.prior = prior
        self.reference = reference            # None, 'scene' or 'yellow'

    @property
    def cache_key(self):
        return (self.height_mode, ())

    def featurize(self, points, context=None) -> ClusterRep:
        return ClusterRep(np.empty(0), ring_profile(points, self.height_mode), context or {})

    def _reference_level(self, rep) -> float:
        return float(rep.context.get(f'{self.reference}_level', np.nan))

    def _profiles(self, reps):
        if not self.reference:
            return [r.prof for r in reps]
        # Intensity relative to the other cones of the scan; scans with no other cone use the training median.
        levels = [self._reference_level(r) for r in reps]
        return [replace(r.prof, y=r.prof.y - (lv if np.isfinite(lv) else self.fallback_ref_))
                for r, lv in zip(reps, levels)]

    def fit(self, reps, y):
        levels = np.array([self._reference_level(r) for r in reps]) if self.reference else np.array([np.nan])
        self.fallback_ref_ = float(np.nanmedian(levels)) if np.isfinite(levels).any() else 0.0
        self.model_ = ZProfileTemplateClassifier(self.offset_mode, self.degree_h, self.degree_d, prior=self.prior)
        self.model_.fit(self._profiles(reps), y)
        return self

    def predict_proba(self, reps) -> np.ndarray:
        return self.model_.predict_proba(self._profiles(reps))[:, 1]

    def prepare_for_latency(self):
        pass

    def describe(self) -> dict:
        return {**self.model_.describe(), 'reference': self.reference,
                'scene_fallback_ref': self.fallback_ref_}


def build_approach(config: dict, seed: int):
    if config['approach'] == 'features':
        return FeatureApproach(config['groups'], config['model'], config.get('height_mode', 'minmax'), seed)
    if config['approach'] == 'ztemplate':
        return TemplateApproach(config.get('offset_mode', 'learned'), config.get('height_mode', 'minmax'),
                                config.get('degree_h', 4), config.get('degree_d', 2),
                                config.get('prior', 'balanced'), config.get('reference'), seed)
    raise ValueError(f"Unknown approach: {config['approach']}")
