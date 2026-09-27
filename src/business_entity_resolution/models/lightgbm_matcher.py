"""
models/lightgbm_matcher.py — LightGBM implementation of MatcherModel.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from business_entity_resolution.models.base import MatcherModel

logger = logging.getLogger(__name__)


class LightGBMMatcher(MatcherModel):
    """
    LightGBM gradient-boosted tree classifier for entity pair matching.

    Parameters
    ----------
    **kwargs:
        LightGBM LGBMClassifier parameters. See config.py for defaults.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._params = kwargs
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMMatcher":
        """Fit LightGBM classifier. Returns self."""
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "LightGBM is required. Install with: pip install lightgbm"
            ) from exc

        logger.info(
            "Fitting LightGBM on %d samples, %d features, %.1f%% positive",
            len(y), X.shape[1], 100.0 * y.mean(),
        )
        self._model = LGBMClassifier(**self._params)
        self._model.fit(X, y)
        logger.info("LightGBM training complete.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return match probability (class 1) for each row."""
        if self._model is None:
            raise RuntimeError("LightGBMMatcher.fit() must be called before predict_proba().")
        proba = self._model.predict_proba(X)
        return proba[:, 1].astype(np.float32)

    def get_feature_importance(self, feature_names: list[str] | None = None) -> dict[str, float]:
        """Return feature importance dict (feature_name -> importance score)."""
        if self._model is None:
            return {}
        importances = self._model.feature_importances_
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(importances))]
        return dict(zip(feature_names, importances.tolist()))
