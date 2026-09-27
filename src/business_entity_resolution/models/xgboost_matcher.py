"""
models/xgboost_matcher.py — XGBoost implementation of MatcherModel.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from business_entity_resolution.models.base import MatcherModel

logger = logging.getLogger(__name__)


class XGBoostMatcher(MatcherModel):
    """
    XGBoost gradient-boosted tree classifier for entity pair matching.

    Parameters
    ----------
    **kwargs:
        XGBoost XGBClassifier parameters. See config.py for defaults.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._params = kwargs
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostMatcher":
        """Fit XGBoost classifier. Returns self."""
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required. Install with: pip install xgboost"
            ) from exc

        logger.info(
            "Fitting XGBoost on %d samples, %d features, %.1f%% positive",
            len(y), X.shape[1], 100.0 * y.mean(),
        )
        self._model = XGBClassifier(**self._params)
        self._model.fit(X, y)
        logger.info("XGBoost training complete.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return match probability (class 1) for each row."""
        if self._model is None:
            raise RuntimeError("XGBoostMatcher.fit() must be called before predict_proba().")
        proba = self._model.predict_proba(X)
        return proba[:, 1].astype(np.float32)

    def get_feature_importance(self, feature_names: list[str] | None = None) -> dict[str, float]:
        """Return feature importance dict."""
        if self._model is None:
            return {}
        importances = self._model.feature_importances_
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(importances))]
        return dict(zip(feature_names, importances.tolist()))
