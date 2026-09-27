"""
models/ensemble.py — EnsembleMatcher combining multiple MatcherModels.

Combines registered MatcherModel instances via weighted average of their
predict_proba outputs.  Presents the same MatcherModel interface to downstream
code — callers don't need to know whether they're talking to one model or many.
"""

from __future__ import annotations

import logging

import numpy as np

from business_entity_resolution.models.base import MatcherModel

logger = logging.getLogger(__name__)


class EnsembleMatcher(MatcherModel):
    """
    Weighted ensemble of MatcherModel instances.

    Parameters
    ----------
    models:
        List of (model, weight) tuples.  Weights need not sum to 1 — they are
        normalized automatically.
    """

    def __init__(self, models: list[tuple[MatcherModel, float]]) -> None:
        if not models:
            raise ValueError("EnsembleMatcher requires at least one model.")
        self._models_weights = models
        total_weight = sum(w for _, w in models)
        if total_weight <= 0:
            raise ValueError("Sum of ensemble weights must be positive.")
        # Normalize weights
        self._normalized_weights = [w / total_weight for _, w in models]

    @classmethod
    def from_models_and_weights(
        cls, models: list[MatcherModel], weights: list[float]
    ) -> "EnsembleMatcher":
        """
        Convenience constructor from separate model and weight lists.

        Parameters
        ----------
        models:
            List of MatcherModel instances.
        weights:
            Weight for each model (must be same length as *models*).
        """
        if len(models) != len(weights):
            raise ValueError("models and weights must have the same length.")
        return cls(list(zip(models, weights)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleMatcher":
        """Fit all component models on the same training data."""
        for i, (model, _) in enumerate(self._models_weights):
            logger.info("Fitting ensemble component %d/%d: %s", i + 1, len(self._models_weights), type(model).__name__)
            model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return weighted average of component models' probabilities.

        Returns
        -------
        np.ndarray
            Shape (n_samples,) float32 probability of match.
        """
        result = np.zeros(len(X), dtype=np.float64)
        for (model, _), weight in zip(self._models_weights, self._normalized_weights):
            result += weight * model.predict_proba(X).astype(np.float64)
        return result.astype(np.float32)

    @property
    def component_models(self) -> list[MatcherModel]:
        """Return the list of component MatcherModel instances."""
        return [m for m, _ in self._models_weights]
