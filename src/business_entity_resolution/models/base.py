"""
models/base.py — MatcherModel abstract interface.

Every concrete model must implement fit() and predict_proba().
The EnsembleMatcher wraps multiple MatcherModels behind the same interface,
so downstream code never needs to know whether it's talking to a single model
or an ensemble.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class MatcherModel(ABC):
    """
    Abstract interface for a binary classification model that scores
    candidate entity pairs as match / non-match.

    Implement this interface to add new model types (e.g., random forest,
    neural network) without changing blocking, feature, or evaluation code.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MatcherModel":
        """
        Train the model on feature matrix *X* with labels *y*.

        Parameters
        ----------
        X:
            Feature matrix of shape (n_samples, n_features).
        y:
            Binary labels: 1 = true match, 0 = non-match.
            Shape: (n_samples,)

        Returns
        -------
        MatcherModel
            self (for chaining).
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return match probability for each row in *X*.

        Parameters
        ----------
        X:
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            1-D float array of shape (n_samples,) with values in [0, 1].
            Higher = more likely to be a true match.
        """
        ...
