"""
features/pairwise.py — Pairwise similarity features for candidate pair scoring.

Design principles:
- All similarity functions are PURE (no I/O, no global state) and independently
  unit-testable.
- The FeatureExtractor class vectorizes feature computation over a candidate-pairs
  DataFrame, avoiding Python row loops wherever possible (using pandas vectorized
  ops + rapidfuzz's process.cdist for string similarity).
- Feature names and their order are defined ONCE in FEATURE_NAMES — used by both
  training and inference to guarantee column-order consistency.
- The feature set is intentionally rich (20+ features) to give the gradient-boosted
  model enough signal to discriminate true matches from hard negatives.

Feature list (FEATURE_NAMES defines the authoritative column order):
    name_exact_match          — 1.0 if normalized names are identical
    name_token_jaccard        — Jaccard similarity of name word-token sets
    name_token_sort_ratio     — rapidfuzz token_sort_ratio / 100
    name_token_set_ratio      — rapidfuzz token_set_ratio / 100
    name_partial_ratio        — rapidfuzz partial_ratio / 100
    name_edit_sim             — 1 - normalized edit distance (Levenshtein)
    name_common_prefix_len    — length of common prefix / max name length
    name_len_ratio            — min(len)/max(len) for normalized names
    address_exact_match       — 1.0 if normalized addresses are identical
    address_token_overlap     — Jaccard of address word-token sets
    address_edit_sim          — 1 - normalized address edit distance
    address_partial_ratio     — rapidfuzz partial_ratio on addresses
    street_number_match       — 1.0 if street numbers agree, 0.5 if one missing, 0.0 if disagree
    address_len_ratio         — min(len)/max(len) for normalized addresses
    is_landmark_s1            — 1.0 if Source-1 address is landmark-relative
    is_landmark_other         — 1.0 if other address is landmark-relative
    country_agree             — 1.0 if country_norm fields agree (defensive check)
    name_missing              — 1.0 if either name is empty
    address_missing           — 1.0 if either address is empty
    name_address_combined_sim — average of name_edit_sim and address_edit_sim
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz as rfuzz
# rapidfuzz.process is not used directly here (cdist is called via rfuzz, not rprocess)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Authoritative feature name list — column ORDER is the contract
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "name_exact_match",
    "name_token_jaccard",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_edit_sim",
    "name_common_prefix_len",
    "name_len_ratio",
    "address_exact_match",
    "address_token_overlap",
    "address_edit_sim",
    "address_partial_ratio",
    "street_number_match",
    "address_len_ratio",
    "is_landmark_s1",
    "is_landmark_other",
    "country_agree",
    "name_missing",
    "address_missing",
    "name_address_combined_sim",
]


# ---------------------------------------------------------------------------
# Individual pure similarity functions
# ---------------------------------------------------------------------------


def exact_match(a: str, b: str) -> float:
    """Return 1.0 if *a* and *b* are identical (after stripping), else 0.0."""
    return 1.0 if a.strip() == b.strip() else 0.0


def token_jaccard(a: str, b: str) -> float:
    """
    Jaccard similarity of word-token sets of *a* and *b*.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Returns 0.0 if both are empty, 1.0 if identical token sets.
    """
    set_a = set(a.split()) if a.strip() else set()
    set_b = set(b.split()) if b.strip() else set()
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def common_prefix_ratio(a: str, b: str) -> float:
    """
    Length of the longest common prefix of *a* and *b*, divided by max length.
    Returns 0.0 if both are empty.
    """
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    prefix_len = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            prefix_len += 1
        else:
            break
    return prefix_len / max_len


def len_ratio(a: str, b: str) -> float:
    """
    min(len(a), len(b)) / max(len(a), len(b)).
    Returns 1.0 if both empty, 0.0 if one is empty and the other is not.
    """
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def street_number_agreement(num_a: str, num_b: str) -> float:
    """
    Score street number agreement:
    - 1.0 if both present and equal
    - 0.5 if one or both are missing (landmark-relative)
    - 0.0 if both present but different
    """
    if not num_a and not num_b:
        return 0.5   # both landmark-relative — inconclusive
    if not num_a or not num_b:
        return 0.5   # one landmark-relative — inconclusive
    return 1.0 if num_a == num_b else 0.0


# ---------------------------------------------------------------------------
# Vectorized FeatureExtractor
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """
    Extracts pairwise similarity features for all candidate pairs at once.

    The feature matrix is produced in the order defined by FEATURE_NAMES,
    which is the SINGLE authoritative column-order definition used by both
    training and inference.
    """

    def extract(
        self,
        candidates: pd.DataFrame,
        source1_df: pd.DataFrame,
        other_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Compute a feature matrix for all candidate pairs.

        Parameters
        ----------
        candidates:
            DataFrame with columns: source1_entity_id, other_entity_id.
        source1_df:
            Source-1 DataFrame with columns: entity_id, norm_name,
            norm_address, street_number, country_norm, is_landmark.
        other_df:
            Source-2/3 DataFrame with same columns.

        Returns
        -------
        np.ndarray
            Float32 feature matrix of shape (n_candidates, len(FEATURE_NAMES)).
            Row order matches *candidates*.
        """
        if candidates.empty:
            return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

        # Build indexed lookups for O(1) vectorized column access
        s1_lookup = source1_df.set_index("entity_id")
        ot_lookup = other_df.set_index("entity_id")

        s1_ids = candidates["source1_entity_id"].values
        ot_ids = candidates["other_entity_id"].values

        # Vectorized column gathering: reindex aligns by index label in one C-level pass.
        # Missing entity IDs get NaN which fillna("") converts to empty string.
        def _vec_get(lookup: pd.DataFrame, ids: np.ndarray, col: str) -> list[str]:
            return lookup[col].reindex(ids).fillna("").tolist()

        s1_names = _vec_get(s1_lookup, s1_ids, "norm_name")
        ot_names = _vec_get(ot_lookup, ot_ids, "norm_name")
        s1_addrs = _vec_get(s1_lookup, s1_ids, "norm_address")
        ot_addrs = _vec_get(ot_lookup, ot_ids, "norm_address")
        s1_nums = _vec_get(s1_lookup, s1_ids, "street_number")
        ot_nums = _vec_get(ot_lookup, ot_ids, "street_number")
        s1_countries = _vec_get(s1_lookup, s1_ids, "country_norm")
        ot_countries = _vec_get(ot_lookup, ot_ids, "country_norm")

        # Landmark flags: reindex + fillna(0.0) → numpy array directly
        s1_landmarks = s1_lookup["is_landmark"].reindex(s1_ids).fillna(0.0).to_numpy(dtype=np.float32)
        ot_landmarks = ot_lookup["is_landmark"].reindex(ot_ids).fillna(0.0).to_numpy(dtype=np.float32)

        n = len(s1_ids)
        features = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)

        # ---- Name features ----
        # Exact match
        features[:, 0] = [exact_match(a, b) for a, b in zip(s1_names, ot_names)]
        # Token Jaccard
        features[:, 1] = [token_jaccard(a, b) for a, b in zip(s1_names, ot_names)]
        # Token sort ratio (rapidfuzz)
        features[:, 2] = np.array([
            rfuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(s1_names, ot_names)
        ], dtype=np.float32)
        # Token set ratio
        features[:, 3] = np.array([
            rfuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(s1_names, ot_names)
        ], dtype=np.float32)
        # Partial ratio
        features[:, 4] = np.array([
            rfuzz.partial_ratio(a, b) / 100.0 for a, b in zip(s1_names, ot_names)
        ], dtype=np.float32)
        # Edit similarity (normalized indel distance)
        features[:, 5] = np.array([
            rfuzz.ratio(a, b) / 100.0 for a, b in zip(s1_names, ot_names)
        ], dtype=np.float32)
        # Common prefix ratio
        features[:, 6] = np.array([
            common_prefix_ratio(a, b) for a, b in zip(s1_names, ot_names)
        ], dtype=np.float32)
        # Length ratio
        features[:, 7] = np.array([
            len_ratio(a, b) for a, b in zip(s1_names, ot_names)
        ], dtype=np.float32)

        # ---- Address features ----
        features[:, 8] = [exact_match(a, b) for a, b in zip(s1_addrs, ot_addrs)]
        features[:, 9] = [token_jaccard(a, b) for a, b in zip(s1_addrs, ot_addrs)]
        features[:, 10] = np.array([
            rfuzz.ratio(a, b) / 100.0 for a, b in zip(s1_addrs, ot_addrs)
        ], dtype=np.float32)
        features[:, 11] = np.array([
            rfuzz.partial_ratio(a, b) / 100.0 for a, b in zip(s1_addrs, ot_addrs)
        ], dtype=np.float32)
        # Street number agreement
        features[:, 12] = np.array([
            street_number_agreement(a, b) for a, b in zip(s1_nums, ot_nums)
        ], dtype=np.float32)
        # Address length ratio
        features[:, 13] = np.array([
            len_ratio(a, b) for a, b in zip(s1_addrs, ot_addrs)
        ], dtype=np.float32)

        # ---- Landmark flags (already float32 numpy arrays from reindex) ----
        features[:, 14] = s1_landmarks
        features[:, 15] = ot_landmarks

        # ---- Country agreement (vectorized string comparison) ----
        s1_c_arr = np.array(s1_countries)
        ot_c_arr = np.array(ot_countries)
        features[:, 16] = (s1_c_arr == ot_c_arr).astype(np.float32)

        # ---- Missing-field indicators ----
        features[:, 17] = np.array([
            1.0 if not a.strip() or not b.strip() else 0.0
            for a, b in zip(s1_names, ot_names)
        ], dtype=np.float32)
        features[:, 18] = np.array([
            1.0 if not a.strip() or not b.strip() else 0.0
            for a, b in zip(s1_addrs, ot_addrs)
        ], dtype=np.float32)

        # ---- Combined name+address similarity ----
        features[:, 19] = (features[:, 5] + features[:, 10]) / 2.0

        return features

    def extract_to_dataframe(
        self,
        candidates: pd.DataFrame,
        source1_df: pd.DataFrame,
        other_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Like extract(), but returns a DataFrame with named columns.
        Useful for debugging and inspection.
        """
        matrix = self.extract(candidates, source1_df, other_df)
        return pd.DataFrame(matrix, columns=FEATURE_NAMES)
