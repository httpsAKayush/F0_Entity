"""
decision/threshold_tuning.py — Decision threshold grid search for F0.5.

Tunes the match probability threshold (and optionally a per-country threshold)
directly against macro F0.5 on a held-out validation split.

Key design decisions:
- Uses the EXACT same metric implementation as evaluation/metrics.py (imported,
  never re-implemented here) to prevent formula drift between tuning and eval.
- Threshold grid is configurable (from config.py).
- Per-country thresholds are supported when enabled in config AND the validation
  set has enough Source-1 entities per country.
- Returns a dict of thresholds that can be serialized to YAML for reproducibility.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from business_entity_resolution.evaluation.metrics import (
    EvaluationResult,
    build_ground_truth_dict,
    evaluate,
    f05_single,
)

logger = logging.getLogger(__name__)


def _make_predictions(
    scored_pairs: pd.DataFrame,
    source1_ids: list[str],
    threshold: float,
) -> dict[str, set[str]]:
    """
    Convert scored candidate pairs to prediction dict using *threshold*.

    Parameters
    ----------
    scored_pairs:
        DataFrame with columns: source1_entity_id, other_entity_id, score.
    source1_ids:
        Complete list of source1 entity ids to include in predictions.
    threshold:
        Pairs with score >= threshold are included as matches.

    Returns
    -------
    dict[str, set[str]]
        source1_entity_id → set of predicted matched ids.
    """
    predictions: dict[str, set[str]] = {eid: set() for eid in source1_ids}
    matches = scored_pairs[scored_pairs["score"] >= threshold]
    for _, row in matches.iterrows():
        s1id = row["source1_entity_id"]
        if s1id in predictions:
            predictions[s1id].add(row["other_entity_id"])
    return predictions


def tune_threshold(
    scored_pairs: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    source1_ids: list[str],
    threshold_grid: list[float],
    default_threshold: float = 0.5,
) -> tuple[float, EvaluationResult]:
    """
    Find the threshold on *threshold_grid* that maximizes macro F0.5.

    Parameters
    ----------
    scored_pairs:
        DataFrame with columns: source1_entity_id, other_entity_id, score.
        Contains all candidate pairs from the validation set with their
        model-predicted match probabilities.
    ground_truth_df:
        Validation-set ground truth (source1_entity_id, source_id, source_tag).
    source1_ids:
        All source1 entity ids in the validation set (including singletons).
    threshold_grid:
        List of threshold values to try.
    default_threshold:
        Fallback if validation data is too sparse for reliable tuning.

    Returns
    -------
    tuple[float, EvaluationResult]
        (best_threshold, evaluation_result_at_best_threshold)
    """
    gt_dict = build_ground_truth_dict(ground_truth_df, source1_ids)

    best_thresh = default_threshold
    best_f05 = -1.0
    best_result: Optional[EvaluationResult] = None

    for thresh in threshold_grid:
        preds = _make_predictions(scored_pairs, source1_ids, thresh)
        result = evaluate(preds, gt_dict)
        logger.debug("  threshold=%.2f  macro_f05=%.4f", thresh, result.macro_f05)
        if result.macro_f05 > best_f05:
            best_f05 = result.macro_f05
            best_thresh = thresh
            best_result = result

    if best_result is None:
        best_result = evaluate(
            _make_predictions(scored_pairs, source1_ids, best_thresh),
            gt_dict,
        )

    logger.info(
        "Threshold tuning: best_threshold=%.2f  macro_f05=%.4f  "
        "precision=%.4f  recall=%.4f  singleton_acc=%.4f",
        best_thresh, best_result.macro_f05,
        best_result.macro_precision, best_result.macro_recall,
        best_result.singleton_accuracy,
    )

    return best_thresh, best_result


def tune_per_country_thresholds(
    scored_pairs: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    source1_df: pd.DataFrame,
    threshold_grid: list[float],
    default_threshold: float = 0.5,
    min_country_size: int = 100,
) -> dict[str, float]:
    """
    Tune a separate threshold per country when enough validation data is available.

    Countries with fewer than *min_country_size* Source-1 entities fall back to
    the globally-tuned threshold.

    Parameters
    ----------
    scored_pairs:
        Scored candidate pairs for the validation set.
    ground_truth_df:
        Validation ground truth.
    source1_df:
        Source-1 validation DataFrame with entity_id and country_norm columns.
    threshold_grid:
        Grid of thresholds to search.
    default_threshold:
        Threshold to use for countries with insufficient validation data.
    min_country_size:
        Minimum Source-1 entities in a country partition to tune it separately.

    Returns
    -------
    dict[str, float]
        Mapping country_norm → best_threshold.  "_default" key holds the
        fallback threshold for unseen countries.
    """
    # First compute a global threshold as the fallback
    all_s1_ids = source1_df["entity_id"].tolist()
    global_thresh, _ = tune_threshold(
        scored_pairs, ground_truth_df, all_s1_ids, threshold_grid, default_threshold
    )

    country_thresholds: dict[str, float] = {"_default": global_thresh}

    countries = source1_df["country_norm"].value_counts()
    for country, count in countries.items():
        if count < min_country_size:
            logger.debug(
                "Country '%s' has only %d val entities (< %d), using global threshold %.2f",
                country, count, min_country_size, global_thresh,
            )
            country_thresholds[str(country)] = global_thresh
            continue

        # Filter to this country
        country_s1_ids = source1_df[source1_df["country_norm"] == country]["entity_id"].tolist()
        country_pairs = scored_pairs[
            scored_pairs["source1_entity_id"].isin(set(country_s1_ids))
        ]
        country_gt = ground_truth_df[
            ground_truth_df["source1_entity_id"].isin(set(country_s1_ids))
        ]

        if country_pairs.empty:
            country_thresholds[str(country)] = global_thresh
            continue

        best_thresh, result = tune_threshold(
            country_pairs, country_gt, country_s1_ids, threshold_grid, global_thresh
        )
        logger.info(
            "Country '%s': best_threshold=%.2f  macro_f05=%.4f",
            country, best_thresh, result.macro_f05,
        )
        country_thresholds[str(country)] = best_thresh

    return country_thresholds
