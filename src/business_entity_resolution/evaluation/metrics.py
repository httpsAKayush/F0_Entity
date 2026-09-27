"""
evaluation/metrics.py — Exact metric implementations for the pipeline.

F0.5 formula (exact, per spec):
    F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)

Computed per Source-1 entity then macro-averaged across all Source-1 entities.

Key behaviors:
- An entity with true empty match list (singleton): prediction=[] → score 1.0
- An entity with true empty match list but prediction≠[] → score 0.0
- Edge case: both true and predicted are empty → precision=1, recall=1, f0.5=1
- Edge case: true non-empty, predicted empty → precision=undefined, recall=0, f0.5=0
  (we handle undefined precision as 0 when predicted is empty but true is non-empty)

Singleton accuracy is tracked separately from overall F0.5 because it is an
easy source of silent score loss.

All functions in this module are PURE (no I/O, no global state).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Per-entity F0.5
# ---------------------------------------------------------------------------


def f05_single(
    predicted: set[str],
    true_matches: set[str],
) -> float:
    """
    Compute F0.5 for a single Source-1 entity.

    Parameters
    ----------
    predicted:
        Set of predicted matched entity ids.
    true_matches:
        Set of true matched entity ids.  Empty set means singleton.

    Returns
    -------
    float
        F0.5 score in [0, 1].
    """
    # Both empty → perfect score (correctly predicted singleton)
    if not predicted and not true_matches:
        return 1.0

    # Predicted empty, true non-empty → missed all matches
    if not predicted:
        return 0.0

    # True empty, predicted non-empty → false merge of a singleton
    if not true_matches:
        return 0.0

    # Normal case
    tp = len(predicted & true_matches)
    precision = tp / len(predicted)
    recall = tp / len(true_matches)

    if precision == 0.0 and recall == 0.0:
        return 0.0

    # F0.5 = (1 + 0.5²) × P × R / (0.5² × P + R) = 1.25 × P × R / (0.25 × P + R)
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f05


# ---------------------------------------------------------------------------
# Macro-averaged F0.5
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Result of a full evaluation run."""
    macro_f05: float
    macro_precision: float
    macro_recall: float
    singleton_accuracy: float
    n_entities: int
    n_singletons: int
    n_non_singletons: int
    per_entity_f05: Optional[dict[str, float]] = None


def evaluate(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
    return_per_entity: bool = False,
) -> EvaluationResult:
    """
    Compute macro-averaged F0.5 over all Source-1 entities.

    Parameters
    ----------
    predictions:
        dict mapping source1_entity_id → set of predicted matched entity ids.
        Every Source-1 entity must appear here (even singletons with empty set).
    ground_truth:
        dict mapping source1_entity_id → set of true matched entity ids.
        Singletons have empty sets.  Source-1 entities NOT in this dict are
        treated as singletons.
    return_per_entity:
        If True, include per-entity F0.5 scores in the result.

    Returns
    -------
    EvaluationResult
    """
    entity_ids = set(predictions.keys())
    f05_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    singleton_correct = 0
    n_singletons = 0
    n_non_singletons = 0
    per_entity: dict[str, float] = {}

    for eid in entity_ids:
        pred = predictions.get(eid, set())
        true = ground_truth.get(eid, set())

        f05 = f05_single(pred, true)
        f05_scores.append(f05)

        # Precision / recall for macro averages
        if pred or true:
            tp = len(pred & true)
            p = tp / len(pred) if pred else 0.0
            r = tp / len(true) if true else 0.0
        else:
            p, r = 1.0, 1.0

        precision_scores.append(p)
        recall_scores.append(r)

        # Singleton tracking
        if not true:
            n_singletons += 1
            if not pred:
                singleton_correct += 1
        else:
            n_non_singletons += 1

        if return_per_entity:
            per_entity[eid] = f05

    n = len(f05_scores)
    macro_f05 = sum(f05_scores) / n if n > 0 else 0.0
    macro_p = sum(precision_scores) / n if n > 0 else 0.0
    macro_r = sum(recall_scores) / n if n > 0 else 0.0
    singleton_acc = singleton_correct / n_singletons if n_singletons > 0 else 1.0

    return EvaluationResult(
        macro_f05=macro_f05,
        macro_precision=macro_p,
        macro_recall=macro_r,
        singleton_accuracy=singleton_acc,
        n_entities=n,
        n_singletons=n_singletons,
        n_non_singletons=n_non_singletons,
        per_entity_f05=per_entity if return_per_entity else None,
    )


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------


def build_ground_truth_dict(
    ground_truth_df: pd.DataFrame,
    source1_ids: list[str],
) -> dict[str, set[str]]:
    """
    Build a dict mapping source1_entity_id → set of matched entity ids.

    Source-1 entities NOT appearing in ground_truth_df are singletons (empty set).

    Parameters
    ----------
    ground_truth_df:
        DataFrame with columns: source1_entity_id, source_id.
    source1_ids:
        Complete list of all Source-1 entity ids (including singletons).

    Returns
    -------
    dict[str, set[str]]
    """
    # Initialise every S1 entity as a singleton (empty set)
    gt: dict[str, set[str]] = {eid: set() for eid in source1_ids}

    if ground_truth_df.empty:
        return gt

    # Vectorized groupby: group source_id values per source1_entity_id
    grouped = (
        ground_truth_df[["source1_entity_id", "source_id"]]
        .dropna(subset=["source1_entity_id", "source_id"])
        .groupby("source1_entity_id")["source_id"]
        .apply(set)
    )
    for s1id, id_set in grouped.items():
        gt[s1id] = id_set  # overwrites the empty-set default; also covers unseen S1 IDs
    return gt


def build_predictions_dict(
    matching_results_df: pd.DataFrame,
) -> dict[str, set[str]]:
    """
    Build predictions dict from a matching_results DataFrame in the official format.

    Parameters
    ----------
    matching_results_df:
        DataFrame with columns: source1_entity_id, matched_entity_ids.
        ``matched_entity_ids`` is a comma-separated string combining S2 and S3
        matched IDs, or an empty string for singletons.

    Returns
    -------
    dict[str, set[str]]
        source1_entity_id -> set of predicted matched entity IDs.
    """
    if matching_results_df.empty:
        return {}

    def _parse(cell) -> set[str]:
        s = str(cell).strip() if (cell is not None and str(cell) != "nan") else ""
        return {x.strip() for x in s.split(",") if x.strip()} if s else set()

    # Fully vectorized: apply _parse to the matched_entity_ids column only,
    # then zip with source1_entity_id — no row-level dict access needed.
    ids = matching_results_df["source1_entity_id"].tolist()
    col = "matched_entity_ids"
    if col not in matching_results_df.columns:
        return {eid: set() for eid in ids}
    parsed = matching_results_df[col].apply(_parse)
    return dict(zip(ids, parsed))



# ---------------------------------------------------------------------------
# Blocking recall ceiling diagnostic
# ---------------------------------------------------------------------------


def blocking_recall_ceiling(
    candidates: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    source_tag: str = "source2",
) -> float:
    """
    Compute the fraction of ground-truth pairs that survive candidate generation.

    This is the UPPER BOUND on recall achievable by any downstream model:
    if a true match is not in the candidate set, no model can recover it.

    Parameters
    ----------
    candidates:
        Candidate pairs DataFrame with columns: source1_entity_id, other_entity_id.
    ground_truth_df:
        Ground truth DataFrame with columns: source1_entity_id, source_id, source_tag.
    source_tag:
        "source2" or "source3" — which source to evaluate.

    Returns
    -------
    float
        Recall ceiling in [0, 1].  1.0 = all GT pairs are in the candidate set.
    """
    gt_filtered = ground_truth_df[ground_truth_df["source_tag"] == source_tag]
    if gt_filtered.empty:
        return 1.0  # no GT pairs → vacuously perfect

    gt_pairs = set(zip(gt_filtered["source1_entity_id"], gt_filtered["source_id"]))
    cand_pairs = set(zip(candidates["source1_entity_id"], candidates["other_entity_id"]))
    hits = len(gt_pairs & cand_pairs)
    return hits / len(gt_pairs)
