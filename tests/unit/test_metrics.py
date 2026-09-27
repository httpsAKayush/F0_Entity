"""
tests/unit/test_metrics.py — Unit tests for evaluation/metrics.py.

Tests the EXACT F0.5 formula with hand-computed expected values for:
- All-singleton case
- All-matched case
- Mixed case
- Edge cases: empty predicted, empty true, both empty
"""

import pytest
from business_entity_resolution.evaluation.metrics import (
    EvaluationResult,
    build_ground_truth_dict,
    build_predictions_dict,
    evaluate,
    f05_single,
)
import pandas as pd


class TestF05Single:
    """Hand-computed F0.5 values for verification."""

    def test_perfect_match(self):
        """Both predicted and true are identical non-empty sets."""
        result = f05_single({"A", "B"}, {"A", "B"})
        assert result == 1.0

    def test_both_empty_singleton(self):
        """Correctly predicting a singleton (empty) → score 1.0."""
        result = f05_single(set(), set())
        assert result == 1.0

    def test_singleton_with_false_prediction(self):
        """Predicting a match for a true singleton → score 0.0."""
        result = f05_single({"A"}, set())
        assert result == 0.0

    def test_missed_all_matches(self):
        """Predicting empty for a non-singleton → score 0.0."""
        result = f05_single(set(), {"A", "B"})
        assert result == 0.0

    def test_partial_precision(self):
        """
        Predicted: {A, B, C}  True: {A, B}
        TP=2, FP=1, FN=0
        Precision = 2/3, Recall = 2/2 = 1.0
        F0.5 = (1.25 * 2/3 * 1.0) / (0.25 * 2/3 + 1.0)
             = (5/6) / (1/6 + 1)
             = (5/6) / (7/6)
             = 5/7 ≈ 0.7143
        """
        result = f05_single({"A", "B", "C"}, {"A", "B"})
        expected = (1.25 * (2.0 / 3.0) * 1.0) / (0.25 * (2.0 / 3.0) + 1.0)
        assert abs(result - expected) < 1e-9

    def test_partial_recall(self):
        """
        Predicted: {A}  True: {A, B}
        TP=1, FP=0, FN=1
        Precision = 1/1 = 1.0, Recall = 1/2 = 0.5
        F0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5)
             = 0.625 / 0.75
             = 5/6 ≈ 0.8333
        """
        result = f05_single({"A"}, {"A", "B"})
        expected = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5)
        assert abs(result - expected) < 1e-9

    def test_no_overlap(self):
        """Predicted {A}, True {B} → TP=0 → F0.5=0."""
        result = f05_single({"A"}, {"B"})
        assert result == 0.0

    def test_precision_weighted_higher_than_recall(self):
        """
        F0.5 weights precision about 2x over recall.
        A false positive (extra prediction) should hurt more than a false negative (missed).
        """
        # Case 1: extra prediction (FP)
        # Predicted {A, B, C, D, E}, True {A}
        score_fp = f05_single({"A", "B", "C", "D", "E"}, {"A"})
        # Case 2: extra miss (FN)
        # Predicted {A}, True {A, B, C, D, E}
        score_fn = f05_single({"A"}, {"A", "B", "C", "D", "E"})
        # F0.5 penalizes false positives more → score_fp < score_fn
        assert score_fp < score_fn


class TestMacroEvaluation:
    def test_all_singletons_correct(self):
        """All entities are singletons, correctly predicted as empty."""
        predictions = {"E1": set(), "E2": set(), "E3": set()}
        ground_truth = {"E1": set(), "E2": set(), "E3": set()}
        result = evaluate(predictions, ground_truth)
        assert result.macro_f05 == 1.0
        assert result.singleton_accuracy == 1.0
        assert result.n_singletons == 3
        assert result.n_non_singletons == 0

    def test_all_singletons_wrong(self):
        """All entities are singletons, but we predict matches for all."""
        predictions = {"E1": {"A"}, "E2": {"B"}, "E3": {"C"}}
        ground_truth = {"E1": set(), "E2": set(), "E3": set()}
        result = evaluate(predictions, ground_truth)
        assert result.macro_f05 == 0.0
        assert result.singleton_accuracy == 0.0

    def test_all_matched_perfect(self):
        """All entities have matches, all predicted correctly."""
        predictions = {"E1": {"A", "B"}, "E2": {"C"}}
        ground_truth = {"E1": {"A", "B"}, "E2": {"C"}}
        result = evaluate(predictions, ground_truth)
        assert result.macro_f05 == 1.0

    def test_mixed_case(self):
        """Mix of correct, partial, and singleton predictions."""
        predictions = {
            "E1": {"A"},        # True: {A} → F0.5 = 1.0
            "E2": set(),        # True: {B} → F0.5 = 0.0 (missed)
            "E3": set(),        # True: set() → F0.5 = 1.0 (correct singleton)
            "E4": {"X", "Y"},   # True: set() → F0.5 = 0.0 (false merge of singleton)
        }
        ground_truth = {
            "E1": {"A"},
            "E2": {"B"},
            "E3": set(),
            "E4": set(),
        }
        result = evaluate(predictions, ground_truth)
        # Macro average: (1.0 + 0.0 + 1.0 + 0.0) / 4 = 0.5
        assert abs(result.macro_f05 - 0.5) < 1e-9
        # Singletons: E3 correct, E4 wrong → accuracy = 1/2
        assert result.n_singletons == 2
        assert abs(result.singleton_accuracy - 0.5) < 1e-9

    def test_singleton_missing_from_gt(self):
        """Source-1 entities not in ground truth are treated as singletons."""
        predictions = {"E1": {"A"}, "E2": set()}
        ground_truth = {"E1": {"A"}}  # E2 not in GT → singleton
        result = evaluate(predictions, ground_truth)
        # E1: F0.5 = 1.0, E2: F0.5 = 1.0 (correctly predicted as singleton)
        assert result.macro_f05 == 1.0

    def test_n_entities(self):
        predictions = {"E1": set(), "E2": set(), "E3": {"A"}}
        ground_truth = {"E1": set(), "E2": set(), "E3": {"A"}}
        result = evaluate(predictions, ground_truth)
        assert result.n_entities == 3

    def test_per_entity_return(self):
        predictions = {"E1": {"A"}, "E2": set()}
        ground_truth = {"E1": {"A"}, "E2": set()}
        result = evaluate(predictions, ground_truth, return_per_entity=True)
        assert result.per_entity_f05 is not None
        assert "E1" in result.per_entity_f05
        assert "E2" in result.per_entity_f05
        assert result.per_entity_f05["E1"] == 1.0
        assert result.per_entity_f05["E2"] == 1.0


class TestGroundTruthHelpers:
    def test_build_ground_truth_dict(self):
        gt_df = pd.DataFrame({
            "source1_entity_id": ["E1", "E1", "E2"],
            "source_id": ["A", "B", "C"],
        })
        result = build_ground_truth_dict(gt_df, ["E1", "E2", "E3"])
        assert result["E1"] == {"A", "B"}
        assert result["E2"] == {"C"}
        assert result["E3"] == set()   # singleton

    def test_build_predictions_dict(self):
        mr_df = pd.DataFrame({
            "source1_entity_id": ["E1", "E2", "E3"],
            "matched_entity_ids": ["S2-A,S3-X", "S2-C", ""],
        })
        result = build_predictions_dict(mr_df)
        assert result["E1"] == {"S2-A", "S3-X"}
        assert result["E2"] == {"S2-C"}
        assert result["E3"] == set()

