"""
tests/unit/test_aggregation.py — Unit tests for the Aggregator.

Tests the official ML Challenge 2026 output format:
  - matching_results: source1_entity_id, matched_entity_ids (combined S2+S3)
  - candidate_pairs: source1_entity_id, candidate_entity_ids
"""

import pytest
import pandas as pd

from business_entity_resolution.aggregation.aggregator import Aggregator


def _make_scored(rows: list[tuple]) -> pd.DataFrame:
    """Helper to create scored pairs DataFrame."""
    return pd.DataFrame(rows, columns=["source1_entity_id", "other_entity_id", "score"])


class TestAggregator:
    def test_one_row_per_entity(self):
        """Every source1 entity appears exactly once in both output DataFrames."""
        agg = Aggregator(threshold=0.5)
        scored_s2 = _make_scored([("S1-001", "S2-001", 0.9)])
        matching, candidates = agg.aggregate(
            all_source1_ids=["S1-001", "S1-002", "S1-003"],
            scored_pairs_s2=scored_s2,
        )
        assert len(matching) == 3
        assert len(candidates) == 3
        assert set(matching["source1_entity_id"]) == {"S1-001", "S1-002", "S1-003"}
        assert not matching["source1_entity_id"].duplicated().any()

    def test_official_column_names(self):
        """Output DataFrames must have the official column names."""
        agg = Aggregator(threshold=0.5)
        matching, candidates = agg.aggregate(all_source1_ids=["S1-001"])
        assert list(matching.columns) == ["source1_entity_id", "matched_entity_ids"]
        assert list(candidates.columns) == ["source1_entity_id", "candidate_entity_ids"]

    def test_singleton_has_empty_matched_ids(self):
        """S1-002 with no candidates -> empty matched_entity_ids."""
        agg = Aggregator(threshold=0.5)
        scored_s2 = _make_scored([("S1-001", "S2-001", 0.9)])
        matching, _ = agg.aggregate(
            all_source1_ids=["S1-001", "S1-002"],
            scored_pairs_s2=scored_s2,
        )
        s2_row = matching[matching["source1_entity_id"] == "S1-002"].iloc[0]
        assert s2_row["matched_entity_ids"] == ""

    def test_below_threshold_excluded(self):
        """Pairs with score < threshold must not appear in matched_entity_ids."""
        agg = Aggregator(threshold=0.5)
        scored_s2 = _make_scored([
            ("S1-001", "S2-001", 0.9),   # above
            ("S1-001", "S2-002", 0.3),   # below
        ])
        matching, _ = agg.aggregate(all_source1_ids=["S1-001"], scored_pairs_s2=scored_s2)
        row = matching[matching["source1_entity_id"] == "S1-001"].iloc[0]
        assert "S2-001" in row["matched_entity_ids"]
        assert "S2-002" not in row["matched_entity_ids"]

    def test_s2_and_s3_combined_into_single_column(self):
        """S2 and S3 matches must be merged into matched_entity_ids, not separate cols."""
        agg = Aggregator(threshold=0.5)
        scored_s2 = _make_scored([("S1-001", "S2-001", 0.8)])
        scored_s3 = _make_scored([("S1-001", "S3-001", 0.9)])
        matching, _ = agg.aggregate(
            all_source1_ids=["S1-001"],
            scored_pairs_s2=scored_s2,
            scored_pairs_s3=scored_s3,
        )
        row = matching.iloc[0]
        ids = set(row["matched_entity_ids"].split(","))
        assert "S2-001" in ids
        assert "S3-001" in ids
        # Must be a single column, not two
        assert "source2_match_ids" not in matching.columns
        assert "source3_match_ids" not in matching.columns

    def test_multiple_matches_comma_separated(self):
        """Multiple matches produce a comma-separated string."""
        agg = Aggregator(threshold=0.5)
        scored_s2 = _make_scored([
            ("S1-001", "S2-001", 0.95),
            ("S1-001", "S2-002", 0.75),
        ])
        scored_s3 = _make_scored([("S1-001", "S3-001", 0.85)])
        matching, _ = agg.aggregate(
            all_source1_ids=["S1-001"],
            scored_pairs_s2=scored_s2,
            scored_pairs_s3=scored_s3,
        )
        ids = set(matching.iloc[0]["matched_entity_ids"].split(","))
        assert ids == {"S2-001", "S2-002", "S3-001"}

    def test_no_candidates_at_all(self):
        """No scored pairs -> all entities predicted as singletons."""
        agg = Aggregator(threshold=0.5)
        matching, candidates = agg.aggregate(all_source1_ids=["S1-001", "S1-002"])
        assert len(matching) == 2
        assert (matching["matched_entity_ids"] == "").all()
        assert (candidates["candidate_entity_ids"] == "").all()

    def test_candidate_pairs_includes_below_threshold(self):
        """candidate_entity_ids must include ALL scored candidates, not just matches."""
        agg = Aggregator(threshold=0.8)
        scored_s2 = _make_scored([
            ("S1-001", "S2-001", 0.9),  # above threshold -> match AND candidate
            ("S1-001", "S2-002", 0.3),  # below threshold -> candidate only, not match
        ])
        matching, candidates = agg.aggregate(
            all_source1_ids=["S1-001"], scored_pairs_s2=scored_s2
        )
        # Matching: only S2-001 (above threshold)
        match_ids = set(matching.iloc[0]["matched_entity_ids"].split(","))
        assert match_ids == {"S2-001"}

        # Candidates: both S2-001 and S2-002
        cand_ids = set(candidates.iloc[0]["candidate_entity_ids"].split(","))
        assert "S2-001" in cand_ids
        assert "S2-002" in cand_ids

    def test_per_country_threshold(self):
        """Per-country thresholds apply correctly."""
        country_map = {"S1-001": "US", "S1-002": "IN"}
        per_country = {"US": 0.8, "IN": 0.3, "_default": 0.5}
        agg = Aggregator(
            threshold=0.5,
            per_country_thresholds=per_country,
            source1_country_map=country_map,
        )
        # S1-001 (US, thresh=0.8): score 0.7 -> no match
        # S1-002 (IN, thresh=0.3): score 0.4 -> match
        scored = _make_scored([("S1-001", "S2-001", 0.7), ("S1-002", "S2-002", 0.4)])
        matching, _ = agg.aggregate(
            all_source1_ids=["S1-001", "S1-002"], scored_pairs_s2=scored
        )
        row1 = matching[matching["source1_entity_id"] == "S1-001"].iloc[0]
        row2 = matching[matching["source1_entity_id"] == "S1-002"].iloc[0]
        assert row1["matched_entity_ids"] == ""  # 0.7 < 0.8 -> no match
        assert "S2-002" in row2["matched_entity_ids"]  # 0.4 >= 0.3 -> match

    def test_no_duplicate_ids_in_matched(self):
        """The same ID from both S2 and S3 appears only once in matched_entity_ids."""
        agg = Aggregator(threshold=0.5)
        # Suppose some ID appears in both sources (edge case)
        scored_s2 = _make_scored([("S1-001", "S2-001", 0.9)])
        scored_s3 = _make_scored([("S1-001", "S2-001", 0.85)])  # same ID in s3
        matching, _ = agg.aggregate(
            all_source1_ids=["S1-001"],
            scored_pairs_s2=scored_s2,
            scored_pairs_s3=scored_s3,
        )
        ids = matching.iloc[0]["matched_entity_ids"].split(",")
        assert len(ids) == len(set(ids)), "Duplicate IDs found in matched_entity_ids"
