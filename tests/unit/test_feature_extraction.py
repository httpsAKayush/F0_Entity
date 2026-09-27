"""
tests/unit/test_feature_extraction.py — Unit tests for pairwise feature functions.

Tests known input → known expected output for each feature function.
"""

import numpy as np
import pandas as pd
import pytest

from business_entity_resolution.features.pairwise import (
    FEATURE_NAMES,
    FeatureExtractor,
    common_prefix_ratio,
    exact_match,
    len_ratio,
    street_number_agreement,
    token_jaccard,
)


class TestExactMatch:
    def test_identical(self):
        assert exact_match("acme corp", "acme corp") == 1.0

    def test_different(self):
        assert exact_match("acme corp", "delta inc") == 0.0

    def test_whitespace_stripped(self):
        assert exact_match("acme corp ", " acme corp") == 1.0

    def test_empty_both(self):
        assert exact_match("", "") == 1.0

    def test_empty_one(self):
        assert exact_match("acme", "") == 0.0


class TestTokenJaccard:
    def test_identical(self):
        assert token_jaccard("acme robotics", "acme robotics") == 1.0

    def test_no_overlap(self):
        assert token_jaccard("acme robotics", "delta foods") == 0.0

    def test_partial_overlap(self):
        # {acme, corp} ∩ {acme, foods} = {acme}; union = {acme, corp, foods} → J = 1/3
        result = token_jaccard("acme corp", "acme foods")
        assert abs(result - 1.0 / 3.0) < 1e-9

    def test_both_empty(self):
        assert token_jaccard("", "") == 1.0

    def test_one_empty(self):
        assert token_jaccard("acme", "") == 0.0

    def test_superset(self):
        # {acme, robotics, inc} ∩ {acme, robotics} = {acme, robotics}; union = 3
        result = token_jaccard("acme robotics inc", "acme robotics")
        assert abs(result - 2.0 / 3.0) < 1e-9


class TestCommonPrefixRatio:
    def test_identical(self):
        assert common_prefix_ratio("hello", "hello") == 1.0

    def test_no_common_prefix(self):
        result = common_prefix_ratio("abc", "xyz")
        assert result == 0.0

    def test_partial(self):
        # "acme" and "acorn" share "ac" (2 chars), max length = 5 → 2/5
        result = common_prefix_ratio("acme", "acorn")
        assert abs(result - 2.0 / 5.0) < 1e-9

    def test_both_empty(self):
        assert common_prefix_ratio("", "") == 1.0


class TestLenRatio:
    def test_equal_lengths(self):
        assert len_ratio("hello", "world") == 1.0

    def test_different_lengths(self):
        # min=3, max=6 → 3/6 = 0.5
        result = len_ratio("abc", "abcdef")
        assert abs(result - 0.5) < 1e-9

    def test_one_empty(self):
        assert len_ratio("abc", "") == 0.0

    def test_both_empty(self):
        assert len_ratio("", "") == 1.0


class TestStreetNumberAgreement:
    def test_both_match(self):
        assert street_number_agreement("123", "123") == 1.0

    def test_both_different(self):
        assert street_number_agreement("123", "456") == 0.0

    def test_one_empty(self):
        # One landmark-relative → inconclusive → 0.5
        assert street_number_agreement("123", "") == 0.5
        assert street_number_agreement("", "123") == 0.5

    def test_both_empty(self):
        # Both landmark-relative → inconclusive → 0.5
        assert street_number_agreement("", "") == 0.5


class TestFeatureExtractor:
    def _make_source_df(self, records):
        return pd.DataFrame(records, columns=[
            "entity_id", "norm_name", "norm_address", "street_number",
            "is_landmark", "country_norm"
        ])

    def _make_candidates(self, pairs):
        return pd.DataFrame(
            [(s1, ot, "test") for s1, ot in pairs],
            columns=["source1_entity_id", "other_entity_id", "strategy_name"]
        )

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == 20

    def test_exact_match_feature(self):
        s1 = self._make_source_df([
            ("A1", "acme corp", "123 main st", "123", 0, "US"),
        ])
        ot = self._make_source_df([
            ("B1", "acme corp", "123 main st", "123", 0, "US"),
            ("B2", "delta inc", "456 elm st", "456", 0, "US"),
        ])
        cands = self._make_candidates([("A1", "B1"), ("A1", "B2")])
        extractor = FeatureExtractor()
        X = extractor.extract(cands, s1, ot)

        # name_exact_match is feature 0
        assert X[0, 0] == 1.0   # A1-B1: names match
        assert X[1, 0] == 0.0   # A1-B2: names don't match

    def test_street_number_agreement_feature(self):
        s1 = self._make_source_df([("A1", "a", "123 main st", "123", 0, "US")])
        ot = self._make_source_df([
            ("B1", "b", "123 elm st", "123", 0, "US"),
            ("B2", "c", "456 oak st", "456", 0, "US"),
            ("B3", "d", "near city hall", "", 1, "US"),
        ])
        cands = self._make_candidates([("A1", "B1"), ("A1", "B2"), ("A1", "B3")])
        extractor = FeatureExtractor()
        X = extractor.extract(cands, s1, ot)

        # street_number_match is feature 12
        assert X[0, 12] == 1.0   # 123 == 123
        assert X[1, 12] == 0.0   # 123 != 456
        assert X[2, 12] == 0.5   # 123 vs "" → inconclusive

    def test_output_shape(self):
        s1 = self._make_source_df([("A1", "acme", "123 st", "123", 0, "US")])
        ot = self._make_source_df([("B1", "acme", "123 st", "123", 0, "US")])
        cands = self._make_candidates([("A1", "B1")])
        extractor = FeatureExtractor()
        X = extractor.extract(cands, s1, ot)
        assert X.shape == (1, len(FEATURE_NAMES))

    def test_empty_candidates(self):
        s1 = self._make_source_df([("A1", "acme", "123 st", "123", 0, "US")])
        ot = self._make_source_df([("B1", "acme", "123 st", "123", 0, "US")])
        cands = pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "strategy_name"])
        extractor = FeatureExtractor()
        X = extractor.extract(cands, s1, ot)
        assert X.shape == (0, len(FEATURE_NAMES))

    def test_country_agreement_feature(self):
        s1 = self._make_source_df([("A1", "a", "123 st", "123", 0, "US")])
        ot = self._make_source_df([
            ("B1", "b", "123 st", "123", 0, "US"),
            ("B2", "c", "456 st", "456", 0, "IN"),
        ])
        cands = self._make_candidates([("A1", "B1"), ("A1", "B2")])
        extractor = FeatureExtractor()
        X = extractor.extract(cands, s1, ot)
        # country_agree is feature 16
        assert X[0, 16] == 1.0   # US == US
        assert X[1, 16] == 0.0   # US != IN

    def test_feature_names_are_canonical(self):
        # FEATURE_NAMES must not have duplicates
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
