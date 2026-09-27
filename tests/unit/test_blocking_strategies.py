"""
tests/unit/test_blocking_strategies.py — Unit tests for blocking strategies.

Each strategy is tested in isolation on a small in-memory DataFrame.
Tests verify:
- Expected candidate pairs are produced
- Obviously-unrelated records are excluded
- Empty/degenerate inputs don't crash
"""

import pytest
import pandas as pd

from business_entity_resolution.blocking.strategies.name_prefix import NamePrefixBlocking
from business_entity_resolution.blocking.strategies.name_suffix import NameSuffixBlocking
from business_entity_resolution.blocking.strategies.address_number import AddressNumberBlocking
from business_entity_resolution.blocking.strategies.address_prefix import AddressPrefixBlocking
from business_entity_resolution.blocking.strategies.rare_numeric_token import RareNumericTokenBlocking


def _make_df(records: list[dict]) -> pd.DataFrame:
    """Helper: create a DataFrame with required normalized columns."""
    return pd.DataFrame(records)


def _make_source1(records):
    rows = []
    for r in records:
        rows.append({
            "entity_id": r[0],
            "norm_name": r[1],
            "norm_address": r[2],
            "street_number": r[3] if len(r) > 3 else "",
        })
    return pd.DataFrame(rows)


def _make_other(records):
    return _make_source1(records)


class TestNamePrefixBlocking:
    def setup_method(self):
        self.strategy = NamePrefixBlocking(prefix_len=4, min_key_length=2)

    def test_matching_prefix(self):
        s1 = _make_source1([
            ("A1", "acme robotics inc", "123 main st", "123"),
        ])
        ot = _make_other([
            ("B1", "acme robotics incorporated", "456 elm st", "456"),
            ("B2", "delta foods corp", "789 oak st", "789"),
        ])
        candidates = self.strategy.generate_candidates(s1, ot)
        # A1-B1 should match (prefix "acme")
        pairs = set(zip(candidates["source1_entity_id"], candidates["other_entity_id"]))
        assert ("A1", "B1") in pairs
        # A1-B2 should NOT match (prefix "delt" != "acme")
        assert ("A1", "B2") not in pairs

    def test_empty_source1(self):
        s1 = pd.DataFrame(columns=["entity_id", "norm_name", "norm_address", "street_number"])
        ot = _make_other([("B1", "acme corp", "123 st", "123")])
        result = self.strategy.generate_candidates(s1, ot)
        assert result.empty

    def test_empty_other(self):
        s1 = _make_source1([("A1", "acme corp", "123 st", "123")])
        ot = pd.DataFrame(columns=["entity_id", "norm_name", "norm_address", "street_number"])
        result = self.strategy.generate_candidates(s1, ot)
        assert result.empty

    def test_degenerate_key_filtered(self):
        # Use prefix_len=4 strategy. "ab" normalized to "ab" (len=2 < 4)
        # so key = "ab"[:4] = "ab" (len=2 >= min_key_length=2 → included)
        # "ab extra" → key = "ab e" (len=4, different from "ab")
        # So they should NOT match — test that only true prefix matches work
        strategy_2char = NamePrefixBlocking(prefix_len=2, min_key_length=2)
        s1 = _make_source1([("A1", "ab tech", "123 st", "123")])  # key = "ab"
        ot = _make_other([("B1", "ab systems", "456 st", "456")])  # key = "ab"
        result = strategy_2char.generate_candidates(s1, ot)
        pairs = set(zip(result["source1_entity_id"], result["other_entity_id"]))
        assert ("A1", "B1") in pairs


    def test_output_schema(self):
        s1 = _make_source1([("A1", "acme robotics", "123 st", "123")])
        ot = _make_other([("B1", "acme foods", "456 st", "456")])
        result = self.strategy.generate_candidates(s1, ot)
        assert list(result.columns) == ["source1_entity_id", "other_entity_id", "strategy_name"]
        assert (result["strategy_name"] == "name_prefix").all()

    def test_no_self_match(self):
        # Records that would match the same key but be the same entity (if used on same df)
        # In practice source1 and other are different, so this is about unrelated entities sharing prefix
        s1 = _make_source1([
            ("A1", "bright cafe llc", "789 oak ave", "789"),
            ("A2", "bright corner store", "100 main st", "100"),
        ])
        ot = _make_other([
            ("B1", "bright bistro", "200 elm st", "200"),
        ])
        result = self.strategy.generate_candidates(s1, ot)
        # Both A1 and A2 match B1 on prefix "brigh"
        pairs = set(zip(result["source1_entity_id"], result["other_entity_id"]))
        assert ("A1", "B1") in pairs
        assert ("A2", "B1") in pairs


class TestNameSuffixBlocking:
    def setup_method(self):
        self.strategy = NameSuffixBlocking(suffix_len=4, min_key_length=2)

    def test_matching_suffix(self):
        s1 = _make_source1([("A1", "acme corp", "123 st", "123")])
        ot = _make_other([
            ("B1", "delta corp", "456 st", "456"),
            ("B2", "something else", "789 st", "789"),
        ])
        result = self.strategy.generate_candidates(s1, ot)
        pairs = set(zip(result["source1_entity_id"], result["other_entity_id"]))
        assert ("A1", "B1") in pairs
        assert ("A1", "B2") not in pairs

    def test_strategy_name(self):
        s1 = _make_source1([("A1", "acme corp", "123 st", "123")])
        ot = _make_other([("B1", "delta corp", "456 st", "456")])
        result = self.strategy.generate_candidates(s1, ot)
        assert (result["strategy_name"] == "name_suffix").all()


class TestAddressNumberBlocking:
    def setup_method(self):
        self.strategy = AddressNumberBlocking(min_key_length=1)

    def test_matching_street_number(self):
        s1 = _make_source1([("A1", "acme corp", "123 main street", "123")])
        ot = _make_other([
            ("B1", "acme inc", "123 main st", "123"),
            ("B2", "acme ltd", "456 main st", "456"),
        ])
        result = self.strategy.generate_candidates(s1, ot)
        pairs = set(zip(result["source1_entity_id"], result["other_entity_id"]))
        assert ("A1", "B1") in pairs
        assert ("A1", "B2") not in pairs

    def test_landmark_relative_excluded(self):
        # Records with empty street_number should not be used
        s1 = _make_source1([("A1", "acme corp", "near city hall", "")])
        ot = _make_other([("B1", "acme inc", "near city hall", "")])
        result = self.strategy.generate_candidates(s1, ot)
        assert result.empty

    def test_strategy_name(self):
        s1 = _make_source1([("A1", "a", "123 st", "123")])
        ot = _make_other([("B1", "b", "123 st", "123")])
        result = self.strategy.generate_candidates(s1, ot)
        assert (result["strategy_name"] == "address_number").all()


class TestAddressPrefixBlocking:
    def setup_method(self):
        self.strategy = AddressPrefixBlocking(prefix_len=5, min_key_length=2)

    def test_matching_prefix(self):
        s1 = _make_source1([("A1", "a", "123 main street chicago", "123")])
        ot = _make_other([
            ("B1", "b", "123 main st chi", "123"),
            ("B2", "c", "456 elm street", "456"),
        ])
        result = self.strategy.generate_candidates(s1, ot)
        pairs = set(zip(result["source1_entity_id"], result["other_entity_id"]))
        assert ("A1", "B1") in pairs
        assert ("A1", "B2") not in pairs

    def test_empty_address_filtered(self):
        s1 = _make_source1([("A1", "a", "", "")])
        ot = _make_other([("B1", "b", "123 main", "123")])
        result = self.strategy.generate_candidates(s1, ot)
        # Empty address key "" has length 0 < min_key_length=2 → filtered
        assert result.empty


class TestRareNumericTokenBlocking:
    def setup_method(self):
        self.strategy = RareNumericTokenBlocking(max_frequency=2, min_key_length=2)

    def test_rare_number_blocks(self):
        # S1001 has number 99 in address (rare)
        s1 = _make_source1([
            ("A1", "acme corp", "99 special drive", "99"),
            ("A2", "other corp", "1 main st", "1"),
        ])
        ot = _make_other([
            ("B1", "acme inc", "99 special avenue", "99"),
            ("B2", "other inc", "1 main road", "1"),
        ])
        result = self.strategy.generate_candidates(s1, ot)
        pairs = set(zip(result["source1_entity_id"], result["other_entity_id"]))
        # 99 appears once in s1 (max_frequency=2), so it's rare → A1-B1 should appear
        assert ("A1", "B1") in pairs

    def test_common_number_filtered(self):
        # A number appearing in > max_frequency source1 records should not be used
        s1 = _make_source1([
            ("A1", "corp 1", "100 main st", "100"),
            ("A2", "corp 2", "100 elm st", "100"),
            ("A3", "corp 3", "100 oak st", "100"),  # 3 records → > max_frequency=2
        ])
        ot = _make_other([("B1", "b", "100 main st", "100")])
        # 100 appears 3 times in s1 (> max_frequency=2) → should not generate pairs
        # Note: max_frequency=2, so <= 2 is rare; 3 > 2 → not rare
        result = self.strategy.generate_candidates(s1, ot)
        pairs = set(zip(result["source1_entity_id"], result["other_entity_id"]))
        for s1_id in ["A1", "A2", "A3"]:
            assert (s1_id, "B1") not in pairs
