"""
blocking/strategies/name_suffix.py — Blocking on normalized-name suffix.

Key construction:
    norm_name[-suffix_len:]

Complements name_prefix by catching cases where names differ at the start
(e.g. "The Acme Corp" vs "Acme Corp") but agree at the end.
"""

from __future__ import annotations

import pandas as pd

from business_entity_resolution.blocking.base import BlockingStrategy


class NameSuffixBlocking(BlockingStrategy):
    """Block candidate pairs on the last *suffix_len* characters of normalized name."""

    def __init__(self, suffix_len: int = 5, min_key_length: int = 2) -> None:
        self._suffix_len = suffix_len
        self._min_key_length = min_key_length

    @property
    def name(self) -> str:
        return "name_suffix"

    def generate_candidates(
        self,
        source1_df: pd.DataFrame,
        other_df: pd.DataFrame,
    ) -> pd.DataFrame:
        s1 = source1_df[["entity_id", "norm_name"]].copy()
        ot = other_df[["entity_id", "norm_name"]].copy()

        s1["key"] = s1["norm_name"].str[-self._suffix_len:]
        ot["key"] = ot["norm_name"].str[-self._suffix_len:]

        s1 = s1[s1["key"].str.len() >= self._min_key_length]
        ot = ot[ot["key"].str.len() >= self._min_key_length]

        if s1.empty or ot.empty:
            return self._empty_candidates()

        merged = s1.merge(ot, on="key", suffixes=("_s1", "_ot"))
        merged = merged.rename(
            columns={"entity_id_s1": "source1_entity_id", "entity_id_ot": "other_entity_id"}
        )
        merged["strategy_name"] = self.name
        return merged[["source1_entity_id", "other_entity_id", "strategy_name"]]
