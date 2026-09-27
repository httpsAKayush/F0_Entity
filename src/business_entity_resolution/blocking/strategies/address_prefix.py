"""
blocking/strategies/address_prefix.py — Blocking on normalized-address prefix.

Key construction:
    norm_address[:prefix_len]

Catches matches based on the start of the address string. Works well for
structured addresses that start with a street number or consistent prefix.
"""

from __future__ import annotations

import pandas as pd

from business_entity_resolution.blocking.base import BlockingStrategy


class AddressPrefixBlocking(BlockingStrategy):
    """Block candidate pairs on the first *prefix_len* characters of normalized address."""

    def __init__(self, prefix_len: int = 6, min_key_length: int = 2) -> None:
        self._prefix_len = prefix_len
        self._min_key_length = min_key_length

    @property
    def name(self) -> str:
        return "address_prefix"

    def generate_candidates(
        self,
        source1_df: pd.DataFrame,
        other_df: pd.DataFrame,
    ) -> pd.DataFrame:
        s1 = source1_df[["entity_id", "norm_address"]].copy()
        ot = other_df[["entity_id", "norm_address"]].copy()

        s1["key"] = s1["norm_address"].str[:self._prefix_len]
        ot["key"] = ot["norm_address"].str[:self._prefix_len]

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
