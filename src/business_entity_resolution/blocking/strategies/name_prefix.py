"""
blocking/strategies/name_prefix.py — Blocking on normalized-name prefix.

Key construction:
    norm_name[:prefix_len]

This is a simple, fast blocking channel that captures most name-based matches
when the first few characters of the business name agree.
"""

from __future__ import annotations

import pandas as pd

from business_entity_resolution.blocking.base import BlockingStrategy


class NamePrefixBlocking(BlockingStrategy):
    """
    Block candidate pairs on the first *prefix_len* characters of the
    normalized business name.

    Records whose normalized names share the same prefix are placed in the
    same block and cross-joined to produce candidate pairs.
    """

    def __init__(self, prefix_len: int = 5, min_key_length: int = 2) -> None:
        """
        Parameters
        ----------
        prefix_len:
            Number of characters to use from the start of the normalized name.
        min_key_length:
            Minimum key length required; shorter keys are skipped to avoid
            degenerate blocks.
        """
        self._prefix_len = prefix_len
        self._min_key_length = min_key_length

    @property
    def name(self) -> str:
        return "name_prefix"

    def generate_candidates(
        self,
        source1_df: pd.DataFrame,
        other_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate candidate pairs sharing the same name prefix."""
        s1 = source1_df[["entity_id", "norm_name"]].copy()
        ot = other_df[["entity_id", "norm_name"]].copy()

        s1["key"] = s1["norm_name"].str[:self._prefix_len]
        ot["key"] = ot["norm_name"].str[:self._prefix_len]

        # Filter degenerate (too-short) keys
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
