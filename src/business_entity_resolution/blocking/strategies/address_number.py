"""
blocking/strategies/address_number.py — Blocking on street number.

Key construction:
    street_number (the numeric token extracted by normalize_address)

Street numbers are highly discriminating when present — two businesses with
the same street number in the same country partition are very likely related.
Records without a street number (landmark-relative) are excluded from this
blocking channel (but may still appear via other channels).
"""

from __future__ import annotations

import pandas as pd

from business_entity_resolution.blocking.base import BlockingStrategy


class AddressNumberBlocking(BlockingStrategy):
    """
    Block candidate pairs on the parsed street number token.

    Landmark-relative addresses (where street_number is empty) are excluded
    from this channel — they will surface via address_prefix or tfidf_nn.
    """

    def __init__(self, min_token_len: int = 1, min_key_length: int = 1) -> None:
        self._min_token_len = min_token_len
        self._min_key_length = min_key_length

    @property
    def name(self) -> str:
        return "address_number"

    def generate_candidates(
        self,
        source1_df: pd.DataFrame,
        other_df: pd.DataFrame,
    ) -> pd.DataFrame:
        s1 = source1_df[["entity_id", "street_number"]].copy()
        ot = other_df[["entity_id", "street_number"]].copy()

        # Only keep records that have a non-empty street number
        s1 = s1[s1["street_number"].str.len() >= max(self._min_key_length, 1)]
        ot = ot[ot["street_number"].str.len() >= max(self._min_key_length, 1)]

        if s1.empty or ot.empty:
            return self._empty_candidates()

        s1 = s1.rename(columns={"street_number": "key"})
        ot = ot.rename(columns={"street_number": "key"})

        merged = s1.merge(ot, on="key", suffixes=("_s1", "_ot"))
        merged = merged.rename(
            columns={"entity_id_s1": "source1_entity_id", "entity_id_ot": "other_entity_id"}
        )
        merged["strategy_name"] = self.name
        return merged[["source1_entity_id", "other_entity_id", "strategy_name"]]
