"""
blocking/strategies/rare_numeric_token.py — Blocking on rare numeric tokens.

Key construction:
    Each numeric token in the normalized name or address that appears in
    <= max_frequency Source-1 records is used as a blocking key.

Rationale: numeric tokens like building/plot/zone numbers are highly
discriminating when rare. Common numbers (e.g., "100", "1") are filtered out
as they would create enormous, uninformative blocks.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from business_entity_resolution.blocking.base import BlockingStrategy

_DIGIT_TOKEN_RE = re.compile(r"\b\d+\b")


def _extract_numeric_tokens(text: str) -> list[str]:
    """Extract all standalone numeric tokens from a string."""
    return _DIGIT_TOKEN_RE.findall(text or "")


class RareNumericTokenBlocking(BlockingStrategy):
    """
    Block on numeric tokens that appear in few Source-1 records.

    Tokens that appear in more than *max_frequency* Source-1 records are
    skipped (they are common and create useless mega-blocks).

    Each Source-1 record may contribute multiple blocking keys (one per
    unique rare numeric token in its name+address).
    """

    def __init__(self, max_frequency: int = 50, min_key_length: int = 2) -> None:
        self._max_frequency = max_frequency
        self._min_key_length = min_key_length

    @property
    def name(self) -> str:
        return "rare_numeric_token"

    def generate_candidates(
        self,
        source1_df: pd.DataFrame,
        other_df: pd.DataFrame,
    ) -> pd.DataFrame:
        min_len = self._min_key_length

        # ── Source-1: vectorized token extraction ──────────────────────────
        # Concatenate norm_name and norm_address into a single text column,
        # then explode all numeric tokens into one row each — no iterrows.
        s1_text = (
            source1_df["norm_name"].fillna("") + " " + source1_df["norm_address"].fillna("")
        )
        s1_tokens_ser = s1_text.apply(
            lambda t: list({tok for tok in _extract_numeric_tokens(t) if len(tok) >= min_len})
        )

        # Build token → entity_id mapping and frequency counter
        s1_exploded = (
            source1_df[["entity_id"]]
            .assign(token=s1_tokens_ser)
            .explode("token")
            .dropna(subset=["token"])
        )
        s1_exploded = s1_exploded[s1_exploded["token"] != ""]

        if s1_exploded.empty:
            return self._empty_candidates()

        token_freq: Counter = Counter(s1_exploded["token"])

        # Keep only rare tokens
        rare_tokens = {t for t, freq in token_freq.items() if freq <= self._max_frequency}
        if not rare_tokens:
            return self._empty_candidates()

        s1_rare = s1_exploded[s1_exploded["token"].isin(rare_tokens)]

        # ── Other source: vectorized token extraction ──────────────────────
        ot_text = (
            other_df["norm_name"].fillna("") + " " + other_df["norm_address"].fillna("")
        )
        ot_tokens_ser = ot_text.apply(
            lambda t: list({tok for tok in _extract_numeric_tokens(t) if len(tok) >= min_len and tok in rare_tokens})
        )
        ot_exploded = (
            other_df[["entity_id"]]
            .assign(token=ot_tokens_ser)
            .explode("token")
            .dropna(subset=["token"])
        )
        ot_exploded = ot_exploded[ot_exploded["token"] != ""]

        if ot_exploded.empty:
            return self._empty_candidates()

        # ── Generate pairs via merge on token ─────────────────────────────
        # A pandas merge replaces the nested Python loop entirely.
        merged = s1_rare.merge(
            ot_exploded.rename(columns={"entity_id": "other_entity_id"}),
            on="token",
        )[["entity_id", "other_entity_id"]].drop_duplicates()

        pairs = list(zip(merged["entity_id"], merged["other_entity_id"]))
        return self._make_candidates(pairs, self.name)
