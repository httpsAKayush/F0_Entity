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
        # Build frequency table from Source-1
        s1_tokens: dict[str, list[str]] = {}  # entity_id -> list[numeric tokens]
        token_freq: Counter[str] = Counter()

        for _, row in source1_df[["entity_id", "norm_name", "norm_address"]].iterrows():
            combined = f"{row['norm_name']} {row['norm_address']}"
            tokens = set(_extract_numeric_tokens(combined))
            tokens = {t for t in tokens if len(t) >= self._min_key_length}
            s1_tokens[row["entity_id"]] = list(tokens)
            token_freq.update(tokens)

        # Identify rare tokens
        rare_tokens = {t for t, freq in token_freq.items() if freq <= self._max_frequency}

        if not rare_tokens:
            return self._empty_candidates()

        # Index other_df by numeric tokens
        ot_index: dict[str, list[str]] = {}  # token -> list[entity_id]
        for _, row in other_df[["entity_id", "norm_name", "norm_address"]].iterrows():
            combined = f"{row['norm_name']} {row['norm_address']}"
            tokens = set(_extract_numeric_tokens(combined))
            tokens &= rare_tokens  # only use tokens rare in Source-1
            for tok in tokens:
                ot_index.setdefault(tok, []).append(row["entity_id"])

        # Generate pairs
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for s1_id, tokens in s1_tokens.items():
            for tok in tokens:
                if tok not in rare_tokens:
                    continue
                for ot_id in ot_index.get(tok, []):
                    pair = (s1_id, ot_id)
                    if pair not in seen:
                        seen.add(pair)
                        pairs.append(pair)

        return self._make_candidates(pairs, self.name)
