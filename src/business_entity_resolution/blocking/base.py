"""
blocking/base.py — Abstract BlockingStrategy interface.

Every concrete blocking strategy must subclass BlockingStrategy and implement
generate_candidates().  This interface is the contract that the BlockingEngine
relies on — callers never need to know which concrete strategy they are using.

Adding a new blocking channel:
    1. Create a new module in blocking/strategies/
    2. Subclass BlockingStrategy
    3. Implement generate_candidates()
    4. Register the new strategy name in BlockingEngine or pass it explicitly
    That's it — no other code changes required (open/closed principle).

Candidate DataFrame columns (contract, all strategies must produce these):
    source1_entity_id : str   — Source-1 entity id
    other_entity_id   : str   — Source-2 or Source-3 entity id
    strategy_name     : str   — Name of the strategy that generated this pair
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BlockingStrategy(ABC):
    """
    Abstract base class for all blocking strategies.

    A blocking strategy generates *candidate pairs* — (source1_id, other_id)
    tuples that are plausible matches and warrant pairwise feature extraction.

    Each strategy is independently applicable to any country partition and must
    produce a DataFrame with exactly three columns: source1_entity_id,
    other_entity_id, strategy_name.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, unique name for this strategy (used in diagnostics and logging)."""
        ...

    @abstractmethod
    def generate_candidates(
        self,
        source1_df: pd.DataFrame,
        other_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate candidate pairs from two source DataFrames.

        Both DataFrames have been pre-partitioned by country (the caller
        guarantees they contain only records from the same country), and
        have the following columns already added by the pipeline:
            norm_name    : str   — normalized business name
            norm_address : str   — normalized address (raw_normalized field)
            street_number: str   — street number or empty string
            country_norm : str   — normalized country code

        Parameters
        ----------
        source1_df:
            Source-1 records for one country partition.
            Must have column: entity_id, norm_name, norm_address, street_number
        other_df:
            Source-2 or Source-3 records for the same country partition.
            Must have column: entity_id, norm_name, norm_address, street_number

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
                source1_entity_id : str
                other_entity_id   : str
                strategy_name     : str
            May be empty if no candidates are found.
        """
        ...

    @staticmethod
    def _empty_candidates() -> pd.DataFrame:
        """Return an empty candidate DataFrame with the correct schema."""
        return pd.DataFrame(
            columns=["source1_entity_id", "other_entity_id", "strategy_name"]
        )

    @staticmethod
    def _make_candidates(
        pairs: list[tuple[str, str]],
        strategy_name: str,
    ) -> pd.DataFrame:
        """
        Convenience factory to create the result DataFrame from a list of pairs.

        Parameters
        ----------
        pairs:
            List of (source1_entity_id, other_entity_id) tuples.
        strategy_name:
            Name of the strategy (filled into the strategy_name column).

        Returns
        -------
        pd.DataFrame
        """
        if not pairs:
            return BlockingStrategy._empty_candidates()
        return pd.DataFrame(
            pairs,
            columns=["source1_entity_id", "other_entity_id"],
        ).assign(strategy_name=strategy_name)
