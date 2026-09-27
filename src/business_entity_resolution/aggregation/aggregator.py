"""
aggregation/aggregator.py — Collapses scored candidate pairs into one row per
Source-1 entity, producing the final matching_results.tsv and candidate_pairs.tsv
that conform to the official ML Challenge 2026 submission format.

Official output schema
----------------------
matching_results.tsv:
    source1_entity_id : str
    matched_entity_ids : str   comma-separated S2-/S3- IDs combined, or empty string

candidate_pairs.tsv:
    source1_entity_id : str
    candidate_entity_ids : str   comma-separated S2-/S3- candidate IDs, or empty string

Key design decisions
--------------------
- Source-2 and Source-3 matches are COMBINED into a single ``matched_entity_ids``
  column (the official scorer column).  Source-tagging is preserved in an optional
  extra attribute available programmatically but not written to the TSV.
- Every Source-1 entity appears EXACTLY ONCE in both output files.
- Singletons (no surviving candidates above threshold) have empty second columns.
- ``candidate_pairs.tsv`` lists all above-threshold candidates, not just the final
  matches.  The validator issues a soft warning if any match is absent from the
  candidate set — this is expected behaviour.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class Aggregator:
    """
    Converts scored candidate pairs to one-row-per-Source-1-entity output
    conforming to the official ML Challenge 2026 validator requirements.

    Usage
    -----
    .. code-block:: python

        aggregator = Aggregator(threshold=0.5)
        results_df, candidates_df = aggregator.aggregate(
            scored_pairs_s2=scored_s2,   # DataFrame: source1_entity_id, other_entity_id, score
            scored_pairs_s3=scored_s3,
            all_source1_ids=source1_df["entity_id"].tolist(),
        )
    """

    def __init__(
        self,
        threshold: float = 0.5,
        per_country_thresholds: dict[str, float] | None = None,
        source1_country_map: dict[str, str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        threshold:
            Global match probability threshold.
        per_country_thresholds:
            Optional dict mapping country_norm -> threshold.  Takes priority over
            *threshold* when a Source-1 entity's country is found in this dict.
            "_default" key is used as fallback for unknown countries.
        source1_country_map:
            Optional dict mapping source1_entity_id -> country_norm.  Required when
            *per_country_thresholds* is provided.
        """
        self._threshold = threshold
        self._per_country = per_country_thresholds or {}
        self._country_map = source1_country_map or {}

    def _get_threshold(self, entity_id: str) -> float:
        """Return the applicable threshold for *entity_id*."""
        if not self._per_country:
            return self._threshold
        country = self._country_map.get(entity_id, "")
        if country in self._per_country:
            return self._per_country[country]
        return self._per_country.get("_default", self._threshold)

    def _apply_threshold_to_pairs(
        self, scored_pairs: pd.DataFrame
    ) -> dict[str, list[str]]:
        """
        Apply per-entity thresholds, returning source1_id -> list of matched IDs.
        """
        matched: dict[str, list[str]] = {}
        if scored_pairs.empty:
            return matched
        for _, row in scored_pairs.iterrows():
            s1id = row["source1_entity_id"]
            score = float(row["score"])
            thresh = self._get_threshold(s1id)
            if score >= thresh:
                matched.setdefault(s1id, []).append(row["other_entity_id"])
        return matched

    def aggregate(
        self,
        all_source1_ids: list[str],
        scored_pairs_s2: pd.DataFrame | None = None,
        scored_pairs_s3: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregate scored pairs into the official output format.

        Parameters
        ----------
        all_source1_ids:
            Complete list of all Source-1 entity ids (from source1_df).
            Every entity will appear in the output regardless of whether it
            has any matching candidates.
        scored_pairs_s2:
            Scored pairs for Source-2.
            Columns: source1_entity_id, other_entity_id, score.
        scored_pairs_s3:
            Scored pairs for Source-3.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (matching_results_df, candidate_pairs_df)

            matching_results_df columns: source1_entity_id, matched_entity_ids
            candidate_pairs_df columns:  source1_entity_id, candidate_entity_ids

            Both DataFrames have one row per Source-1 entity.
        """
        empty_df = pd.DataFrame(
            columns=["source1_entity_id", "other_entity_id", "score"]
        )
        s2_pairs = empty_df if scored_pairs_s2 is None else scored_pairs_s2
        s3_pairs = empty_df if scored_pairs_s3 is None else scored_pairs_s3

        # Apply threshold to get matches
        s2_matched = self._apply_threshold_to_pairs(s2_pairs)
        s3_matched = self._apply_threshold_to_pairs(s3_pairs)

        # Build candidate set (all pairs that were scored, regardless of threshold)
        s2_all = self._all_candidates(s2_pairs)
        s3_all = self._all_candidates(s3_pairs)

        logger.info(
            "Aggregation: %d s1 entities with S2 matches, %d with S3 matches",
            len(s2_matched), len(s3_matched),
        )

        matching_rows = []
        candidate_rows = []

        for s1id in all_source1_ids:
            # Combine S2 and S3 matches into a single comma-separated list
            combined_matches = sorted(
                set(s2_matched.get(s1id, [])) | set(s3_matched.get(s1id, []))
            )
            combined_candidates = sorted(
                set(s2_all.get(s1id, [])) | set(s3_all.get(s1id, []))
            )

            matching_rows.append({
                "source1_entity_id": s1id,
                "matched_entity_ids": ",".join(combined_matches),
            })
            candidate_rows.append({
                "source1_entity_id": s1id,
                "candidate_entity_ids": ",".join(combined_candidates),
            })

        matching_df = pd.DataFrame(matching_rows)
        candidates_df = pd.DataFrame(candidate_rows)

        # Sanity checks
        assert len(matching_df) == len(all_source1_ids), (
            f"Aggregator bug: expected {len(all_source1_ids)} rows, "
            f"got {len(matching_df)}"
        )
        assert not matching_df["source1_entity_id"].duplicated().any(), (
            "Aggregator bug: duplicate source1_entity_ids in matching_results"
        )

        return matching_df, candidates_df

    def _all_candidates(
        self, scored_pairs: pd.DataFrame
    ) -> dict[str, list[str]]:
        """Return all candidates (regardless of threshold) per source1 entity."""
        result: dict[str, list[str]] = {}
        if scored_pairs.empty or "other_entity_id" not in scored_pairs.columns:
            return result
        for _, row in scored_pairs.iterrows():
            result.setdefault(row["source1_entity_id"], []).append(
                row["other_entity_id"]
            )
        return result
