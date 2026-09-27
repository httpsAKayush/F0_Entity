"""
blocking/strategies/tfidf_nn.py — TF-IDF nearest-neighbor blocking.

Uses character n-gram TF-IDF vectors to find approximate nearest neighbors
between Source-1 and Source-2/3 records.  This is a soft blocking channel
that catches name variants and typos that exact-key strategies would miss.

Key construction:
    Combined string: norm_name + " " + norm_address
    Vectorized with character n-gram TF-IDF (configurable ngram_range, analyzer).

Candidate generation:
    For each Source-1 record, retrieve the top-K most similar Source-2/3 records
    by cosine similarity.  Pairs below min_similarity are discarded.

Memory note:
    The TF-IDF similarity matrix is computed in batches to avoid building a full
    N x M matrix in RAM.  For 2.2M x 5M records this is still large, so we
    partition by country upstream; within a country the matrix is manageable.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from business_entity_resolution.blocking.base import BlockingStrategy

logger = logging.getLogger(__name__)

# Batch size for similarity computation (number of Source-1 rows per batch)
_BATCH_SIZE = 500


class TFIDFNearestNeighborBlocking(BlockingStrategy):
    """
    TF-IDF character-n-gram nearest-neighbor blocking.

    For each Source-1 record, finds the top-K most similar Source-2/3 records
    by cosine similarity of their TF-IDF vectors.
    """

    def __init__(
        self,
        top_k: int = 20,
        min_similarity: float = 0.10,
        max_features: int = 200_000,
        ngram_range: tuple[int, int] = (1, 2),
        analyzer: str = "char_wb",
        min_key_length: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        top_k:
            Maximum number of candidate matches per Source-1 record.
        min_similarity:
            Cosine similarity threshold; pairs below this are discarded.
        max_features:
            TF-IDF vocabulary size cap (limits memory usage).
        ngram_range:
            Character n-gram range for TF-IDF.
        analyzer:
            'char_wb' (word-boundary-padded chars) or 'char'.
        min_key_length:
            Minimum length of the combined text; shorter entries are skipped.
        """
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._max_features = max_features
        self._ngram_range = ngram_range
        self._analyzer = analyzer
        self._min_key_length = min_key_length

    @property
    def name(self) -> str:
        return "tfidf_nn"

    def _build_text(self, df: pd.DataFrame) -> pd.Series:
        """Combine norm_name and norm_address into a single text field."""
        return df["norm_name"].fillna("") + " " + df["norm_address"].fillna("")

    def generate_candidates(
        self,
        source1_df: pd.DataFrame,
        other_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate candidates via TF-IDF cosine similarity nearest neighbors."""
        s1_texts = self._build_text(source1_df).tolist()
        ot_texts = self._build_text(other_df).tolist()
        s1_ids = source1_df["entity_id"].tolist()
        ot_ids = other_df["entity_id"].tolist()

        # Filter empty texts
        s1_mask = [len(t.strip()) >= self._min_key_length for t in s1_texts]
        ot_mask = [len(t.strip()) >= self._min_key_length for t in ot_texts]

        s1_texts_f = [t for t, m in zip(s1_texts, s1_mask) if m]
        s1_ids_f = [i for i, m in zip(s1_ids, s1_mask) if m]
        ot_texts_f = [t for t, m in zip(ot_texts, ot_mask) if m]
        ot_ids_f = [i for i, m in zip(ot_ids, ot_mask) if m]

        if not s1_texts_f or not ot_texts_f:
            return self._empty_candidates()

        logger.debug(
            "TF-IDF blocking: %d source1 x %d other records", len(s1_texts_f), len(ot_texts_f)
        )

        # Fit TF-IDF on the union of both sides (so the vocabulary covers all terms)
        vectorizer = TfidfVectorizer(
            analyzer=self._analyzer,
            ngram_range=self._ngram_range,
            max_features=self._max_features,
            sublinear_tf=True,
            min_df=1,
        )
        all_texts = s1_texts_f + ot_texts_f
        vectorizer.fit(all_texts)

        s1_matrix = vectorizer.transform(s1_texts_f)
        ot_matrix = vectorizer.transform(ot_texts_f)

        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        top_k = min(self._top_k, len(ot_ids_f))

        # Process Source-1 in batches to limit peak memory
        for batch_start in range(0, len(s1_ids_f), _BATCH_SIZE):
            batch_end = min(batch_start + _BATCH_SIZE, len(s1_ids_f))
            s1_batch = s1_matrix[batch_start:batch_end]
            s1_ids_batch = s1_ids_f[batch_start:batch_end]

            # Compute cosine similarity for this batch: shape (batch, |ot|)
            sim = cosine_similarity(s1_batch, ot_matrix)

            for row_idx, s1_id in enumerate(s1_ids_batch):
                row = sim[row_idx]
                # Get indices of top-K highest similarities
                if len(row) <= top_k:
                    top_indices = np.argsort(row)[::-1]
                else:
                    top_indices = np.argpartition(row, -top_k)[-top_k:]
                    top_indices = top_indices[np.argsort(row[top_indices])[::-1]]

                for idx in top_indices:
                    score = row[idx]
                    if score < self._min_similarity:
                        break
                    ot_id = ot_ids_f[idx]
                    pair = (s1_id, ot_id)
                    if pair not in seen:
                        seen.add(pair)
                        pairs.append(pair)

        return self._make_candidates(pairs, self.name)
