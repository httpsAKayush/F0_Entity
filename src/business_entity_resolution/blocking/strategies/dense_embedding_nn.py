"""
blocking/strategies/dense_embedding_nn.py — Dense embedding nearest-neighbor blocking.

Optional, config-toggleable blocking channel that uses a pretrained multilingual
sentence embedding model to find semantic nearest neighbors.

This channel is DISABLED by default (use_dense_embedding: false in config) because:
1. It requires downloading a pretrained model (~420MB) on first run.
2. It is the most compute-intensive channel (~10x slower than TF-IDF).
3. For many record types, TF-IDF char n-grams already achieve high recall.

Enable via: config.blocking.use_dense_embedding = True
Or:         export BER_USE_DENSE=1

Network note:
    The ``sentence-transformers`` library downloads model weights on first use.
    This is the ONLY place in the pipeline that may make a network call, and
    only on first run (subsequent runs use the local cache).  This is documented
    here and in README.md, and is flagged in the grep audit.
    # NETWORK_CALL_PERMITTED: model weight download via sentence-transformers
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from business_entity_resolution.blocking.base import BlockingStrategy

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_BATCH_SIZE_EMBED = 512


class DenseEmbeddingNNBlocking(BlockingStrategy):
    """
    Dense semantic embedding nearest-neighbor blocking.

    Uses a multilingual pretrained sentence embedding model to embed business
    name + address strings into a dense vector space, then finds approximate
    nearest neighbors using cosine similarity.

    Requires: pip install sentence-transformers
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 20,
        min_similarity: float = 0.60,
        batch_size: int = 512,
        min_key_length: int = 2,
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._batch_size = batch_size
        self._min_key_length = min_key_length
        self._model = None  # lazy init

    @property
    def name(self) -> str:
        return "dense_embedding_nn"

    def _get_model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                logger.info(
                    "Loading dense embedding model: %s "
                    "(first run downloads ~420MB — see dense_embedding_nn.py header)",
                    self._model_name,
                )
                self._model = SentenceTransformer(self._model_name)
            except ImportError as exc:
                raise ImportError(
                    "Dense embedding blocking requires 'sentence-transformers'. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize for cosine via dot product
        )
        return embeddings

    def generate_candidates(
        self,
        source1_df: pd.DataFrame,
        other_df: pd.DataFrame,
    ) -> pd.DataFrame:
        s1_texts = (
            source1_df["norm_name"].fillna("") + " " + source1_df["norm_address"].fillna("")
        ).tolist()
        ot_texts = (
            other_df["norm_name"].fillna("") + " " + other_df["norm_address"].fillna("")
        ).tolist()
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

        logger.info(
            "Dense embedding: embedding %d source1 + %d other records",
            len(s1_texts_f), len(ot_texts_f),
        )
        s1_embeds = self._embed(s1_texts_f)
        ot_embeds = self._embed(ot_texts_f)

        # Cosine similarity via dot product (embeddings are L2-normalized)
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        top_k = min(self._top_k, len(ot_ids_f))

        for batch_start in range(0, len(s1_ids_f), _BATCH_SIZE_EMBED):
            batch_end = min(batch_start + _BATCH_SIZE_EMBED, len(s1_ids_f))
            s1_batch = s1_embeds[batch_start:batch_end]
            s1_ids_batch = s1_ids_f[batch_start:batch_end]

            sim = s1_batch @ ot_embeds.T  # (batch, |ot|)

            for row_idx, s1_id in enumerate(s1_ids_batch):
                row = sim[row_idx]
                if len(row) <= top_k:
                    top_indices = np.argsort(row)[::-1]
                else:
                    top_indices = np.argpartition(row, -top_k)[-top_k:]
                    top_indices = top_indices[np.argsort(row[top_indices])[::-1]]

                for idx in top_indices:
                    score = float(row[idx])
                    if score < self._min_similarity:
                        break
                    ot_id = ot_ids_f[idx]
                    pair = (s1_id, ot_id)
                    if pair not in seen:
                        seen.add(pair)
                        pairs.append(pair)

        return self._make_candidates(pairs, self.name)
