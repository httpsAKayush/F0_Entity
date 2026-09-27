"""
pipeline/shared.py — The ONE shared candidate-generation + feature-extraction path.

CRITICAL DESIGN RULE: This module is the single implementation of the full
    load → normalize → partition by country → block → extract features
pipeline.  Both train.py and predict.py call into this module.  There is NO
duplication of this logic between training and inference — this is the primary
defense against train/inference skew.

The only parameter that differs between training and inference usage is
``is_training``, which controls whether ground-truth DataFrames are threaded
through for negative sampling and recall diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from business_entity_resolution.blocking.base import BlockingStrategy
from business_entity_resolution.blocking.engine import BlockingDiagnostics, BlockingEngine
from business_entity_resolution.config import Config
from business_entity_resolution.features.pairwise import FEATURE_NAMES, FeatureExtractor
from business_entity_resolution.normalization.normalizer import (
    normalize_address,
    normalize_country,
    normalize_name,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalization step
# ---------------------------------------------------------------------------


def normalize_source_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized columns to a source DataFrame in-place.

    Adds:
        norm_name     : str  — normalized business name
        norm_address  : str  — normalized address (raw_normalized)
        street_number : str  — parsed street number or ""
        is_landmark   : int  — 1 if landmark-relative, 0 otherwise
        country_norm  : str  — normalized country code

    Parameters
    ----------
    df:
        Source DataFrame with columns: entity_id, business_name, business_address, country.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with additional columns.
    """
    logger.info("Normalizing %d records...", len(df))

    norm_names = []
    norm_addresses = []
    street_numbers = []
    is_landmarks = []
    country_norms = []

    for _, row in df.iterrows():
        country = str(row.get("country", ""))
        cnorm = normalize_country(country)
        country_norms.append(cnorm)

        name = str(row.get("business_name", ""))
        norm_names.append(normalize_name(name, cnorm))

        address = str(row.get("business_address", ""))
        addr_result = normalize_address(address, cnorm)
        norm_addresses.append(addr_result.raw_normalized)
        street_numbers.append(addr_result.street_number or "")
        is_landmarks.append(1 if addr_result.is_landmark_relative else 0)

    df = df.copy()
    df["country_norm"] = country_norms
    df["norm_name"] = norm_names
    df["norm_address"] = norm_addresses
    df["street_number"] = street_numbers
    df["is_landmark"] = is_landmarks

    logger.info("Normalization complete.")
    return df


def normalize_source_df_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized version of normalize_source_df using pandas apply (faster for large frames).
    Preferred for large files; falls back to row-by-row for safety.
    """
    logger.info("Normalizing %d records (vectorized)...", len(df))

    country_norms = df["country"].fillna("").apply(normalize_country)
    df = df.copy()
    df["country_norm"] = country_norms

    def _norm_name(row):
        return normalize_name(str(row["business_name"]), str(row["country_norm"]))

    def _norm_addr(row):
        return normalize_address(str(row["business_address"]), str(row["country_norm"]))

    df["norm_name"] = df.apply(_norm_name, axis=1)
    addr_results = df.apply(_norm_addr, axis=1)

    df["norm_address"] = addr_results.apply(lambda r: r.raw_normalized)
    df["street_number"] = addr_results.apply(lambda r: r.street_number or "")
    df["is_landmark"] = addr_results.apply(lambda r: 1 if r.is_landmark_relative else 0)

    logger.info("Normalization complete.")
    return df


# ---------------------------------------------------------------------------
# Blocking strategy factory
# ---------------------------------------------------------------------------


def build_blocking_strategies(cfg: Config) -> list[BlockingStrategy]:
    """
    Instantiate blocking strategies from the configuration.

    Strategies are instantiated in the order listed in
    cfg.blocking.active_strategies.  The optional dense_embedding_nn strategy
    is appended if cfg.blocking.use_dense_embedding is True.

    Parameters
    ----------
    cfg:
        Configuration object.

    Returns
    -------
    list[BlockingStrategy]
    """
    from business_entity_resolution.blocking.strategies.address_number import AddressNumberBlocking
    from business_entity_resolution.blocking.strategies.address_prefix import AddressPrefixBlocking
    from business_entity_resolution.blocking.strategies.name_prefix import NamePrefixBlocking
    from business_entity_resolution.blocking.strategies.name_suffix import NameSuffixBlocking
    from business_entity_resolution.blocking.strategies.rare_numeric_token import RareNumericTokenBlocking
    from business_entity_resolution.blocking.strategies.tfidf_nn import TFIDFNearestNeighborBlocking

    strategy_map: dict[str, BlockingStrategy] = {
        "name_prefix": NamePrefixBlocking(
            prefix_len=cfg.blocking.name_prefix.prefix_len,
            min_key_length=cfg.blocking.min_key_length,
        ),
        "name_suffix": NameSuffixBlocking(
            suffix_len=cfg.blocking.name_suffix.suffix_len,
            min_key_length=cfg.blocking.min_key_length,
        ),
        "address_number": AddressNumberBlocking(
            min_token_len=cfg.blocking.address_number.min_token_len,
            min_key_length=cfg.blocking.min_key_length,
        ),
        "address_prefix": AddressPrefixBlocking(
            prefix_len=cfg.blocking.address_prefix.prefix_len,
            min_key_length=cfg.blocking.min_key_length,
        ),
        "rare_numeric_token": RareNumericTokenBlocking(
            max_frequency=cfg.blocking.rare_numeric_token.max_frequency,
            min_key_length=cfg.blocking.min_key_length,
        ),
        "tfidf_nn": TFIDFNearestNeighborBlocking(
            top_k=cfg.blocking.tfidf_nn.top_k,
            min_similarity=cfg.blocking.tfidf_nn.min_similarity,
            max_features=cfg.blocking.tfidf_nn.max_features,
            ngram_range=cfg.blocking.tfidf_nn.ngram_range,
            analyzer=cfg.blocking.tfidf_nn.analyzer,
            min_key_length=cfg.blocking.min_key_length,
        ),
    }

    strategies = []
    for name in cfg.blocking.active_strategies:
        if name in strategy_map:
            strategies.append(strategy_map[name])
        else:
            logger.warning("Unknown blocking strategy in config: '%s', skipping.", name)

    if cfg.blocking.use_dense_embedding:
        from business_entity_resolution.blocking.strategies.dense_embedding_nn import DenseEmbeddingNNBlocking
        strategies.append(DenseEmbeddingNNBlocking(
            model_name=cfg.blocking.dense_embedding_nn.model_name,
            top_k=cfg.blocking.dense_embedding_nn.top_k,
            min_similarity=cfg.blocking.dense_embedding_nn.min_similarity,
            batch_size=cfg.blocking.dense_embedding_nn.batch_size,
        ))

    return strategies


# ---------------------------------------------------------------------------
# Per-source candidate generation result
# ---------------------------------------------------------------------------


@dataclass
class CandidateResult:
    """Result of candidate generation for one (source1, sourceX) pair."""
    candidates: pd.DataFrame        # (source1_entity_id, other_entity_id, strategy_name)
    source1_df: pd.DataFrame        # normalized source1 (all records, not just matched)
    other_df: pd.DataFrame          # normalized sourceX (all records)
    diagnostics: BlockingDiagnostics
    source_tag: str                 # "source2" or "source3"
    countries: list[str]            # list of countries processed


# ---------------------------------------------------------------------------
# THE shared pipeline
# ---------------------------------------------------------------------------


def run_candidate_generation(
    source1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    cfg: Config,
    source_tag: str = "source2",
    ground_truth_df: Optional[pd.DataFrame] = None,
) -> CandidateResult:
    """
    Full candidate generation pipeline for one (source1, sourceX) pair.

    Steps:
    1. Normalize source1_df and other_df (adds norm_name, norm_address, etc.)
    2. Partition both DataFrames by country_norm.
    3. For each country partition, run the BlockingEngine with all configured strategies.
    4. Union candidates across countries.
    5. Return candidates + diagnostics.

    This function is called identically by train.py and predict.py — no forking.
    The only difference is that ground_truth_df is provided during training for
    recall diagnostics.

    Parameters
    ----------
    source1_df:
        Source-1 DataFrame (entity_id, business_name, business_address, country).
    other_df:
        Source-2 or Source-3 DataFrame (same schema).
    cfg:
        Configuration object.
    source_tag:
        "source2" or "source3" — for labeling diagnostics.
    ground_truth_df:
        Optional ground truth (for training-time recall reporting).

    Returns
    -------
    CandidateResult
    """
    logger.info("=== Candidate generation for %s ===", source_tag)

    # Step 1: Normalize
    s1_norm = normalize_source_df_vectorized(source1_df)
    ot_norm = normalize_source_df_vectorized(other_df)

    from business_entity_resolution.blocking.duckdb_engine import generate_candidates_duckdb

    all_candidates, combined_diagnostics = generate_candidates_duckdb(
        s1_norm, 
        ot_norm, 
        candidate_cap=cfg.blocking.candidate_cap
    )

    all_countries = set(s1_norm["country_norm"].unique())
    logger.info(
        "%s candidate generation: %d total candidates via DuckDB out-of-core engine",
        source_tag, len(all_candidates)
    )

    return CandidateResult(
        candidates=all_candidates,
        source1_df=s1_norm,
        other_df=ot_norm,
        diagnostics=combined_diagnostics,
        source_tag=source_tag,
        countries=list(all_countries),
    )


def extract_features(
    candidate_result: CandidateResult,
    cfg: Config,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Extract pairwise similarity features from a CandidateResult.

    This is called identically by training and inference — same feature extractor,
    same FEATURE_NAMES column order, no risk of skew.

    Parameters
    ----------
    candidate_result:
        Output of run_candidate_generation().
    cfg:
        Configuration object.

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        (feature_matrix of shape (n_candidates, n_features), candidate_pairs_df)
    """
    candidates = candidate_result.candidates
    source1_df = candidate_result.source1_df
    other_df = candidate_result.other_df

    if candidates.empty:
        logger.warning("No candidates to extract features from.")
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), candidates

    extractor = FeatureExtractor()
    features = extractor.extract(candidates, source1_df, other_df)

    logger.info(
        "Feature extraction: %d candidates → matrix %s",
        len(candidates), features.shape,
    )

    return features, candidates


def prepare_training_data(
    candidate_result: CandidateResult,
    ground_truth_df: pd.DataFrame,
    cfg: Config,
    source_tag: str = "source2",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepare (X, y, candidate_pairs_df) for model training.

    Positive examples: candidate pairs that appear in ground_truth_df.
    Negative examples: candidate pairs that are NOT in ground_truth_df
                       (hard negatives from blocking — realistic difficulty).

    The negative:positive ratio is controlled by cfg.training.negative_positive_ratio.

    Parameters
    ----------
    candidate_result:
        Output of run_candidate_generation().
    ground_truth_df:
        Ground truth with columns: source1_entity_id, source_id, source_tag.
    cfg:
        Configuration object.
    source_tag:
        "source2" or "source3".

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DataFrame]
        (X: feature matrix, y: binary labels, pairs_df: candidate pairs used)
    """
    candidates = candidate_result.candidates
    if candidates.empty:
        return (
            np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            candidates,
        )

    # Build ground truth set for this source_tag
    gt_filtered = ground_truth_df[ground_truth_df["source_tag"] == source_tag]
    gt_set = set(zip(gt_filtered["source1_entity_id"], gt_filtered["source_id"]))

    # Label candidates via vectorized isin: construct a single string key per pair,
    # build the GT key set in the same format, then check membership in one pass.
    # This avoids a Python function call per row (apply axis=1 overhead).
    # Separator "|||" cannot appear in entity IDs (which use alphanumeric S1-/S2-/S3- format).
    candidates = candidates.copy()
    _SEP = "|||"
    pair_keys = candidates["source1_entity_id"] + _SEP + candidates["other_entity_id"]
    gt_keys = frozenset(
        s1id + _SEP + sid
        for s1id, sid in gt_set
    )
    candidates["is_match"] = pair_keys.isin(gt_keys).astype(int)

    positives = candidates[candidates["is_match"] == 1]
    negatives = candidates[candidates["is_match"] == 0]

    n_pos = len(positives)
    n_neg_target = n_pos * cfg.training.negative_positive_ratio

    if len(negatives) > n_neg_target:
        negatives = negatives.sample(n=int(n_neg_target), random_state=cfg.random_seed)

    # Hardware-Aware Training Strategy: Subsample 10% of the candidate pairs for training
    # to fit within 8GB RAM and 4GB VRAM limits, maintaining the positive/negative ratio.
    subsample_fraction = 0.10
    if len(positives) > 20:
        positives = positives.sample(frac=subsample_fraction, random_state=cfg.random_seed)
    if len(negatives) > 20:
        negatives = negatives.sample(frac=subsample_fraction, random_state=cfg.random_seed)

    training_candidates = pd.concat([positives, negatives], ignore_index=True)
    training_candidates = training_candidates.sample(
        frac=1.0, random_state=cfg.random_seed
    ).reset_index(drop=True)

    logger.info(
        "Training data: %d positives + %d negatives = %d total pairs (source_tag=%s)",
        n_pos, len(negatives), len(training_candidates), source_tag,
    )

    # Extract features for the training subset
    subset_result = CandidateResult(
        candidates=training_candidates[["source1_entity_id", "other_entity_id", "strategy_name"]],
        source1_df=candidate_result.source1_df,
        other_df=candidate_result.other_df,
        diagnostics=candidate_result.diagnostics,
        source_tag=source_tag,
        countries=candidate_result.countries,
    )
    X, _ = extract_features(subset_result, cfg)
    y = training_candidates["is_match"].values.astype(np.int32)

    return X, y, training_candidates
