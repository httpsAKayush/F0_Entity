"""
pipeline/predict.py — End-to-end inference entry point.

Workflow:
1. Load saved model artifact + tuned config.
2. Load and validate test source1, source2, source3 files.
3. Run candidate generation (shared.py) — IDENTICAL code path as training.
4. Score candidates using the loaded model.
5. Aggregate to one row per Source-1 entity.
6. Write output/matching_results.tsv and output/candidate_pairs.tsv.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from business_entity_resolution.aggregation.aggregator import Aggregator
from business_entity_resolution.config import Config, load_config
from business_entity_resolution.features.pairwise import FEATURE_NAMES
from business_entity_resolution.io.loaders import load_source1, load_source2, load_source3
from business_entity_resolution.pipeline.shared import extract_features, run_candidate_generation

logger = logging.getLogger(__name__)


def _setup_logging(cfg: Config) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        format=cfg.logging.format,
    )


def predict(cfg: Optional[Config] = None, config_path: Optional[str] = None) -> None:
    """
    Run the full inference pipeline.

    Parameters
    ----------
    cfg:
        Pre-loaded Config object.  If None, loads from *config_path*.
    config_path:
        Path to YAML config file.
    """
    t_start = time.time()

    if cfg is None:
        cfg = load_config(config_path)

    _setup_logging(cfg)
    logger.info("=== Business Entity Resolution - INFERENCE ===")

    # Resolve paths
    test_dir = cfg.paths.test_dir()
    model_dir = cfg.paths.model_path()
    output_dir = cfg.paths.output_path()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model artifact
    model_artifact_path = model_dir / cfg.paths.model_artifact_file
    if not model_artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_artifact_path}. "
            "Run 'ber train' first to produce a trained model."
        )

    logger.info("Loading model from %s", model_artifact_path)
    with model_artifact_path.open("rb") as f:
        model = pickle.load(f)

    # Load tuned config
    tuned_config_path = model_dir / cfg.paths.config_artifact_file
    threshold = cfg.decision.default_threshold
    per_country_thresholds: dict[str, float] = {}

    if tuned_config_path.exists():
        with tuned_config_path.open("r") as f:
            tuned = yaml.safe_load(f) or {}
        threshold = tuned.get("best_threshold", threshold)
        per_country_thresholds = tuned.get("thresholds", {})
        logger.info("Loaded tuned threshold: %.2f", threshold)
    else:
        logger.warning("No tuned config found at %s; using default threshold %.2f",
                       tuned_config_path, threshold)

    # Load test data
    logger.info("Loading test data from %s", test_dir)
    source1_df = load_source1(
        test_dir / cfg.paths.source1_file,
        encoding=cfg.io.source_encoding,
        delimiter=cfg.io.delimiter,
    )
    source2_df = load_source2(
        test_dir / cfg.paths.source2_file,
        encoding=cfg.io.source_encoding,
        delimiter=cfg.io.delimiter,
    )
    source3_df = load_source3(
        test_dir / cfg.paths.source3_file,
        encoding=cfg.io.source_encoding,
        delimiter=cfg.io.delimiter,
    )

    all_s1_ids = source1_df["entity_id"].tolist()

    # Run candidate generation (same code path as training — no skew possible)
    logger.info("--- Source 2: candidate generation ---")
    cand_result_s2 = run_candidate_generation(
        source1_df, source2_df, cfg, source_tag="source2"
    )

    logger.info("--- Source 3: candidate generation ---")
    cand_result_s3 = run_candidate_generation(
        source1_df, source3_df, cfg, source_tag="source3"
    )

    # Chunked inference to stay under RAM limits
    chunk_size = 250000
    
    def _score_in_chunks(cand_result, model, cfg):
        cands = cand_result.candidates
        if cands.empty:
            return pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "score"])
            
        from business_entity_resolution.pipeline.shared import CandidateResult
        
        scored_chunks = []
        for start in range(0, len(cands), chunk_size):
            chunk_cands = cands.iloc[start:start+chunk_size]
            chunk_res = CandidateResult(
                candidates=chunk_cands,
                source1_df=cand_result.source1_df,
                other_df=cand_result.other_df,
                diagnostics=cand_result.diagnostics,
                source_tag=cand_result.source_tag,
                countries=cand_result.countries,
            )
            X_chunk, cands_chunk = extract_features(chunk_res, cfg)
            if len(X_chunk) > 0:
                scored = cands_chunk[["source1_entity_id", "other_entity_id"]].copy()
                scored["score"] = model.predict_proba(X_chunk)
                scored_chunks.append(scored)
                
        if scored_chunks:
            return pd.concat(scored_chunks, ignore_index=True)
        return pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "score"])

    scored_s2 = _score_in_chunks(cand_result_s2, model, cfg)
    scored_s3 = _score_in_chunks(cand_result_s3, model, cfg)

    # Build country map for per-entity threshold lookup
    source1_country_map: dict[str, str] = {}
    if per_country_thresholds and cand_result_s2.source1_df is not None:
        s1_norm = cand_result_s2.source1_df
        source1_country_map = dict(zip(s1_norm["entity_id"], s1_norm["country_norm"]))

    # Aggregate — returns (matching_results_df, candidate_pairs_df)
    aggregator = Aggregator(
        threshold=threshold,
        per_country_thresholds=per_country_thresholds if cfg.decision.per_country else None,
        source1_country_map=source1_country_map if cfg.decision.per_country else None,
    )

    results_df, candidate_pairs_df = aggregator.aggregate(
        all_source1_ids=all_s1_ids,
        scored_pairs_s2=scored_s2 if not scored_s2.empty else None,
        scored_pairs_s3=scored_s3 if not scored_s3.empty else None,
    )

    # Write outputs (official format: UTF-8 tab-separated)
    matching_results_path = output_dir / cfg.paths.matching_results_file
    candidate_pairs_path = output_dir / cfg.paths.candidate_pairs_file

    results_df.to_csv(
        matching_results_path, sep=cfg.io.delimiter, index=False, encoding="utf-8"
    )
    logger.info("Wrote matching results: %s (%d rows)", matching_results_path, len(results_df))

    candidate_pairs_df.to_csv(
        candidate_pairs_path, sep=cfg.io.delimiter, index=False, encoding="utf-8"
    )
    logger.info(
        "Wrote candidate pairs: %s (%d rows)", candidate_pairs_path, len(candidate_pairs_df)
    )

    n_with_matches = results_df["matched_entity_ids"].str.len() > 0
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"  Total Source-1 entities: {len(results_df):,}")
    print(f"  Entities with >=1 match: {n_with_matches.sum():,}")
    print(f"  Predicted singletons:    {(~n_with_matches).sum():,}")
    print(f"  Total candidate entries: {len(candidate_pairs_df):,}")
    print(f"  Elapsed time: {time.time() - t_start:.1f}s")
    print(f"\n  Output files:")
    print(f"    {matching_results_path}")
    print(f"    {candidate_pairs_path}")
    print("=" * 60 + "\n")

