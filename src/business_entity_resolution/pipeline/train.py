"""
pipeline/train.py — End-to-end training entry point.

Workflow:
1. Load and validate source1, source2, source3, ground_truth files.
2. Entity-level (NOT pair-level) train/validation split to prevent data leakage.
3. Run candidate generation (shared.py) on training partition.
4. Print blocking recall ceiling diagnostic.
5. Prepare training data with hard negatives.
6. Fit the configured model(s).
7. Run candidate generation on validation partition.
8. Score validation candidates.
9. Tune decision threshold against macro F0.5 on validation set.
10. Print full diagnostics block.
11. Save model artifacts + tuned config.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from business_entity_resolution.config import Config, load_config
from business_entity_resolution.decision.threshold_tuning import (
    tune_per_country_thresholds,
    tune_threshold,
)
from business_entity_resolution.evaluation.metrics import (
    build_ground_truth_dict,
    evaluate,
)
from business_entity_resolution.io.loaders import (
    load_ground_truth,
    load_source1,
    load_source2,
    load_source3,
)
from business_entity_resolution.models.base import MatcherModel
from business_entity_resolution.models.ensemble import EnsembleMatcher
from business_entity_resolution.models.lightgbm_matcher import LightGBMMatcher
from business_entity_resolution.models.xgboost_matcher import XGBoostMatcher
from business_entity_resolution.pipeline.shared import (
    extract_features,
    prepare_training_data,
    run_candidate_generation,
)

logger = logging.getLogger(__name__)


def _setup_logging(cfg: Config) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        format=cfg.logging.format,
    )


def _build_model(cfg: Config) -> MatcherModel:
    """Instantiate the configured model(s)."""
    lgbm = LightGBMMatcher(**{
        "n_estimators": cfg.models.lightgbm.n_estimators,
        "learning_rate": cfg.models.lightgbm.learning_rate,
        "num_leaves": cfg.models.lightgbm.num_leaves,
        "max_depth": cfg.models.lightgbm.max_depth,
        "min_child_samples": cfg.models.lightgbm.min_child_samples,
        "subsample": cfg.models.lightgbm.subsample,
        "colsample_bytree": cfg.models.lightgbm.colsample_bytree,
        "reg_alpha": cfg.models.lightgbm.reg_alpha,
        "reg_lambda": cfg.models.lightgbm.reg_lambda,
        "n_jobs": cfg.models.lightgbm.n_jobs,
        "random_state": cfg.models.lightgbm.random_state,
        "verbose": cfg.models.lightgbm.verbose,
    })

    xgb = XGBoostMatcher(**{
        "n_estimators": cfg.models.xgboost.n_estimators,
        "learning_rate": cfg.models.xgboost.learning_rate,
        "max_depth": cfg.models.xgboost.max_depth,
        "min_child_weight": cfg.models.xgboost.min_child_weight,
        "subsample": cfg.models.xgboost.subsample,
        "colsample_bytree": cfg.models.xgboost.colsample_bytree,
        "reg_alpha": cfg.models.xgboost.reg_alpha,
        "reg_lambda": cfg.models.xgboost.reg_lambda,
        "n_jobs": cfg.models.xgboost.n_jobs,
        "random_state": cfg.models.xgboost.random_state,
        "verbosity": cfg.models.xgboost.verbosity,
        "eval_metric": cfg.models.xgboost.eval_metric,
    })

    active = cfg.models.active_model
    if active == "lightgbm":
        return lgbm
    elif active == "xgboost":
        return xgb
    else:  # ensemble (default)
        weights = cfg.models.ensemble.weights
        return EnsembleMatcher.from_models_and_weights([lgbm, xgb], weights)


def train(cfg: Optional[Config] = None, config_path: Optional[str] = None) -> None:
    """
    Run the full training pipeline.

    Parameters
    ----------
    cfg:
        Pre-loaded Config object.  If None, loads from *config_path*.
    config_path:
        Path to YAML config file.  Ignored if *cfg* is provided.
    """
    t_start = time.time()

    if cfg is None:
        cfg = load_config(config_path)

    _setup_logging(cfg)
    logger.info("=== Business Entity Resolution - TRAINING ===")
    logger.info("Config: %s", config_path or "default")

    # Resolve paths
    train_dir = cfg.paths.train_dir()
    model_dir = cfg.paths.model_path()
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading training data from %s", train_dir)
    source1_df = load_source1(
        train_dir / cfg.paths.source1_file,
        encoding=cfg.io.source_encoding,
        delimiter=cfg.io.delimiter,
    )
    source2_df = load_source2(
        train_dir / cfg.paths.source2_file,
        encoding=cfg.io.source_encoding,
        delimiter=cfg.io.delimiter,
    )
    source3_df = load_source3(
        train_dir / cfg.paths.source3_file,
        encoding=cfg.io.source_encoding,
        delimiter=cfg.io.delimiter,
    )
    ground_truth_df = load_ground_truth(
        train_dir / cfg.paths.ground_truth_file,
        encoding=cfg.io.source_encoding,
        delimiter=cfg.io.delimiter,
    )

    # Entity-level train/val split (split on Source-1 entities, not pairs)
    # This prevents leakage: the model never sees a validation entity's pairs during training.
    np.random.seed(cfg.random_seed)
    all_s1_ids = source1_df["entity_id"].tolist()  # convert to plain list for shuffle
    np.random.shuffle(all_s1_ids)
    n_val = max(1, int(len(all_s1_ids) * cfg.training.validation_fraction))
    val_ids = set(all_s1_ids[:n_val])
    train_ids = set(all_s1_ids[n_val:])

    logger.info("Train/val split: %d train entities, %d val entities", len(train_ids), len(val_ids))

    s1_train = source1_df[source1_df["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val = source1_df[source1_df["entity_id"].isin(val_ids)].reset_index(drop=True)
    gt_train = ground_truth_df[ground_truth_df["source1_entity_id"].isin(train_ids)]
    gt_val = ground_truth_df[ground_truth_df["source1_entity_id"].isin(val_ids)]

    # ----------------------------------------------------------------
    # Run candidate generation + feature extraction for TRAINING data
    # ----------------------------------------------------------------
    logger.info("--- Source 2: Training candidate generation ---")
    cand_result_s2_train = run_candidate_generation(
        s1_train, source2_df, cfg, source_tag="source2", ground_truth_df=gt_train
    )
    cand_result_s2_train.diagnostics.print_summary(prefix="[S2-TRAIN] ")

    logger.info("--- Source 3: Training candidate generation ---")
    cand_result_s3_train = run_candidate_generation(
        s1_train, source3_df, cfg, source_tag="source3", ground_truth_df=gt_train
    )
    cand_result_s3_train.diagnostics.print_summary(prefix="[S3-TRAIN] ")

    # Prepare training data (X, y)
    X_s2, y_s2, _ = prepare_training_data(cand_result_s2_train, gt_train, cfg, "source2")
    X_s3, y_s3, _ = prepare_training_data(cand_result_s3_train, gt_train, cfg, "source3")

    # Combine source2 and source3 training data (same feature space)
    X_train = np.vstack([X_s2, X_s3]) if len(X_s2) and len(X_s3) else (X_s2 if len(X_s2) else X_s3)
    y_train = np.concatenate([y_s2, y_s3]) if len(y_s2) and len(y_s3) else (y_s2 if len(y_s2) else y_s3)

    logger.info("Total training pairs: %d (%d positive)", len(y_train), y_train.sum())

    # ----------------------------------------------------------------
    # Fit model
    # ----------------------------------------------------------------
    model = _build_model(cfg)
    logger.info("Fitting model: %s", type(model).__name__)
    model.fit(X_train, y_train)

    # ----------------------------------------------------------------
    # Run candidate generation + feature extraction for VALIDATION data
    # ----------------------------------------------------------------
    logger.info("--- Source 2: Validation candidate generation ---")
    cand_result_s2_val = run_candidate_generation(
        s1_val, source2_df, cfg, source_tag="source2", ground_truth_df=gt_val
    )

    logger.info("--- Source 3: Validation candidate generation ---")
    cand_result_s3_val = run_candidate_generation(
        s1_val, source3_df, cfg, source_tag="source3", ground_truth_df=gt_val
    )

    # Score validation candidates
    X_val_s2, cands_s2_val = extract_features(cand_result_s2_val, cfg)
    X_val_s3, cands_s3_val = extract_features(cand_result_s3_val, cfg)

    scored_s2_val = cands_s2_val.copy() if not cands_s2_val.empty else pd.DataFrame(
        columns=["source1_entity_id", "other_entity_id", "strategy_name"]
    )
    scored_s3_val = cands_s3_val.copy() if not cands_s3_val.empty else pd.DataFrame(
        columns=["source1_entity_id", "other_entity_id", "strategy_name"]
    )

    if len(X_val_s2) > 0 and not scored_s2_val.empty:
        scored_s2_val = scored_s2_val.copy()
        scored_s2_val["score"] = model.predict_proba(X_val_s2)
    if len(X_val_s3) > 0 and not scored_s3_val.empty:
        scored_s3_val = scored_s3_val.copy()
        scored_s3_val["score"] = model.predict_proba(X_val_s3)

    # Combine s2 + s3 scored pairs for threshold tuning
    all_scored_val = pd.concat(
        [df for df in [scored_s2_val, scored_s3_val] if "score" in df.columns and not df.empty],
        ignore_index=True,
    )
    # threshold_tuning._make_predictions expects: source1_entity_id, other_entity_id, score
    # — don't rename; pass the original column name directly
    all_scored_val_for_tuning = all_scored_val  # already has other_entity_id

    # ----------------------------------------------------------------
    # Threshold tuning
    # ----------------------------------------------------------------
    val_s1_ids = s1_val["entity_id"].tolist()

    # We tune threshold on combined source2+source3 scored pairs
    # by comparing against the combined ground truth
    if not all_scored_val_for_tuning.empty:
        if cfg.decision.per_country:
            thresholds = tune_per_country_thresholds(
                all_scored_val_for_tuning,
                gt_val,
                cand_result_s2_val.source1_df,
                cfg.decision.threshold_grid,
                cfg.decision.default_threshold,
                cfg.decision.min_country_val_size,
            )
            best_thresh = thresholds.get("_default", cfg.decision.default_threshold)
        else:
            best_thresh, val_result = tune_threshold(
                all_scored_val_for_tuning,
                gt_val,
                val_s1_ids,
                cfg.decision.threshold_grid,
                cfg.decision.default_threshold,
            )
            thresholds = {"_default": best_thresh}
    else:
        best_thresh = cfg.decision.default_threshold
        thresholds = {"_default": best_thresh}
        logger.warning("No scored validation pairs - using default threshold %.2f", best_thresh)

    # ----------------------------------------------------------------
    # Final validation evaluation at best threshold
    # ----------------------------------------------------------------
    combined_predictions: dict[str, set[str]] = {eid: set() for eid in val_s1_ids}

    for scored_df, source_t in [(scored_s2_val, "source2"), (scored_s3_val, "source3")]:
        if "score" not in (scored_df.columns if not scored_df.empty else []):
            continue
        matches = scored_df[scored_df["score"] >= best_thresh]
        for _, row in matches.iterrows():
            s1id = row["source1_entity_id"]
            if s1id in combined_predictions:
                combined_predictions[s1id].add(row["other_entity_id"])

    gt_dict_val = build_ground_truth_dict(gt_val, val_s1_ids)
    final_result = evaluate(combined_predictions, gt_dict_val)

    # ----------------------------------------------------------------
    # Print diagnostics block
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - FULL DIAGNOSTICS BLOCK")
    print("=" * 70)
    print(f"  Total training time: {time.time() - t_start:.1f}s")
    print(f"\n  Source-2 blocking recall ceiling: "
          f"{cand_result_s2_train.diagnostics.blocking_recall_ceiling or 'N/A':.4f}"
          if cand_result_s2_train.diagnostics.blocking_recall_ceiling is not None
          else "\n  Source-2 blocking recall ceiling: N/A (no GT provided)")
    print(f"  Source-3 blocking recall ceiling: "
          f"{cand_result_s3_train.diagnostics.blocking_recall_ceiling or 'N/A':.4f}"
          if cand_result_s3_train.diagnostics.blocking_recall_ceiling is not None
          else "  Source-3 blocking recall ceiling: N/A")
    print(f"\n  Chosen threshold: {best_thresh:.2f}")
    print(f"\n  Holdout macro F0.5:    {final_result.macro_f05:.4f}")
    print(f"  Holdout precision:     {final_result.macro_precision:.4f}")
    print(f"  Holdout recall:        {final_result.macro_recall:.4f}")
    print(f"  Singleton accuracy:    {final_result.singleton_accuracy:.4f}  "
          f"({final_result.n_singletons} singletons)")
    print(f"  Non-singleton entities: {final_result.n_non_singletons}")
    print("=" * 70 + "\n")

    # ----------------------------------------------------------------
    # Save artifacts
    # ----------------------------------------------------------------
    model_artifact_path = model_dir / cfg.paths.model_artifact_file
    with model_artifact_path.open("wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", model_artifact_path)

    tuned_config = {
        "thresholds": thresholds,
        "best_threshold": best_thresh,
        "per_country": cfg.decision.per_country,
        "val_macro_f05": final_result.macro_f05,
        "val_singleton_accuracy": final_result.singleton_accuracy,
    }
    config_artifact_path = model_dir / cfg.paths.config_artifact_file
    with config_artifact_path.open("w") as f:
        yaml.dump(tuned_config, f, default_flow_style=False)
    logger.info("Tuned config saved to %s", config_artifact_path)

    logger.info("Training complete. Total time: %.1fs", time.time() - t_start)
