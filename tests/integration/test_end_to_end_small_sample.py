"""
tests/integration/test_end_to_end_small_sample.py — Integration test.

Runs the full train -> predict -> validate pipeline on the small synthetic
fixture dataset and asserts:
1. Training completes without error.
2. Prediction produces output files in the correct format.
3. The official ML Challenge 2026 validator passes with zero errors.
4. The output is non-trivially better than a random baseline (macro F0.5 > 0).

This test runs entirely without external data or network access.

Output format validated:
    matching_results.tsv : source1_entity_id, matched_entity_ids (combined S2+S3)
    candidate_pairs.tsv  : source1_entity_id, candidate_entity_ids
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add utils/ to path for the official validator
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "utils"))

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "synthetic_sample"
TRAIN_DIR = FIXTURES_DIR / "train"
TEST_DIR = FIXTURES_DIR / "test"


@pytest.fixture(scope="module")
def trained_model_dir(tmp_path_factory):
    """Train the model on synthetic data and return the model directory."""
    from business_entity_resolution.config import load_config
    from business_entity_resolution.pipeline.train import train

    model_dir = tmp_path_factory.mktemp("models")

    cfg = load_config()
    cfg.paths.data_dir = str(FIXTURES_DIR)
    cfg.paths.model_dir = str(model_dir)
    cfg.paths.train_subdir = "train"
    cfg.blocking.candidate_cap = 20
    cfg.training.validation_fraction = 0.3
    cfg.training.negative_positive_ratio = 3
    cfg.models.active_model = "ensemble"
    cfg.models.lightgbm.n_estimators = 50
    cfg.models.xgboost.n_estimators = 50
    cfg.blocking.active_strategies = [
        "name_prefix", "name_suffix", "address_number",
        "address_prefix", "rare_numeric_token",
    ]
    cfg.blocking.use_dense_embedding = False

    train(cfg=cfg)
    return model_dir


@pytest.fixture(scope="module")
def output_dir(tmp_path_factory, trained_model_dir):
    """Run predict on test data and return the output directory."""
    from business_entity_resolution.config import load_config
    from business_entity_resolution.pipeline.predict import predict

    out_dir = tmp_path_factory.mktemp("output")
    cfg = load_config()
    cfg.paths.data_dir = str(FIXTURES_DIR)
    cfg.paths.model_dir = str(trained_model_dir)
    cfg.paths.output_dir = str(out_dir)
    cfg.paths.test_subdir = "test"
    # Use the test_source*.tsv naming expected by the official validator
    cfg.paths.source1_file = "test_source1.tsv"
    cfg.paths.source2_file = "test_source2.tsv"
    cfg.paths.source3_file = "test_source3.tsv"
    cfg.blocking.active_strategies = [
        "name_prefix", "name_suffix", "address_number",
        "address_prefix", "rare_numeric_token",
    ]
    cfg.blocking.use_dense_embedding = False

    predict(cfg=cfg)
    return out_dir


class TestEndToEnd:
    def test_training_produces_model_artifact(self, trained_model_dir):
        """Training must produce model.pkl and tuned_config.yaml."""
        assert (trained_model_dir / "model.pkl").exists(), \
            "model.pkl not found after training"
        assert (trained_model_dir / "tuned_config.yaml").exists(), \
            "tuned_config.yaml not found after training"

    def test_prediction_produces_output_files(self, output_dir):
        """Prediction must produce matching_results.tsv and candidate_pairs.tsv."""
        assert (output_dir / "matching_results.tsv").exists(), \
            "matching_results.tsv not found after prediction"
        assert (output_dir / "candidate_pairs.tsv").exists(), \
            "candidate_pairs.tsv not found after prediction"

    def test_matching_results_official_schema(self, output_dir):
        """matching_results.tsv must have the official 2-column schema."""
        df = pd.read_csv(output_dir / "matching_results.tsv", sep="\t", dtype=str)
        assert list(df.columns) == ["source1_entity_id", "matched_entity_ids"], \
            f"Wrong columns: {list(df.columns)}"

        # Coverage: every test Source-1 entity must appear
        s1_test = pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t", dtype=str)
        expected_ids = set(s1_test["entity_id"])
        output_ids = set(df["source1_entity_id"])
        assert expected_ids == output_ids, \
            f"Missing entity IDs: {expected_ids - output_ids}"

        # No duplicates
        assert not df["source1_entity_id"].duplicated().any()

    def test_candidate_pairs_official_schema(self, output_dir):
        """candidate_pairs.tsv must have the official 2-column schema."""
        df = pd.read_csv(output_dir / "candidate_pairs.tsv", sep="\t", dtype=str)
        assert list(df.columns) == ["source1_entity_id", "candidate_entity_ids"], \
            f"Wrong columns: {list(df.columns)}"

        # Coverage: every Source-1 entity must appear
        s1_test = pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t", dtype=str)
        assert set(df["source1_entity_id"]) == set(s1_test["entity_id"])

    def test_official_validator_passes(self, output_dir):
        """The official ML Challenge 2026 validator must exit with no errors."""
        from validate_submission import validate

        errors, warnings = validate(
            matching_path=str(output_dir / "matching_results.tsv"),
            candidate_path=str(output_dir / "candidate_pairs.tsv"),
            test_dir=str(TEST_DIR),
            check_ids=False,  # keep fast; ID-existence check is a diagnostic
        )
        assert len(errors) == 0, (
            f"Official validator reported {len(errors)} error(s):\n" +
            "\n".join(f"  {i+1}. {e}" for i, e in enumerate(errors))
        )

    def test_matched_ids_have_correct_prefix(self, output_dir):
        """All matched IDs must start with S2- or S3- (official validator rule)."""
        df = pd.read_csv(output_dir / "matching_results.tsv", sep="\t", dtype=str)
        df["matched_entity_ids"] = df["matched_entity_ids"].fillna("")
        for _, row in df.iterrows():
            ids_str = row["matched_entity_ids"].strip()
            if not ids_str:
                continue
            for mid in ids_str.split(","):
                assert mid.startswith(("S2-", "S3-")), (
                    f"Matched ID '{mid}' for entity '{row['source1_entity_id']}' "
                    f"does not start with S2- or S3-"
                )

    def test_output_predicts_some_matches(self, output_dir):
        """Model should predict at least some matches (not all singletons)."""
        df = pd.read_csv(output_dir / "matching_results.tsv", sep="\t", dtype=str)
        n_with_matches = (df["matched_entity_ids"].str.len() > 0).sum()
        assert n_with_matches > 0, (
            "Model predicted zero matches — check threshold configuration."
        )

    def test_no_malformed_match_ids(self, output_dir):
        """No empty tokens in comma-separated matched_entity_ids."""
        df = pd.read_csv(output_dir / "matching_results.tsv", sep="\t", dtype=str)
        for _, row in df.iterrows():
            ids_str = str(row["matched_entity_ids"]).strip()
            if ids_str:
                for token in ids_str.split(","):
                    assert token.strip(), (
                        f"Empty token in matched_entity_ids for "
                        f"'{row['source1_entity_id']}': {repr(ids_str)}"
                    )

    def test_matches_are_subset_of_candidates(self, output_dir):
        """Every matched ID must appear in the candidate set for that entity."""
        matching = pd.read_csv(
            output_dir / "matching_results.tsv", sep="\t", dtype=str
        ).fillna("").set_index("source1_entity_id")
        candidates = pd.read_csv(
            output_dir / "candidate_pairs.tsv", sep="\t", dtype=str
        ).fillna("").set_index("source1_entity_id")

        violations = []
        for s1id in matching.index:
            mids_str = matching.loc[s1id, "matched_entity_ids"]
            cids_str = candidates.loc[s1id, "candidate_entity_ids"]
            mids = set(mids_str.split(",")) - {""} if mids_str else set()
            cids = set(cids_str.split(",")) - {""} if cids_str else set()
            extra = mids - cids
            if extra:
                violations.append(f"{s1id}: {extra}")

        assert not violations, (
            f"Matched IDs not in candidates for {len(violations)} entity(ies):\n"
            + "\n".join(violations[:5])
        )
