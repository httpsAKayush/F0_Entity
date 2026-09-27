"""
tests/unit/test_regressions.py — Regression tests for previously reported bugs.

Each test is named after the bug it guards against.  New regressions should be
added here with a brief comment explaining what broke and what the fix was.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Bug: train.py f-string crash when blocking_recall_ceiling == 0.0
# ---------------------------------------------------------------------------


def test_recall_ceiling_zero_does_not_crash_format():
    """
    Regression: blocking_recall_ceiling = 0.0 (falsy) was formatted via
    ``or 'N/A'`` inside an f-string with ``:.4f``, producing
    ``f"{'N/A':.4f}"`` → ValueError.

    The fix uses explicit ``is not None`` checks.
    """
    # Simulate the diagnostics-print logic with ceiling=0.0
    ceiling = 0.0

    # Old (broken) pattern — kept here as documentation:
    # f"{ceiling or 'N/A':.4f}"  # → ValueError: unsupported format character

    # New (fixed) pattern — must not raise:
    result = f"{ceiling:.4f}" if ceiling is not None else "N/A (no GT provided)"
    assert result == "0.0000", f"Expected '0.0000', got {result!r}"

    # Also test with None (no GT):
    ceiling_none = None
    result_none = f"{ceiling_none:.4f}" if ceiling_none is not None else "N/A (no GT provided)"
    assert result_none == "N/A (no GT provided)"


def test_recall_ceiling_zero_in_blocking_diagnostics():
    """
    Regression: EvaluationResult.macro_f05=0.0 and blocking_recall_ceiling=0.0
    should both be representable without crashing the train pipeline.
    """
    from business_entity_resolution.evaluation.metrics import EvaluationResult

    result = EvaluationResult(
        macro_f05=0.0,
        macro_precision=0.0,
        macro_recall=0.0,
        singleton_accuracy=1.0,
        n_entities=5,
        n_singletons=5,
        n_non_singletons=0,
    )
    # Must not raise
    formatted = f"{result.macro_f05:.4f}"
    assert formatted == "0.0000"


# ---------------------------------------------------------------------------
# Bug: cli.py / package_submission.py imported non-existent validate_submission()
# ---------------------------------------------------------------------------


def test_validate_submission_py_exposes_validate_function():
    """
    Regression: cli.py and package_submission.py both tried to import
    ``validate_submission()`` from utils/validate_submission.py, but that
    module only exposes ``validate()`` with a different signature.

    The fix updates both callers to import and call ``validate()`` correctly.
    This test verifies the real API is importable and callable.
    """
    _repo_root = Path(__file__).parent.parent.parent
    utils_dir = str(_repo_root / "utils")
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)

    # Must import without error
    from validate_submission import validate  # noqa: F401

    # Signature check: validate(matching_path, candidate_path, test_dir, check_ids=False)
    import inspect
    sig = inspect.signature(validate)
    params = list(sig.parameters.keys())
    assert "matching_path" in params, f"Expected 'matching_path' in {params}"
    assert "candidate_path" in params, f"Expected 'candidate_path' in {params}"
    assert "test_dir" in params, f"Expected 'test_dir' in {params}"
    assert "check_ids" in params, f"Expected 'check_ids' in {params}"

    # validate_submission does NOT exist (the old broken name):
    import importlib
    mod = importlib.import_module("validate_submission")
    assert not hasattr(mod, "validate_submission"), (
        "validate_submission() should not exist — the real API is validate()"
    )


def test_cli_validate_command_uses_correct_api(tmp_path):
    """
    Regression: ``ber validate`` called validate_submission(output_dir=...) which
    does not exist.  The fix calls validate(matching_path, candidate_path, test_dir,
    check_ids=False).

    Smoke-test: import the CLI module and verify the validate command exists with
    the corrected options (--test-dir, --check-ids) and NOT --source1-ids.
    """
    import click.testing
    from business_entity_resolution.cli import main

    runner = click.testing.CliRunner()

    # --help should list --test-dir and --check-ids, NOT --source1-ids
    result = runner.invoke(main, ["validate", "--help"])
    assert result.exit_code == 0, f"validate --help failed: {result.output}"
    assert "--test-dir" in result.output, (
        "--test-dir option missing from 'ber validate --help'"
    )
    assert "--check-ids" in result.output, (
        "--check-ids option missing from 'ber validate --help'"
    )
    assert "--source1-ids" not in result.output, (
        "--source1-ids (old broken option) should be gone from 'ber validate'"
    )


def test_cli_package_command_passes_test_dir_and_check_ids():
    """
    Regression: ``ber package`` called scripts/package_submission.py without
    --test-dir or --check-ids.  The fix adds both options to the package command.
    """
    import click.testing
    from business_entity_resolution.cli import main

    runner = click.testing.CliRunner()
    result = runner.invoke(main, ["package", "--help"])
    assert result.exit_code == 0, f"package --help failed: {result.output}"
    assert "--test-dir" in result.output
    assert "--check-ids" in result.output


# ---------------------------------------------------------------------------
# Bug: io/schemas.py had wrong output schema columns
# ---------------------------------------------------------------------------


def test_matching_results_schema_has_correct_columns():
    """
    Regression: MATCHING_RESULTS_SCHEMA declared source2_match_ids /
    source3_match_ids but the Aggregator and official validator expect
    matched_entity_ids.

    load_matching_results() must succeed on a file with the correct columns.
    """
    from business_entity_resolution.io.schemas import MATCHING_RESULTS_SCHEMA

    cols = MATCHING_RESULTS_SCHEMA.column_names()
    assert "matched_entity_ids" in cols, (
        f"matched_entity_ids missing from MATCHING_RESULTS_SCHEMA columns: {cols}"
    )
    assert "source2_match_ids" not in cols, (
        f"source2_match_ids (old wrong column) still in schema: {cols}"
    )
    assert "source3_match_ids" not in cols, (
        f"source3_match_ids (old wrong column) still in schema: {cols}"
    )


def test_candidate_pairs_schema_has_correct_columns():
    """
    Regression: CANDIDATE_PAIRS_SCHEMA declared candidate_entity_id + source_tag
    + score but the Aggregator produces candidate_entity_ids (comma-separated).
    """
    from business_entity_resolution.io.schemas import CANDIDATE_PAIRS_SCHEMA

    cols = CANDIDATE_PAIRS_SCHEMA.column_names()
    assert "candidate_entity_ids" in cols, (
        f"candidate_entity_ids missing from CANDIDATE_PAIRS_SCHEMA: {cols}"
    )
    assert "candidate_entity_id" not in cols, (
        f"candidate_entity_id (old wrong column) still in schema: {cols}"
    )


# ---------------------------------------------------------------------------
# Bug: entity_id uniqueness not validated — duplicates would silently corrupt
# ---------------------------------------------------------------------------


def test_validate_dataframe_rejects_duplicate_entity_ids(tmp_path):
    """
    Regression: duplicate entity_id values in source files were silently
    accepted and would corrupt set_index("entity_id") lookups.

    The fix adds unique=True to the entity_id ColumnSpec and wires a
    SchemaError check into validate_dataframe.
    """
    from business_entity_resolution.io.loaders import SchemaError, validate_dataframe
    from business_entity_resolution.io.schemas import SOURCE1_SCHEMA, ColumnSpec, SchemaSpec

    # Build a schema with unique=True on entity_id
    unique_schema = SchemaSpec(
        file_label="test_source",
        columns=(
            ColumnSpec("entity_id", required=True, unique=True),
            ColumnSpec("business_name", required=True),
            ColumnSpec("business_address", required=True),
            ColumnSpec("country", required=True),
        ),
    )

    # DataFrame with duplicate entity_id
    df_dup = pd.DataFrame({
        "entity_id": ["S1-001", "S1-001", "S1-002"],
        "business_name": ["A", "B", "C"],
        "business_address": ["x", "y", "z"],
        "country": ["US", "US", "US"],
    })

    with pytest.raises(SchemaError, match="duplicate"):
        validate_dataframe(df_dup, unique_schema)

    # DataFrame without duplicates should pass
    df_ok = pd.DataFrame({
        "entity_id": ["S1-001", "S1-002", "S1-003"],
        "business_name": ["A", "B", "C"],
        "business_address": ["x", "y", "z"],
        "country": ["US", "US", "US"],
    })
    validate_dataframe(df_ok, unique_schema)  # must not raise


# ---------------------------------------------------------------------------
# Bug: io/loaders.py crashed when called with str filepath instead of Path
# ---------------------------------------------------------------------------


def test_load_source1_accepts_str_path(tmp_path):
    """
    Regression: load_source1() accepted only Path objects; calling it with a
    str would crash with AttributeError ('str' object has no attribute 'exists').
    """
    from business_entity_resolution.io.loaders import load_source1

    # Write a minimal source1 TSV
    f = tmp_path / "source1.tsv"
    f.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-001\tAcme Inc\t123 Main St\tUS\n",
        encoding="utf-8",
    )

    # Call with str — must not raise
    df = load_source1(str(f))
    assert len(df) == 1
    assert df["entity_id"].iloc[0] == "S1-001"


# ---------------------------------------------------------------------------
# Bug: build_ground_truth_dict used iterrows — vectorized replacement
# ---------------------------------------------------------------------------


def test_build_ground_truth_dict_handles_empty_df():
    """
    Regression guard: vectorized build_ground_truth_dict must handle empty input
    without crashing (iterrows on empty DF is fine; groupby on empty DF needs care).
    """
    from business_entity_resolution.evaluation.metrics import build_ground_truth_dict

    source1_ids = ["S1-001", "S1-002", "S1-003"]
    empty_gt = pd.DataFrame(columns=["source1_entity_id", "source_id", "source_tag"])

    result = build_ground_truth_dict(empty_gt, source1_ids)
    assert result == {"S1-001": set(), "S1-002": set(), "S1-003": set()}


def test_build_ground_truth_dict_correct_grouping():
    """Vectorized groupby must produce the same output as the old iterrows version."""
    from business_entity_resolution.evaluation.metrics import build_ground_truth_dict

    gt = pd.DataFrame({
        "source1_entity_id": ["S1-001", "S1-001", "S1-002"],
        "source_id": ["S2-001", "S3-001", "S2-002"],
        "source_tag": ["source2", "source3", "source2"],
    })
    source1_ids = ["S1-001", "S1-002", "S1-003"]

    result = build_ground_truth_dict(gt, source1_ids)
    assert result["S1-001"] == {"S2-001", "S3-001"}
    assert result["S1-002"] == {"S2-002"}
    assert result["S1-003"] == set()


def test_build_predictions_dict_parses_comma_separated():
    """
    Regression: build_predictions_dict itertuples path caused TypeError when
    .get() was called on a namedtuple.  Vectorized fix must parse correctly.
    """
    from business_entity_resolution.evaluation.metrics import build_predictions_dict

    df = pd.DataFrame({
        "source1_entity_id": ["S1-001", "S1-002", "S1-003"],
        "matched_entity_ids": ["S2-001,S3-001", "", "S2-002"],
    })
    result = build_predictions_dict(df)
    assert result["S1-001"] == {"S2-001", "S3-001"}
    assert result["S1-002"] == set()
    assert result["S1-003"] == {"S2-002"}


# ---------------------------------------------------------------------------
# Bug: _ADDRESS_STOPWORDS defined but never applied
# ---------------------------------------------------------------------------


def test_address_stopwords_are_applied():
    """
    Regression: normalize_address defined _ADDRESS_STOPWORDS but never filtered
    tokens with it.  After the fix, stopwords like 'near', 'the', 'of' are
    stripped from normalized address strings.
    """
    from business_entity_resolution.normalization.normalizer import normalize_address

    result = normalize_address("Near the City Hall, San Jose", "US")
    # "near" and "the" are in _ADDRESS_STOPWORDS — must be absent from raw_normalized
    tokens = result.raw_normalized.split()
    assert "near" not in tokens, (
        f"'near' (stopword) should be removed but got: {result.raw_normalized!r}"
    )
    assert "the" not in tokens, (
        f"'the' (stopword) should be removed but got: {result.raw_normalized!r}"
    )


# ---------------------------------------------------------------------------
# Bug: _make_predictions in threshold_tuning used iterrows
# ---------------------------------------------------------------------------


def test_make_predictions_vectorized_matches_old_behavior():
    """
    Regression: _make_predictions vectorized groupby must produce identical
    output to the old iterrows version.
    """
    from business_entity_resolution.decision.threshold_tuning import _make_predictions

    scored = pd.DataFrame({
        "source1_entity_id": ["S1-001", "S1-001", "S1-002", "S1-003"],
        "other_entity_id": ["S2-001", "S2-002", "S2-003", "S2-004"],
        "score": [0.8, 0.3, 0.6, 0.4],
    })
    s1_ids = ["S1-001", "S1-002", "S1-003", "S1-004"]

    result = _make_predictions(scored, s1_ids, threshold=0.5)

    assert result["S1-001"] == {"S2-001"}   # 0.8 >= 0.5; 0.3 < 0.5
    assert result["S1-002"] == {"S2-003"}   # 0.6 >= 0.5
    assert result["S1-003"] == set()        # 0.4 < 0.5
    assert result["S1-004"] == set()        # not in scored at all


def test_make_predictions_empty_input():
    """Vectorized _make_predictions must handle empty scored DataFrame."""
    from business_entity_resolution.decision.threshold_tuning import _make_predictions

    result = _make_predictions(
        pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "score"]),
        ["S1-001", "S1-002"],
        threshold=0.5,
    )
    assert result == {"S1-001": set(), "S1-002": set()}
