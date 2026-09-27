"""
cli.py — Single CLI entry point with subcommands: train / predict / validate / package.

Usage:
    py -m business_entity_resolution train --config configs/default.yaml
    py -m business_entity_resolution predict --config configs/default.yaml
    py -m business_entity_resolution validate --output-dir output/
    py -m business_entity_resolution package --output-dir output/

Or via the installed 'ber' script:
    ber train --config configs/default.yaml
"""

from __future__ import annotations

import sys

import click

from business_entity_resolution.config import load_config


@click.group()
@click.version_option(package_name="business-entity-resolution")
def main() -> None:
    """Business Entity Resolution pipeline CLI."""


@main.command()
@click.option("--config", default=None, help="Path to YAML config file.")
def train(config: str | None) -> None:
    """Train the matching model end-to-end."""
    from business_entity_resolution.pipeline.train import train as _train
    cfg = load_config(config)
    _train(cfg=cfg, config_path=config)


@main.command()
@click.option("--config", default=None, help="Path to YAML config file.")
def predict(config: str | None) -> None:
    """Run inference and write output files."""
    from business_entity_resolution.pipeline.predict import predict as _predict
    cfg = load_config(config)
    _predict(cfg=cfg, config_path=config)


@main.command()
@click.option(
    "--output-dir", default="output",
    help="Directory containing output TSV files (matching_results.tsv, candidate_pairs.tsv).",
)
@click.option(
    "--test-dir", default="dataset/test",
    help="Folder containing test_source1/2/3.tsv (default: dataset/test).",
)
@click.option(
    "--check-ids", is_flag=True, default=False,
    help="Also verify matched/candidate IDs exist in test Source-2/3 files "
         "(off by default; memory-intensive on the full test set).",
)
def validate(output_dir: str, test_dir: str, check_ids: bool) -> None:
    """Validate the submission output files against the official ML Challenge rules.

    Reads matching_results.tsv (required) and candidate_pairs.tsv (optional) from
    OUTPUT_DIR, and validates them against the test source files in TEST_DIR.
    Exit code 0 = safe to submit; 1 = fix the listed issues.
    """
    import os
    from pathlib import Path

    try:
        from utils.validate_submission import validate as _validate_fn
    except ImportError as exc:
        click.echo(
            f"ERROR: Could not import validate() from utils/validate_submission.py: {exc}",
            err=True,
        )
        sys.exit(1)

    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    click.echo("ML Challenge 2026 - submission validator")
    click.echo(f"  output dir : {output_dir}")
    click.echo(f"  test dir   : {test_dir}")

    try:
        errors, warnings = _validate_fn(
            matching_path=matching_path,
            candidate_path=candidate_path,
            test_dir=test_dir,
            check_ids=check_ids,
        )
    except UnicodeDecodeError:
        click.echo(
            "\nFAIL - 1 issue(s) to fix before submitting:\n"
            "  1. A file is not valid UTF-8. Re-save as plain UTF-8 tab-separated .tsv.",
            err=True,
        )
        sys.exit(1)
    except OSError as exc:
        click.echo(f"\nFAIL - 1 issue(s):\n  1. Could not read a file: {exc}.", err=True)
        sys.exit(1)

    click.echo()
    for warning in warnings:
        click.echo(f"WARNING: {warning}")

    if errors:
        click.echo(f"FAIL - {len(errors)} issue(s) to fix before submitting:")
        for i, error in enumerate(errors, 1):
            click.echo(f"  {i}. {error}")
        sys.exit(1)

    click.echo("PASS - no blocking issues found. Safe to submit.")
    sys.exit(0)


@main.command()
@click.option("--output-dir", default="output", help="Directory containing output TSV files.")
@click.option("--zip-name", default="submission.zip", help="Name for the output zip file.")
@click.option(
    "--test-dir", default="dataset/test",
    help="Folder with test source files, passed through to the validator.",
)
@click.option(
    "--check-ids", is_flag=True, default=False,
    help="Enable the optional ID-existence check during pre-packaging validation.",
)
def package(output_dir: str, zip_name: str, test_dir: str, check_ids: bool) -> None:
    """Package the submission into a zip file after validation."""
    import subprocess
    cmd = [
        sys.executable, "scripts/package_submission.py",
        "--output-dir", output_dir,
        "--zip-name", zip_name,
        "--test-dir", test_dir,
    ]
    if check_ids:
        cmd.append("--check-ids")
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
