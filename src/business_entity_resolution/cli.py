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
@click.option("--output-dir", default="output", help="Directory containing output TSV files.")
@click.option("--source1-ids", default=None, help="Path to source1 TSV (for ID completeness check).")
def validate(output_dir: str, source1_ids: str | None) -> None:
    """Validate the submission output files."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "utils"))
    try:
        from validate_submission import validate_submission
        success = validate_submission(output_dir=output_dir, source1_ids_file=source1_ids)
        sys.exit(0 if success else 1)
    except ImportError:
        click.echo("Running validate_submission.py directly...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "utils/validate_submission.py",
             "--output-dir", output_dir],
            check=False,
        )
        sys.exit(result.returncode)


@main.command()
@click.option("--output-dir", default="output", help="Directory containing output TSV files.")
@click.option("--zip-name", default="submission.zip", help="Name for the output zip file.")
def package(output_dir: str, zip_name: str) -> None:
    """Package the submission into a zip file after validation."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/package_submission.py",
         "--output-dir", output_dir, "--zip-name", zip_name],
        check=False,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
