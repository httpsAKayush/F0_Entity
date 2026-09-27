"""
scripts/package_submission.py — Assembles and validates the final submission zip.

Layout inside the zip:
    output/matching_results.tsv
    output/candidate_pairs.tsv
    code/                          (full source package)
    Documentation_template.md

Validation is run BEFORE finalizing the zip — a failing validation refuses to package.

Usage:
    py scripts/package_submission.py --output-dir output/ --zip-name submission.zip
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def package_submission(
    output_dir: str = "output",
    zip_name: str = "submission.zip",
    project_root: str | None = None,
) -> bool:
    """
    Validate output files and package them into a submission zip.

    Returns True on success, False on failure.
    """
    root = Path(project_root) if project_root else Path.cwd()
    out_path = Path(output_dir)
    zip_path = root / zip_name

    # Run validation first
    print("Running pre-packaging validation...")
    sys.path.insert(0, str(root / "utils"))
    try:
        from validate_submission import validate_submission
        success = validate_submission(output_dir=str(out_path))
    except ImportError as exc:
        print(f"ERROR: Could not import validate_submission: {exc}")
        return False

    if not success:
        print("❌ Packaging refused: fix validation errors first.")
        return False

    # Build the zip
    print(f"\nPackaging submission to: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Output files
        for filename in ["matching_results.tsv", "candidate_pairs.tsv"]:
            src = out_path / filename
            if src.exists():
                zf.write(src, arcname=f"output/{filename}")
                print(f"  Added: output/{filename}")
            else:
                print(f"  WARNING: {src} not found, skipping.")

        # Documentation
        doc_path = root / "Documentation_template.md"
        if doc_path.exists():
            zf.write(doc_path, arcname="Documentation_template.md")
            print("  Added: Documentation_template.md")

        # Source code
        src_dir = root / "src"
        if src_dir.exists():
            for fpath in src_dir.rglob("*.py"):
                arcname = "code/" + str(fpath.relative_to(root)).replace("\\", "/")
                zf.write(fpath, arcname=arcname)
            print(f"  Added: code/ ({sum(1 for _ in src_dir.rglob('*.py'))} Python files)")

        # Include pyproject.toml and README
        for extra in ["pyproject.toml", "README.md", "configs/default.yaml"]:
            extra_path = root / extra
            if extra_path.exists():
                zf.write(extra_path, arcname=f"code/{extra}")
                print(f"  Added: code/{extra}")

    print(f"\n✅ Submission packaged: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the submission zip.")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--zip-name", default="submission.zip")
    args = parser.parse_args()
    success = package_submission(output_dir=args.output_dir, zip_name=args.zip_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
