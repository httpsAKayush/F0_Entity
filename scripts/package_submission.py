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
    py scripts/package_submission.py --output-dir output/ --test-dir dataset/test --check-ids
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
    test_dir: str = "dataset/test",
    check_ids: bool = False,
) -> bool:
    """
    Validate output files and package them into a submission zip.

    Parameters
    ----------
    output_dir:
        Directory containing matching_results.tsv and candidate_pairs.tsv.
    zip_name:
        Output zip filename (relative to project_root).
    project_root:
        Root of the project; defaults to the current working directory.
    test_dir:
        Directory containing test_source1/2/3.tsv, forwarded to the validator.
    check_ids:
        If True, the validator also checks that every matched/candidate ID
        exists in the test Source-2/3 files (memory-intensive; off by default).

    Returns
    -------
    bool
        True on success, False on failure.
    """
    root = Path(project_root) if project_root else Path.cwd()
    out_path = Path(output_dir)
    zip_path = root / zip_name

    # Locate utils/validate_submission.py and import the real validate() API
    utils_dir = str(root / "utils")
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)

    print("Running pre-packaging validation...")
    try:
        from validate_submission import validate  # real function from utils/
    except ImportError as exc:
        print(f"ERROR: Could not import validate() from utils/validate_submission.py: {exc}")
        return False

    matching_path = str(out_path / "matching_results.tsv")
    candidate_path = str(out_path / "candidate_pairs.tsv")

    try:
        errors, warnings = validate(
            matching_path=matching_path,
            candidate_path=candidate_path,
            test_dir=test_dir,
            check_ids=check_ids,
        )
    except Exception as exc:
        print(f"ERROR: Validation raised an exception: {exc}")
        return False

    for warning in warnings:
        print(f"WARNING: {warning}")

    if errors:
        print(f"\nFAIL - {len(errors)} issue(s) to fix before submitting:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        print("\n\u274c Packaging refused: fix validation errors first.")
        return False

    print("PASS - validation succeeded.")

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
        else:
            raise FileNotFoundError(f"CRITICAL: {doc_path.name} is missing. Submission must include documentation.")

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

    print(f"\n\u2705 Submission packaged: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the submission zip.")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument("--zip-name", default="submission.zip", help="Zip filename (default: submission.zip)")
    parser.add_argument(
        "--test-dir", default="dataset/test",
        help="Folder with test source files for the validator (default: dataset/test)",
    )
    parser.add_argument(
        "--check-ids", action="store_true",
        help="Enable ID-existence check in the validator (off by default; memory-intensive)",
    )
    args = parser.parse_args()
    success = package_submission(
        output_dir=args.output_dir,
        zip_name=args.zip_name,
        test_dir=args.test_dir,
        check_ids=args.check_ids,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
