"""
io/loaders.py — Schema-validated, chunk-capable readers for all TSV data files.

Design notes:
- Every loader validates columns and required-field nullness immediately on
  read, raising SchemaError with an actionable message before any bad data
  can propagate into the pipeline.
- Large source files (multi-GB) are supported via iter_chunks(), which yields
  pandas DataFrames of ``chunk_size`` rows to cap peak memory usage.
- We use pandas here (not Polars) because: (a) the schema-validation logic is
  simpler with pandas.read_csv's chunksize iterator, (b) the source files are
  read once and then processed in downstream Polars stages, and (c) chunk-level
  validation is cleaner to express with pandas iterrows patterns.
- Polars is used in the heavy blocking/feature stages where columnar throughput
  matters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional, Union

import pandas as pd

from business_entity_resolution.io.schemas import (
    GROUND_TRUTH_SCHEMA,
    MATCHING_RESULTS_SCHEMA,
    SOURCE1_SCHEMA,
    SOURCE2_SCHEMA,
    SOURCE3_SCHEMA,
    SchemaSpec,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class SchemaError(ValueError):
    """Raised when an input file does not conform to its expected schema."""


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------


def validate_dataframe(df: pd.DataFrame, schema: SchemaSpec, filepath: str = "") -> None:
    """
    Validate that *df* conforms to *schema*.

    Raises
    ------
    SchemaError
        If required columns are missing, required columns contain nulls, or
        unique columns contain duplicate values.
        The error message names the file, the column, and the problem.
    """
    label = f"[{filepath}] " if filepath else ""

    # 1. Column presence
    missing = [col for col in schema.column_names() if col not in df.columns]
    if missing:
        raise SchemaError(
            f"{label}File '{schema.file_label}' is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # 2. Null checks on required columns
    for col in schema.required_columns():
        null_count = df[col].isna().sum()
        if null_count > 0:
            raise SchemaError(
                f"{label}Column '{col}' in '{schema.file_label}' has {null_count:,} "
                f"unexpected null values. All rows in this column must be non-null."
            )

    # 3. Uniqueness checks
    for spec in schema.columns:
        if spec.unique and spec.name in df.columns:
            dup_count = df[spec.name].duplicated().sum()
            if dup_count > 0:
                dupes = df.loc[df[spec.name].duplicated(keep=False), spec.name].unique()[:5]
                raise SchemaError(
                    f"{label}Column '{spec.name}' in '{schema.file_label}' has "
                    f"{dup_count:,} duplicate value(s) — entity_id must be unique. "
                    f"Example duplicates: {list(dupes)}"
                )


# ---------------------------------------------------------------------------
# Full-file loaders (for files that comfortably fit in RAM)
# ---------------------------------------------------------------------------


def _read_tsv(
    filepath: Union[str, Path],
    schema: SchemaSpec,
    encoding: str = "utf-8",
    delimiter: str = "\t",
    dtype: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Read a TSV file, validate schema, and return a DataFrame."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Expected file not found: {filepath}. "
            "Check your config paths and ensure the data directory is set correctly."
        )
    logger.info("Loading %s from %s", schema.file_label, filepath)
    if dtype is None:
        dtype = {col: str for col in schema.column_names()}

    df = pd.read_csv(
        filepath,
        sep=delimiter,
        dtype=dtype,
        encoding=encoding,
        keep_default_na=False,      # treat empty strings as empty strings, not NaN
        na_values=["\\N", "NULL"],  # only explicit NULL markers become NaN
    )
    # Strip leading/trailing whitespace from string columns
    for col in schema.column_names():
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.strip()

    validate_dataframe(df, schema, str(filepath))
    logger.info("Loaded %d rows from %s", len(df), schema.file_label)
    return df


def load_source1(
    filepath: Union[str, Path], encoding: str = "utf-8", delimiter: str = "\t"
) -> pd.DataFrame:
    """Load and validate source1 (reference entities) TSV file."""
    return _read_tsv(Path(filepath), SOURCE1_SCHEMA, encoding=encoding, delimiter=delimiter)


def load_source2(
    filepath: Union[str, Path], encoding: str = "utf-8", delimiter: str = "\t"
) -> pd.DataFrame:
    """Load and validate source2 (vendor-A) TSV file."""
    return _read_tsv(Path(filepath), SOURCE2_SCHEMA, encoding=encoding, delimiter=delimiter)


def load_source3(
    filepath: Union[str, Path], encoding: str = "utf-8", delimiter: str = "\t"
) -> pd.DataFrame:
    """Load and validate source3 (vendor-B) TSV file."""
    return _read_tsv(Path(filepath), SOURCE3_SCHEMA, encoding=encoding, delimiter=delimiter)


def load_ground_truth(
    filepath: Union[str, Path], encoding: str = "utf-8", delimiter: str = "\t"
) -> pd.DataFrame:
    """
    Load and validate ground truth TSV file.

    Returns a DataFrame with columns:
        source1_entity_id, source_id, source_tag

    source_tag is expected to be "source2" or "source3".
    """
    return _read_tsv(Path(filepath), GROUND_TRUTH_SCHEMA, encoding=encoding, delimiter=delimiter)


def load_matching_results(
    filepath: Union[str, Path], encoding: str = "utf-8", delimiter: str = "\t"
) -> pd.DataFrame:
    """Load and validate a matching_results.tsv output file (for validation)."""
    return _read_tsv(Path(filepath), MATCHING_RESULTS_SCHEMA, encoding=encoding, delimiter=delimiter)


def load_candidate_pairs(
    filepath: Union[str, Path], encoding: str = "utf-8", delimiter: str = "\t"
) -> pd.DataFrame:
    """Load and validate a candidate_pairs.tsv output file."""
    from business_entity_resolution.io.schemas import CANDIDATE_PAIRS_SCHEMA
    return _read_tsv(Path(filepath), CANDIDATE_PAIRS_SCHEMA, encoding=encoding, delimiter=delimiter)

# ---------------------------------------------------------------------------
# Chunked / streaming loaders (for large source files that may not fit in RAM)
# ---------------------------------------------------------------------------


def iter_source_chunks(
    filepath: Union[str, Path],
    schema: SchemaSpec,
    chunk_size: int = 100_000,
    encoding: str = "utf-8",
    delimiter: str = "\t",
) -> Iterator[pd.DataFrame]:
    """
    Yield DataFrames of *chunk_size* rows from a large source TSV file.

    Each chunk is schema-validated before being yielded.  This allows the
    caller to process data in a streaming fashion without loading the entire
    file into RAM.

    For columns marked ``unique=True`` in the schema, cross-chunk uniqueness
    is enforced by tracking every seen value in memory.  This raises a
    SchemaError on the first duplicate found across chunks, rather than
    silently allowing it to propagate into the pipeline.

    Parameters
    ----------
    filepath:
        Path to the TSV file.
    schema:
        SchemaSpec to validate against.
    chunk_size:
        Number of rows per chunk.
    encoding:
        File encoding.
    delimiter:
        Column delimiter character.

    Yields
    ------
    pd.DataFrame
        A chunk of rows, validated and stripped.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Expected file not found: {filepath}")

    dtype = {col: str for col in schema.column_names()}
    chunk_reader = pd.read_csv(
        filepath,
        sep=delimiter,
        dtype=dtype,
        encoding=encoding,
        keep_default_na=False,
        na_values=["\\N", "NULL"],
        chunksize=chunk_size,
    )

    # Track values for cross-chunk uniqueness checks
    unique_cols = [spec.name for spec in schema.columns if spec.unique]
    seen_values: dict[str, set] = {col: set() for col in unique_cols}

    chunk_idx = 0
    for chunk in chunk_reader:
        # Strip whitespace
        for col in schema.column_names():
            if col in chunk.columns and chunk[col].dtype == object:
                chunk[col] = chunk[col].str.strip()

        if chunk_idx == 0:
            # Full schema validation on first chunk (column presence + nulls + within-chunk dupes)
            validate_dataframe(chunk, schema, str(filepath))
        else:
            # Subsequent chunks: only null checks (columns already confirmed)
            for col in schema.required_columns():
                null_count = chunk[col].isna().sum()
                if null_count > 0:
                    raise SchemaError(
                        f"[{filepath}] Column '{col}' in chunk {chunk_idx} has "
                        f"{null_count:,} unexpected null values."
                    )

        # Cross-chunk uniqueness tracking
        for col in unique_cols:
            if col not in chunk.columns:
                continue
            col_vals = chunk[col]
            cross_dupes = col_vals[col_vals.isin(seen_values[col])]
            if not cross_dupes.empty:
                raise SchemaError(
                    f"[{filepath}] Column '{col}' contains duplicate values across chunks "
                    f"(entity_id must be globally unique). "
                    f"Examples: {list(cross_dupes.unique()[:5])}"
                )
            seen_values[col].update(col_vals.tolist())

        chunk_idx += 1
        yield chunk


def load_source_full_or_chunked(
    filepath: Union[str, Path],
    schema: SchemaSpec,
    chunk_size: Optional[int] = None,
    encoding: str = "utf-8",
    delimiter: str = "\t",
) -> pd.DataFrame:
    """
    Load a source file either in full or via chunks, then concatenate.

    When *chunk_size* is None, reads in a single ``pd.read_csv`` call.
    This convenience wrapper lets callers not care about file size — they
    always receive a single DataFrame.
    """
    filepath = Path(filepath)
    if chunk_size is None:
        return _read_tsv(filepath, schema, encoding=encoding, delimiter=delimiter)

    chunks = list(iter_source_chunks(filepath, schema, chunk_size, encoding, delimiter))
    if not chunks:
        raise SchemaError(f"File appears to be empty: {filepath}")
    return pd.concat(chunks, ignore_index=True)
