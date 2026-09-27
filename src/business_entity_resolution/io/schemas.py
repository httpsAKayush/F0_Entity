"""
io/schemas.py — Explicit column definitions for all input file types.

Adding a new file type: define a new SchemaSpec instance here and pass it to
``validate_dataframe`` in loaders.py.  No other module needs to change.

Column name decisions (documented here per spec requirements):
- All source files (source1/2/3) share the same four columns:
    entity_id, business_name, business_address, country
- Ground truth file columns:
    source1_entity_id, source_id, source_tag
  where source_tag is either "source2" or "source3".
  We use a long/tidy format (one row per match) rather than a wide format with
  comma-separated ids in a single cell, because it is easier to join and group
  without string parsing at the critical path.
- Output file matching_results.tsv columns (official ML Challenge 2026 format):
    source1_entity_id, matched_entity_ids
  where matched_entity_ids holds a comma-separated string of all S2- and S3-
  matched IDs combined, or an empty string for singletons.
- Output file candidate_pairs.tsv columns (official ML Challenge 2026 format):
    source1_entity_id, candidate_entity_ids
  where candidate_entity_ids holds a comma-separated string of all candidate IDs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a single column: name and whether nulls are forbidden."""
    name: str
    required: bool = True   # if True, column must have zero nulls
    unique: bool = False    # if True, column must have no duplicate values


@dataclass(frozen=True)
class SchemaSpec:
    """Specification for an entire file's expected schema."""
    file_label: str          # human-readable name used in error messages
    columns: tuple[ColumnSpec, ...]

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def required_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.required]


# ---------------------------------------------------------------------------
# Source file schemas (shared across source1, source2, source3)
# ---------------------------------------------------------------------------

SOURCE_SCHEMA = SchemaSpec(
    file_label="source file",
    columns=(
        ColumnSpec("entity_id", required=True),
        ColumnSpec("business_name", required=True),
        ColumnSpec("business_address", required=True),
        ColumnSpec("country", required=True),
    ),
)

SOURCE1_SCHEMA = SchemaSpec(
    file_label="source1",
    columns=SOURCE_SCHEMA.columns,
)

SOURCE2_SCHEMA = SchemaSpec(
    file_label="source2",
    columns=SOURCE_SCHEMA.columns,
)

SOURCE3_SCHEMA = SchemaSpec(
    file_label="source3",
    columns=SOURCE_SCHEMA.columns,
)

# ---------------------------------------------------------------------------
# Ground truth schema
# ---------------------------------------------------------------------------
# Tidy/long format: one row per (source1_entity_id, matched entity).
# source_tag is "source2" or "source3".
# source1 entities with zero matches do NOT appear in this file at all
# (they are singletons) — the pipeline must handle their absence gracefully.

GROUND_TRUTH_SCHEMA = SchemaSpec(
    file_label="ground_truth",
    columns=(
        ColumnSpec("source1_entity_id", required=True),
        ColumnSpec("source_id", required=True),
        ColumnSpec("source_tag", required=True),
    ),
)

# ---------------------------------------------------------------------------
# Output file schemas
# ---------------------------------------------------------------------------

MATCHING_RESULTS_SCHEMA = SchemaSpec(
    file_label="matching_results",
    columns=(
        # Official ML Challenge 2026 format: one row per S1 entity, all S2+S3
        # matches combined into a single comma-separated column.
        ColumnSpec("source1_entity_id", required=True, unique=True),
        ColumnSpec("matched_entity_ids", required=False),
    ),
)

CANDIDATE_PAIRS_SCHEMA = SchemaSpec(
    file_label="candidate_pairs",
    columns=(
        # Official ML Challenge 2026 format: one row per S1 entity, all S2+S3
        # candidates combined into a single comma-separated column.
        ColumnSpec("source1_entity_id", required=True, unique=True),
        ColumnSpec("candidate_entity_ids", required=False),
    ),
)
