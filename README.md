# Amazon ML Challenge 2026 - Entity Resolution

This repository contains the complete pipeline for the Amazon ML Challenge 2026 Entity Resolution track.
The pipeline is designed to strictly adhere to the 8GB System RAM and 4GB VRAM constraint by leveraging out-of-core DuckDB for multi-pass blocking, C-optimized fuzzy matching with RapidFuzz, and CPU-based LightGBM for macro-F0.5 precision-biased arbitration.

## Execution Runbook

Reviewers should run the following commands sequentially from the root of the repository to reproduce the submission end-to-end:

```bash
# 1. Activate Environment & Install Dependencies
conda activate FO_Entity
pip install -e .
pip install duckdb polars pandas rapidfuzz lightgbm xgboost pyarrow pydantic pyyaml click

# 2. Run Local Unit Tests & Regressions (Verifies Schema Fixes & CLI paths)
pytest tests/ -v

# 3. Train Pipeline (Ingests data via DuckDB, blocks, subsamples 10% for LightGBM)
ber train --config configs/default.yaml

# 4. Run Test Inference (Generates matching_results.tsv out-of-core in chunks)
ber predict --config configs/default.yaml

# 5. Execute Strict Validation (Must exit with code 0)
ber validate \
    --output-dir output \
    --test-dir dataset/test

# 6. Package Final Submission Bundle
python scripts/package_submission.py
```
