"""
config.py — Typed configuration loader for the Business Entity Resolution pipeline.

Single source of truth: every path, threshold, hyperparameter, and tunable constant
is defined here (or read from the YAML file). No magic numbers elsewhere in the codebase.

Override precedence: environment variables > YAML file > code defaults.

Environment variables:
    BER_DATA_DIR       — overrides paths.data_dir
    BER_OUTPUT_DIR     — overrides paths.output_dir
    BER_MODEL_DIR      — overrides paths.model_dir
    BER_USE_DENSE      — "1" or "true" to enable dense embedding blocking
    BER_FAST_MODE      — "1" to skip dense embedding (alias for BER_USE_DENSE=0)
    BER_LOG_LEVEL      — logging level string (DEBUG, INFO, WARNING, ERROR)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PathsConfig:
    data_dir: str = "data"
    output_dir: str = "output"
    model_dir: str = "models"
    train_subdir: str = "train"
    test_subdir: str = "test"
    source1_file: str = "source1.tsv"
    source2_file: str = "source2.tsv"
    source3_file: str = "source3.tsv"
    ground_truth_file: str = "ground_truth.tsv"
    candidate_pairs_file: str = "candidate_pairs.tsv"
    matching_results_file: str = "matching_results.tsv"
    model_artifact_file: str = "model.pkl"
    config_artifact_file: str = "tuned_config.yaml"

    # Derived helpers — not in YAML, computed on post_init
    def train_dir(self) -> Path:
        return Path(self.data_dir) / self.train_subdir

    def test_dir(self) -> Path:
        return Path(self.data_dir) / self.test_subdir

    def output_path(self) -> Path:
        return Path(self.output_dir)

    def model_path(self) -> Path:
        return Path(self.model_dir)


@dataclass
class IOConfig:
    chunk_size: int = 100_000
    source_encoding: str = "utf-8"
    delimiter: str = "\t"


@dataclass
class NormalizationConfig:
    name_max_tokens: int = 8
    name_prefix_len: int = 5
    name_suffix_len: int = 5
    address_prefix_len: int = 6


@dataclass
class NamePrefixConfig:
    prefix_len: int = 5


@dataclass
class NameSuffixConfig:
    suffix_len: int = 5


@dataclass
class AddressNumberConfig:
    min_token_len: int = 1


@dataclass
class AddressPrefixConfig:
    prefix_len: int = 6


@dataclass
class RareNumericConfig:
    max_frequency: int = 50


@dataclass
class TFIDFConfig:
    top_k: int = 20
    min_similarity: float = 0.10
    max_features: int = 200_000
    ngram_range: tuple[int, int] = (1, 2)
    analyzer: str = "char_wb"


@dataclass
class DenseEmbeddingConfig:
    top_k: int = 20
    min_similarity: float = 0.60
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 512


@dataclass
class BlockingConfig:
    active_strategies: list[str] = field(default_factory=lambda: [
        "name_prefix", "name_suffix", "address_number",
        "address_prefix", "rare_numeric_token", "tfidf_nn",
    ])
    use_dense_embedding: bool = False
    candidate_cap: int = 30
    max_block_size: int = 500
    min_key_length: int = 2
    name_prefix: NamePrefixConfig = field(default_factory=NamePrefixConfig)
    name_suffix: NameSuffixConfig = field(default_factory=NameSuffixConfig)
    address_number: AddressNumberConfig = field(default_factory=AddressNumberConfig)
    address_prefix: AddressPrefixConfig = field(default_factory=AddressPrefixConfig)
    rare_numeric_token: RareNumericConfig = field(default_factory=RareNumericConfig)
    tfidf_nn: TFIDFConfig = field(default_factory=TFIDFConfig)
    dense_embedding_nn: DenseEmbeddingConfig = field(default_factory=DenseEmbeddingConfig)


@dataclass
class FeaturesConfig:
    name_edit_low_threshold: float = 0.5
    address_edit_low_threshold: float = 0.4


@dataclass
class TrainingConfig:
    validation_fraction: float = 0.15
    negative_positive_ratio: int = 5
    hard_negatives_only: bool = True


@dataclass
class LightGBMConfig:
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    n_jobs: int = -1
    random_state: int = 42
    verbose: int = -1


@dataclass
class XGBoostConfig:
    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    n_jobs: int = -1
    random_state: int = 42
    verbosity: int = 0
    eval_metric: str = "logloss"


@dataclass
class EnsembleConfig:
    weights: list[float] = field(default_factory=lambda: [0.6, 0.4])


@dataclass
class ModelsConfig:
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    active_model: str = "ensemble"


@dataclass
class DecisionConfig:
    threshold_grid: list[float] = field(
        default_factory=lambda: [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    )
    default_threshold: float = 0.5
    per_country: bool = False
    min_country_val_size: int = 100


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


@dataclass
class Config:
    """
    Root configuration object.  Loaded from YAML, then patched by env vars.
    Every pipeline module receives a ``Config`` instance; no module reads env
    vars or YAML files directly — they just read attributes off this object.
    """

    paths: PathsConfig = field(default_factory=PathsConfig)
    io: IOConfig = field(default_factory=IOConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning the merged dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _apply_dataclass(dc_type: type, data: dict[str, Any]) -> Any:
    """
    Recursively construct a dataclass from a (possibly nested) dict,
    ignoring unknown keys to stay forward-compatible with YAML additions.

    ``from __future__ import annotations`` turns all field annotations into
    strings at module load time, so we MUST resolve them at call time via
    ``typing.get_type_hints()``; otherwise ``dataclasses.is_dataclass(fld.type)``
    always receives a ``str`` and never fires.
    """
    import dataclasses
    import typing

    if not dataclasses.is_dataclass(dc_type):
        return data

    # Resolve forward-reference strings → actual type objects.
    # Provide the module's globals so that locally defined dataclass names resolve.
    try:
        import sys
        module = sys.modules.get(dc_type.__module__, None)
        globalns = vars(module) if module is not None else {}
        resolved_hints = typing.get_type_hints(dc_type, globalns=globalns)
    except Exception:
        resolved_hints = {}

    fields = {f.name: f for f in dataclasses.fields(dc_type)}
    kwargs: dict[str, Any] = {}

    for name, fld in fields.items():
        if name not in data:
            continue
        value = data[name]
        # Use the resolved type object, fall back to the raw (string) annotation.
        ftype = resolved_hints.get(name, fld.type)
        try:
            if isinstance(value, dict) and dataclasses.is_dataclass(ftype):
                kwargs[name] = _apply_dataclass(ftype, value)
            elif isinstance(value, list) and name in ("ngram_range",):
                kwargs[name] = tuple(value)
            else:
                kwargs[name] = value
        except Exception:
            kwargs[name] = value

    return dc_type(**kwargs)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply BER_* environment variable overrides to raw config dict."""
    overrides: dict[str, Any] = {}

    if (val := os.getenv("BER_DATA_DIR")):
        overrides.setdefault("paths", {})["data_dir"] = val
    if (val := os.getenv("BER_OUTPUT_DIR")):
        overrides.setdefault("paths", {})["output_dir"] = val
    if (val := os.getenv("BER_MODEL_DIR")):
        overrides.setdefault("paths", {})["model_dir"] = val
    if (val := os.getenv("BER_LOG_LEVEL")):
        overrides.setdefault("logging", {})["level"] = val

    # BER_FAST_MODE=1 disables dense embedding; BER_USE_DENSE=1 enables it
    fast_mode = os.getenv("BER_FAST_MODE", "").strip().lower() in ("1", "true", "yes")
    use_dense = os.getenv("BER_USE_DENSE", "").strip().lower() in ("1", "true", "yes")
    if fast_mode:
        overrides.setdefault("blocking", {})["use_dense_embedding"] = False
    elif use_dense:
        overrides.setdefault("blocking", {})["use_dense_embedding"] = True

    return _deep_update(data, overrides) if overrides else data


def load_config(yaml_path: Optional[str | Path] = None) -> Config:
    """
    Load and return a fully-resolved ``Config`` object.

    Parameters
    ----------
    yaml_path:
        Path to a YAML config file. If None, looks for
        ``configs/default.yaml`` relative to the current working directory,
        then relative to this file's parent (package root).  Falls back to
        code defaults if no file is found.

    Returns
    -------
    Config
        Fully-populated, immutable-by-convention configuration object.
    """
    raw: dict[str, Any] = {}

    search_paths: list[Path] = []
    if yaml_path is not None:
        search_paths = [Path(yaml_path)]
    else:
        search_paths = [
            Path.cwd() / "configs" / "default.yaml",
            Path(__file__).parent.parent.parent / "configs" / "default.yaml",
        ]

    for candidate in search_paths:
        if candidate.exists():
            raw = _load_yaml(candidate)
            break

    raw = _apply_env_overrides(raw)

    # Build nested dataclasses
    cfg = Config(
        paths=_apply_dataclass(PathsConfig, raw.get("paths", {})),
        io=_apply_dataclass(IOConfig, raw.get("io", {})),
        normalization=_apply_dataclass(NormalizationConfig, raw.get("normalization", {})),
        features=_apply_dataclass(FeaturesConfig, raw.get("features", {})),
        training=_apply_dataclass(TrainingConfig, raw.get("training", {})),
        decision=_apply_dataclass(DecisionConfig, raw.get("decision", {})),
        logging=_apply_dataclass(LoggingConfig, raw.get("logging", {})),
        random_seed=raw.get("random_seed", 42),
    )

    # Blocking requires special nested construction
    blk_raw = raw.get("blocking", {})
    cfg.blocking = BlockingConfig(
        active_strategies=blk_raw.get("active_strategies", [
            "name_prefix", "name_suffix", "address_number",
            "address_prefix", "rare_numeric_token", "tfidf_nn",
        ]),
        use_dense_embedding=blk_raw.get("use_dense_embedding", False),
        candidate_cap=blk_raw.get("candidate_cap", 30),
        max_block_size=blk_raw.get("max_block_size", 500),
        min_key_length=blk_raw.get("min_key_length", 2),
        name_prefix=_apply_dataclass(NamePrefixConfig, blk_raw.get("name_prefix", {})),
        name_suffix=_apply_dataclass(NameSuffixConfig, blk_raw.get("name_suffix", {})),
        address_number=_apply_dataclass(AddressNumberConfig, blk_raw.get("address_number", {})),
        address_prefix=_apply_dataclass(AddressPrefixConfig, blk_raw.get("address_prefix", {})),
        rare_numeric_token=_apply_dataclass(RareNumericConfig, blk_raw.get("rare_numeric_token", {})),
        tfidf_nn=_apply_dataclass(TFIDFConfig, blk_raw.get("tfidf_nn", {})),
        dense_embedding_nn=_apply_dataclass(DenseEmbeddingConfig, blk_raw.get("dense_embedding_nn", {})),
    )

    # Models
    mdl_raw = raw.get("models", {})
    cfg.models = ModelsConfig(
        lightgbm=_apply_dataclass(LightGBMConfig, mdl_raw.get("lightgbm", {})),
        xgboost=_apply_dataclass(XGBoostConfig, mdl_raw.get("xgboost", {})),
        ensemble=_apply_dataclass(EnsembleConfig, mdl_raw.get("ensemble", {})),
        active_model=mdl_raw.get("active_model", "ensemble"),
    )

    return cfg
