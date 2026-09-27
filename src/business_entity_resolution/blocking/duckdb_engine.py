import duckdb
import pandas as pd
import logging
from business_entity_resolution.blocking.engine import BlockingDiagnostics

logger = logging.getLogger(__name__)

def generate_candidates_duckdb(
    s1_norm: pd.DataFrame,
    ot_norm: pd.DataFrame,
    candidate_cap: int = 30
) -> tuple[pd.DataFrame, BlockingDiagnostics]:
    """
    DuckDB-based candidate generation matching Phase 2 specifications.
    Operates out-of-core and strictly within a 5GB memory limit.
    """
    import time
    t0 = time.time()
    
    # Initialize connection and memory limits
    con = duckdb.connect(database=':memory:') # Or disk-backed if necessary
    con.execute("PRAGMA memory_limit='5.0GB'")
    con.execute("PRAGMA threads=4")
    
    # Register dataframes
    con.register('s1', s1_norm)
    con.register('ot', ot_norm)
    
    combined_query = """
    WITH s1_tokens AS (
        SELECT 
            entity_id AS s1_id,
            country_norm AS s1_country,
            split_part(norm_name, ' ', 1) AS block_key
        FROM s1
        WHERE length(split_part(norm_name, ' ', 1)) >= 3
    ),
    ot_tokens AS (
        SELECT 
            entity_id AS ot_id,
            country_norm AS ot_country,
            split_part(norm_name, ' ', 1) AS block_key
        FROM ot
        WHERE length(split_part(norm_name, ' ', 1)) >= 3
    ),
    valid_blocks_primary AS (
        SELECT block_key, s1_country AS country
        FROM (
            SELECT block_key, s1_country FROM s1_tokens
            UNION ALL
            SELECT block_key, ot_country FROM ot_tokens
        )
        GROUP BY block_key, country
        HAVING count(*) < 25
    ),
    primary_pass AS (
        SELECT s1_tokens.s1_id AS source1_entity_id, ot_tokens.ot_id AS other_entity_id, 'primary_pass' AS strategy_name
        FROM s1_tokens
        JOIN ot_tokens 
          ON s1_tokens.block_key = ot_tokens.block_key 
          AND s1_tokens.s1_country = ot_tokens.ot_country
        JOIN valid_blocks_primary 
          ON s1_tokens.block_key = valid_blocks_primary.block_key 
          AND s1_tokens.s1_country = valid_blocks_primary.country
    ),
    s1_sec AS (
        SELECT 
            entity_id AS s1_id,
            substring(norm_name, 1, 5) AS name_prefix,
            regexp_replace(norm_address, '[^a-zA-Z0-9]', '', 'g') AS address_alphanum
        FROM s1
        WHERE length(norm_address) > 5 AND length(norm_name) >= 5
    ),
    ot_sec AS (
        SELECT 
            entity_id AS ot_id,
            substring(norm_name, 1, 5) AS name_prefix,
            regexp_replace(norm_address, '[^a-zA-Z0-9]', '', 'g') AS address_alphanum
        FROM ot
        WHERE length(norm_address) > 5 AND length(norm_name) >= 5
    ),
    valid_blocks_sec AS (
        SELECT name_prefix, address_alphanum
        FROM (
            SELECT name_prefix, address_alphanum FROM s1_sec
            UNION ALL
            SELECT name_prefix, address_alphanum FROM ot_sec
        )
        GROUP BY name_prefix, address_alphanum
        HAVING count(*) < 25
    ),
    secondary_pass AS (
        SELECT s1_sec.s1_id AS source1_entity_id, ot_sec.ot_id AS other_entity_id, 'secondary_pass' AS strategy_name
        FROM s1_sec
        JOIN ot_sec 
          ON s1_sec.name_prefix = ot_sec.name_prefix 
          AND s1_sec.address_alphanum = ot_sec.address_alphanum
        JOIN valid_blocks_sec
          ON s1_sec.name_prefix = valid_blocks_sec.name_prefix
          AND s1_sec.address_alphanum = valid_blocks_sec.address_alphanum
    )
    SELECT source1_entity_id, other_entity_id, any_value(strategy_name) AS strategy_name
    FROM (
        SELECT * FROM primary_pass
        UNION ALL
        SELECT * FROM secondary_pass
    )
    GROUP BY source1_entity_id, other_entity_id
    """
    
    logger.info("Executing DuckDB candidate generation...")
    candidates = con.execute(combined_query).df()
    
    # Apply candidate cap
    if candidate_cap > 0:
        counts = candidates.groupby("source1_entity_id").cumcount()
        candidates = candidates[counts < candidate_cap].copy()
        
    diag = BlockingDiagnostics()
    diag.elapsed_seconds = time.time() - t0
    diag.total_candidates_after_cap = len(candidates)
    
    return candidates, diag
