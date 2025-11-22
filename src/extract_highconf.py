import pandas as pd
import yaml
import argparse
import logging
import sys
from pathlib import Path

# Set up logging to track pipeline progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def identify_columns(df, criteria):
    """
    Finds columns in the dataframe that match the config criteria
    (exact match, starts with, or ends with).
    """
    found_cols = set()
    if not criteria: 
        return []
    
    # 1. Exact Matches
    if 'exact' in criteria and criteria['exact']:
        for col in criteria['exact']:
            if col in df.columns: found_cols.add(col)

    # 2. Prefix Matches
    if 'startswith' in criteria and criteria['startswith']:
        for prefix in criteria['startswith']:
            found_cols.update([c for c in df.columns if c.startswith(prefix)])

    # 3. Suffix Matches
    if 'endswith' in criteria and criteria['endswith']:
        for suffix in criteria['endswith']:
            found_cols.update([c for c in df.columns if c.endswith(suffix)])
            
    return list(found_cols)

def process_tcr_data(input_file, output_file, config):
    logger.info(f"Reading input file: {input_file}")
    try:
        sep = ',' if str(input_file).endswith('.csv') else '\t'
        cm = pd.read_csv(input_file, sep=sep)
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        sys.exit(1)

    # ==========================================
    # PRE-PROCESSING: FILTERING
    # ==========================================
    n = config['filtering']['threshold']
    high_cols = identify_columns(cm, config['filtering'].get('high_count_criteria'))
    low_cols = identify_columns(cm, config['filtering'].get('low_count_criteria'))

    logger.info(f"Filtering Criteria: Keep if High cols > {n} AND Low cols <= {n}")

    # Apply masks (Default to True if no columns found for that criteria)
    mask_high = (cm[high_cols] > n).all(axis=1) if high_cols else pd.Series(True, index=cm.index)
    mask_low = (cm[low_cols] <= n).all(axis=1) if low_cols else pd.Series(True, index=cm.index)

    cm_filtered = cm[mask_high & mask_low].copy()
    logger.info(f"Rows remaining after filter: {len(cm_filtered)}")

    # ==========================================================
    # STEP 1: VERTICAL EXPANSION (One Cell -> Multiple Chains)
    # ==========================================================
    # This step handles "Dual Beta" cells. 
    # A single row "TRA;TRB1;TRB2" becomes multiple rows.
    
    seq_col = config['processing']['sequence_column']
    delimiter = config['processing']['delimiter']
    target_prefix = config['processing']['target_chain_prefix'] 
    
    logger.info(f"STEP 1: Exploding '{seq_col}' and filtering for '{target_prefix}'...")

    # 1. Split string into list
    cm_filtered[seq_col] = cm_filtered[seq_col].astype(str).str.split(delimiter)
    
    # 2. Explode (Create new rows)
    cm_exploded = cm_filtered.explode(seq_col)
    
    # 3. Filter (Keep only Target Chain, e.g., TRB)
    cm_exploded = cm_exploded[cm_exploded[seq_col].str.startswith(target_prefix, na=False)]

    # 4. Clean Strings (Remove prefix)
    if config['processing'].get('strip_prefix', False):
        cm_exploded[seq_col] = cm_exploded[seq_col].str.replace(target_prefix, '', regex=False)

    # 5. Assign Count source
    count_source = config['processing'].get('count_source_column')
    if count_source and count_source in cm_exploded.columns:
        cm_exploded['count'] = cm_exploded[count_source]

    # 6. Apply Column Mapping
    col_map = config['column_map']
    valid_map = {k: v for k, v in col_map.items() if k in cm_exploded.columns}
    cm_mapped = cm_exploded[list(valid_map.keys())].copy()
    cm_mapped.rename(columns=valid_map, inplace=True)

    # ==========================================================
    # STEP 2: VERTICAL COMPRESSION (Deduplication & Aggregation)
    # ==========================================================
    # Identical chains from DIFFERENT cells are merged here.
    # We calculate Max Expression (count) and Clonal Expansion (n_instances).
    
    if config.get('deduplication', {}).get('perform_dedup', False):
        dedup_cfg = config['deduplication']
        
        group_cols = dedup_cfg['group_by_cols']       # [junction_aa, v_call, j_call]
        agg_col = dedup_cfg['aggregation_col']          # count
        agg_method = dedup_cfg['aggregation_method']    # max
        instance_col = dedup_cfg['size_metric_name']    # n_instances

        logger.info(f"STEP 2: Grouping by {group_cols}...")

        # A. Calculate 'n_instances' (Clonal Expansion)
        instance_counts = cm_mapped.groupby(group_cols).size().reset_index(name=instance_col)

        # B. Calculate Max Expression & Keep Metadata
        # Build aggregation rules
        agg_rules = {agg_col: agg_method}
        
        # For all metadata columns (like clonotype_id), keep the FIRST one found.
        meta_cols = [c for c in cm_mapped.columns if c not in group_cols and c != agg_col]
        for c in meta_cols:
            agg_rules[c] = 'first'

        cm_grouped = cm_mapped.groupby(group_cols).agg(agg_rules).reset_index()

        # C. Merge Counts back in
        cm_final = pd.merge(cm_grouped, instance_counts, on=group_cols)

        logger.info(f"Reduced from {len(cm_mapped)} chains to {len(cm_final)} unique sequences.")
    else:
        cm_final = cm_mapped

    # ==========================================================
    # STEP 3: ID UNIQUIFICATION (Handling Dual Expressors)
    # ==========================================================
    # If a cell had 2 TRBs, Step 1 split them, Step 2 grouped them.
    # cm_final might now have 2 rows with the same 'clonotype_id'.
    # We must make IDs unique.

    id_col = "clonotype_id"
    if id_col in cm_final.columns:
        # Check for duplicates
        dupes_mask = cm_final.duplicated(subset=[id_col], keep=False)
        
        if dupes_mask.any():
            logger.info(f"STEP 3: Uniquifying {dupes_mask.sum()} IDs from Dual Expressors...")
            
            # Load preferences
            uniq_cfg = config.get('uniquification', {'method': 'counter', 'separator': '_'})
            method = uniq_cfg.get('method', 'counter')
            sep = uniq_cfg.get('separator', '_')

            if method == "counter":
                # Format: clonotype_1, clonotype_2
                counter = cm_final[dupes_mask].groupby(id_col).cumcount() + 1
                cm_final.loc[dupes_mask, id_col] = (
                    cm_final.loc[dupes_mask, id_col].astype(str) 
                    + sep 
                    + counter.astype(str)
                )
            
            elif method == "gene":
                # Format: clonotype_TRBV12
                suffix_col = "v_call"
                if suffix_col in cm_final.columns:
                    cm_final.loc[dupes_mask, id_col] = (
                        cm_final.loc[dupes_mask, id_col].astype(str) 
                        + sep 
                        + cm_final.loc[dupes_mask, suffix_col].astype(str)
                    )
                else:
                    # Fallback to counter if gene column missing
                    counter = cm_final[dupes_mask].groupby(id_col).cumcount() + 1
                    cm_final.loc[dupes_mask, id_col] = cm_final.loc[dupes_mask, id_col] + sep + counter.astype(str)

    # ==========================================
    # SAVE OUTPUT
    # ==========================================
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_sep = ',' if output_file.endswith('.csv') else '\t'
    
    cm_final.to_csv(output_file, sep=out_sep, index=False)
    logger.info(f"Saved output to {output_file}")

def main():
    # CLI Arguments
    parser = argparse.ArgumentParser(description="TCR Count Matrix Filter & Processor")
    parser.add_argument('-i', '--input', required=True, help="Input TSV/CSV file")
    parser.add_argument('-o', '--output', required=True, help="Output file path")
    parser.add_argument('-c', '--config', required=True, help="Path to YAML config")
    
    args = parser.parse_args()

    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    process_tcr_data(args.input, args.output, config)

if __name__ == "__main__":
    main()