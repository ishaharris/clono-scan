import pandas as pd
import yaml
import argparse
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def identify_columns(df, criteria):
    found_cols = set()
    if not criteria: return []
    if 'exact' in criteria and criteria['exact']:
        for col in criteria['exact']:
            if col in df.columns: found_cols.add(col)
    if 'startswith' in criteria and criteria['startswith']:
        for prefix in criteria['startswith']:
            found_cols.update([c for c in df.columns if c.startswith(prefix)])
    return list(found_cols)

def clean_sequence(seq, prefixes_to_strip):
    """
    Basic cleanup: remove whitespace and specific prefixes (e.g., 'TRB:').
    """
    if not seq: return ""
    clean = str(seq).strip()
    for prefix in prefixes_to_strip:
        clean = clean.replace(prefix, '')
    return clean.strip()

def process_row(row, config):
    cols = config['columns']
    processing_cfg = config['processing']
    
    # --- GET DELIMITERS ---
    j_delim = processing_cfg.get('junction_delimiter', ';')
    g_delim = processing_cfg.get('gene_delimiter', ',')
    prefixes = processing_cfg.get('strip_prefixes', [])

    # 1. Get raw strings
    raw_aa = str(row[cols['junction_aa']]) if pd.notna(row[cols['junction_aa']]) else ""
    raw_v = str(row[cols['v_call']]) if pd.notna(row[cols['v_call']]) else ""
    raw_d = str(row[cols['d_call']]) if pd.notna(row[cols['d_call']]) else ""
    raw_j = str(row[cols['j_call']]) if pd.notna(row[cols['j_call']]) else ""
    
    # 2. Split Gene Columns using gene_delimiter
    v_list = [x.strip() for x in raw_v.split(g_delim) if x.strip()]
    d_list = [x.strip() for x in raw_d.split(g_delim) if x.strip()]
    j_list = [x.strip() for x in raw_j.split(g_delim) if x.strip()]

    # 3. Select Genes (Keep first incident only)
    selected_v = next((v for v in v_list if "TRBV" in v), None)
    if not selected_v and v_list:
        selected_v = v_list[0]

    selected_d = d_list[0] if d_list else ""
    selected_j = j_list[0] if j_list else ""

    # 4. Split Junction AA using junction_delimiter
    aa_list = [x.strip() for x in raw_aa.split(j_delim) if x.strip()]
    
    results = []
    
    # Retrieve the pre-calculated sum (dextramer_count)
    # We default to 0 if the column didn't calculate correctly for some reason
    val_count = row['dextramer_count'] if pd.notna(row['dextramer_count']) else 0

    for aa in aa_list:
        # --- FILTER: DROP TRA ---
        if "TRA:" in aa:
            continue
            
        # Clean the sequence
        clean_aa = clean_sequence(aa, prefixes)
        
        if clean_aa:
            results.append({
                'junction_aa': clean_aa,
                'v_call': selected_v,
                'd_call': selected_d,
                'j_call': selected_j,
                'dextramer_count': val_count  # Renamed from 'positive'
            })
            
    return results

def process_tcr_data(input_file, output_file, config):
    logger.info(f"Reading input file: {input_file}")
    try:
        sep = ',' if str(input_file).endswith('.csv') else '\t'
        cm = pd.read_csv(input_file, sep=sep)
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        sys.exit(1)

    # --- FILTERING LOGIC ---
    # Hardcoding threshold to 10 as requested, or use config if needed
    # n = config['filtering']['threshold'] 
    n = 10 
    
    # Identify the relevant columns
    high_cols = identify_columns(cm, config['filtering'].get('high_count_criteria'))
    low_cols = identify_columns(cm, config['filtering'].get('low_count_criteria'))

    if high_cols:
        # Logic: BOTH (all identified) columns must be > 10
        mask_high = (cm[high_cols] > n).all(axis=1)
        logger.info(f"Columns used for Dextramer/Positive check: {high_cols}")
    else:
        mask_high = pd.Series(True, index=cm.index)

    if low_cols:
        mask_low = (cm[low_cols] <= n).all(axis=1)
    else:
        mask_low = pd.Series(True, index=cm.index)

    cm_filtered = cm[mask_high & mask_low].copy()
    logger.info(f"Filtered rows (Threshold > {n}): {len(cm)} -> {len(cm_filtered)}")
    
    # --- CALCULATE SUM ---
    # Calculate the 'dextramer_count' by summing the high_count columns
    if high_cols:
        cm_filtered['dextramer_count'] = cm_filtered[high_cols].sum(axis=1)
    else:
        # Fallback if no high columns defined, checks strictly for a 'positive' col in config
        pos_col = config['columns'].get('positive')
        if pos_col and pos_col in cm_filtered.columns:
            cm_filtered['dextramer_count'] = cm_filtered[pos_col]
        else:
            cm_filtered['dextramer_count'] = 0
            logger.warning("No columns found to sum for dextramer_count. Setting to 0.")

    # Explode Logic
    logger.info("Exploding TRB chains...")
    exploded_data = []
    for idx, row in cm_filtered.iterrows():
        exploded_data.extend(process_row(row, config))
    
    if not exploded_data:
        logger.error("No valid TRB chains found after processing!")
        sys.exit(1)

    df_exploded = pd.DataFrame(exploded_data)

    # Aggregation: Group by Amino Acid sequence
    logger.info("Aggregating by unique Junction AA...")
    
    agg_rules = {
        'dextramer_count': 'sum', # Aggregating the new column
        'v_call': 'first', 
        'd_call': 'first',
        'j_call': 'first'
    }
    
    # Grouping by junction_aa
    df_grouped = df_exploded.groupby('junction_aa').agg(agg_rules).reset_index()

    # Sort by count descending (Most abundant first)
    df_final = df_grouped.sort_values(by='dextramer_count', ascending=False).copy()
    
    # --- NEW: Add ID Column (1 to n) ---
    df_final.insert(0, 'id', range(1, 1 + len(df_final)))

    # Final Column Selection
    cols_to_save = ['id', 'junction_aa', 'dextramer_count', 'v_call', 'd_call', 'j_call']
    df_final = df_final[cols_to_save]

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_sep = ',' if output_file.endswith('.csv') else '\t'
    df_final.to_csv(output_file, sep=out_sep, index=False)
    logger.info(f"Saved {len(df_final)} unique TRBs to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()

    if not Path(args.config).exists():
        sys.exit("Config file not found.")

    config = load_config(args.config)
    process_tcr_data(args.input, args.output, config)

if __name__ == "__main__":
    main()