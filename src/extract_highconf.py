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

def apply_alignment_rules(seqs, attributes, tag="Attribute"):
    """
    Strictly aligns attribute lists (AA, V, D, J) to the anchor Nucleotide list.
    """
    n_s = len(seqs)
    n_a = len(attributes)

    if n_s == 0: return []

    # Rule 1: Perfect Match (2 NT, 2 AA) -> Keep order 1:1
    if n_s == n_a:
        return attributes

    # Rule 2: 2 NT, 1 Attribute -> Duplicate Attribute to both NTs
    # (e.g. 2 RNA reads, but only 1 V-call assigned by software)
    if n_s == 2 and n_a == 1:
        return [attributes[0], attributes[0]]
    
    # Rule 3: 1 NT, 2 Attributes -> Take First Attribute
    if n_s == 1 and n_a == 2:
        return [attributes[0]]

    # Edge Case: Mismatch lengths (e.g. 3 NT, 2 AA) -> Pad with None
    if n_s > n_a:
        return attributes + [None] * (n_s - n_a)
    else:
        return attributes[:n_s]

def clean_aa_sequence(aa_str, prefixes_to_strip):
    """
    Removes prefixes like 'TRB:' from amino acid strings.
    Input: 'TRB:CAS...  ' -> Output: 'CAS...'
    """
    if not aa_str: return ""
    clean = aa_str.strip()
    for prefix in prefixes_to_strip:
        clean = clean.replace(prefix, '')
    return clean.strip()

def process_row(row, config):
    """
    Extracts parallel lists and iterates by Index to ensure NT/AA coupling.
    """
    cols = config['columns']
    delim = config['processing']['delimiter']
    
    # 1. Safe String Extraction
    raw_nt = str(row[cols['junction']]) if pd.notna(row[cols['junction']]) else ""
    raw_aa = str(row[cols['junction_aa']]) if pd.notna(row[cols['junction_aa']]) else ""
    raw_v = str(row[cols['v_call']]) if pd.notna(row[cols['v_call']]) else ""
    raw_d = str(row[cols['d_call']]) if pd.notna(row[cols['d_call']]) else ""
    raw_j = str(row[cols['j_call']]) if pd.notna(row[cols['j_call']]) else ""
    
    # 2. Split into Lists (Preserving Order)
    nt_list = [x.strip() for x in raw_nt.split(delim) if x.strip()]
    
    # If no junction, skip row
    if not nt_list: return []

    # 3. Handle AA Cleaning
    # Split first, THEN clean specific items. 
    # Example: "TRA:AAA; TRB:BBB" -> ["TRA:AAA", "TRB:BBB"] -> ["AAA", "BBB"]
    aa_dirty = [x.strip() for x in raw_aa.split(delim) if x.strip()]
    aa_list = [clean_aa_sequence(x, config['processing'].get('strip_prefixes', [])) for x in aa_dirty]

    v_list = [x.strip() for x in raw_v.split(delim) if x.strip()]
    d_list = [x.strip() for x in raw_d.split(delim) if x.strip()]
    j_list = [x.strip() for x in raw_j.split(delim) if x.strip()]

    # 4. Align Lists (Ensure all lists are length of nt_list)
    aligned_aa = apply_alignment_rules(nt_list, aa_list, "AA")
    aligned_v = apply_alignment_rules(nt_list, v_list, "V")
    aligned_d = apply_alignment_rules(nt_list, d_list, "D")
    aligned_j = apply_alignment_rules(nt_list, j_list, "J")

    results = []
    val_positive = row[cols['positive']] if pd.notna(row[cols['positive']]) else 0

    # 5. Iterate by Index (Coupling Guarantee)
    for i, seq in enumerate(nt_list):
        
        # Retrieve parallel attributes
        aa_val = aligned_aa[i] if i < len(aligned_aa) else None
        v_val = aligned_v[i] if i < len(aligned_v) else None
        d_val = aligned_d[i] if i < len(aligned_d) else None
        j_val = aligned_j[i] if i < len(aligned_j) else None

        # 6. Filter TRB Only
        # We check the V-gene associated with THIS index.
        if v_val and "TRB" in v_val:
            results.append({
                'junction': seq,       # NT
                'junction_aa': aa_val, # AA (Correctly paired via index i)
                'v_call': v_val,
                'd_call': d_val,
                'j_call': j_val,
                'positive': val_positive,
                'original_index': row.name
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

    # Filter Rows
    n = config['filtering']['threshold']
    high_cols = identify_columns(cm, config['filtering'].get('high_count_criteria'))
    low_cols = identify_columns(cm, config['filtering'].get('low_count_criteria'))

    if high_cols or low_cols:
        mask_high = (cm[high_cols] > n).all(axis=1) if high_cols else pd.Series(True, index=cm.index)
        mask_low = (cm[low_cols] <= n).all(axis=1) if low_cols else pd.Series(True, index=cm.index)
        cm_filtered = cm[mask_high & mask_low].copy()
        logger.info(f"Filtered rows: {len(cm)} -> {len(cm_filtered)}")
    else:
        cm_filtered = cm.copy()
    
    # Explode
    logger.info("Exploding rows and syncing AA/NT sequences...")
    exploded_data = []
    for idx, row in cm_filtered.iterrows():
        row.name = idx
        exploded_data.extend(process_row(row, config))
    
    if not exploded_data:
        logger.error("No valid TRB chains found!")
        sys.exit(1)

    df_exploded = pd.DataFrame(exploded_data)

    # Aggregation
    logger.info("Aggregating clones by Nucleotide sequence...")
    agg_rules = {
        'positive': 'sum',
        'junction_aa': 'first',
        'v_call': 'first',
        'd_call': 'first',
        'j_call': 'first',
        'original_index': 'count'
    }
    
    df_grouped = df_exploded.groupby('junction').agg(agg_rules).reset_index()
    df_grouped.rename(columns={'original_index': 'n_chains_aggregated'}, inplace=True)

    # Ranking
    logger.info("Ranking and assigning Clonotype IDs...")
    df_grouped = df_grouped.sort_values(by='positive', ascending=False).reset_index(drop=True)
    df_grouped['clonotype_id'] = 'clonotype' + (df_grouped.index + 1).astype(str)
    
    cols = ['clonotype_id', 'positive', 'junction', 'junction_aa', 'v_call', 'd_call', 'j_call', 'n_chains_aggregated']
    df_final = df_grouped[cols]

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_sep = ',' if output_file.endswith('.csv') else '\t'
    df_final.to_csv(output_file, sep=out_sep, index=False)
    logger.info(f"Saved {len(df_final)} clones to {output_file}")

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