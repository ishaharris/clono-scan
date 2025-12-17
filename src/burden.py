import os
import yaml
import random
import pandas as pd
import numpy as np
import glob
from functools import partial
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from rapidfuzz.distance import Levenshtein as L

# -------------------------------------------------------------------------
# 1) Config Loader
# -------------------------------------------------------------------------
def load_config(config_path="burden.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# -------------------------------------------------------------------------
# 2) Data Loader: Parse the Matched Sequence File
# -------------------------------------------------------------------------
def load_matched_pairs(filepath):
    """
    Parses the input file (pair_id, sequence_type, junction_aa) and returns
    a dictionary organizing sequences by pair_id.
    
    Returns:
        pairs: dict -> { pair_id: { 'test': 'CAS...', 'controls': ['CAS...', ...] } }
        all_unique_seqs: set -> all sequences to search for in the patient file
    """
    print(f"📖 Loading matched sequences from {filepath}...")
    try:
        # Detect separator based on extension or default to comma/tab
        sep = '\t' if filepath.endswith('.tsv') else ','
        df = pd.read_csv(filepath, sep=sep)
        
        # Standardize column names if necessary (trim whitespace)
        df.columns = df.columns.str.strip()
        
        pairs = defaultdict(lambda: {'test': None, 'controls': []})
        all_unique_seqs = set()

        for _, row in df.iterrows():
            pid = row['pair_id']
            seq = row['junction_aa']
            stype = row['sequence_type'] # 'test' or 'control'
            
            # Skip invalid sequences
            if not isinstance(seq, str) or pd.isna(seq):
                continue
                
            all_unique_seqs.add(seq)
            
            if stype == 'test':
                pairs[pid]['test'] = seq
            elif stype == 'control':
                pairs[pid]['controls'].append(seq)
        
        # Validation: Remove pairs that don't have a test sequence
        valid_pairs = {k: v for k, v in pairs.items() if v['test'] is not None}
        
        print(f"✅ Loaded {len(valid_pairs)} valid pairs from {len(df)} rows.")
        return valid_pairs, list(all_unique_seqs)

    except Exception as e:
        print(f"❌ Error loading matched file: {e}")
        raise

# -------------------------------------------------------------------------
# 3) Worker Function (Updated for Subfolders)
# -------------------------------------------------------------------------
def process_patient_ratios(
    relative_path: str, # Now receiving the path relative to input_dir
    input_dir: str,
    sep: str,
    seq_col: str,
    freq_col: str,
    max_distance: int,
    hc_buckets: dict,
    pair_data: dict,
    epsilon: float
) -> tuple:
    
    full_path = os.path.join(input_dir, relative_path)
    
    # --- PATIENT ID LOGIC ---
    # If the path is 'sample_id=P00001/file.parquet', we want 'P00001'
    # We look for the folder name containing 'sample_id='
    parts = relative_path.split(os.sep)
    patient_id = "unknown"
    for part in parts:
        if "sample_id=" in part:
            patient_id = part.split("=")[1]
            break
    
    # Fallback: if no 'sample_id=' found, use the filename
    if patient_id == "unknown":
        patient_id = os.path.splitext(os.path.basename(relative_path))[0]

    found_freqs = defaultdict(float)

    try:
        # Load Parquet (Subfolders are almost always Parquet)
        if full_path.endswith('.parquet'):
            rep = pd.read_parquet(full_path, columns=[seq_col, freq_col]).dropna()
        else:
            rep = pd.read_csv(full_path, sep=sep, usecols=[seq_col, freq_col], encoding='latin-1').dropna()
        
        seqs = rep[seq_col].to_numpy()
        freqs = rep[freq_col].to_numpy()

        for seq, w in zip(seqs, freqs):
            L_seq = len(seq)
            for tgt_len in range(L_seq - max_distance, L_seq + max_distance + 1):
                candidates = hc_buckets.get(tgt_len, [])
                for hc in candidates:
                    d = L.distance(seq, hc, score_cutoff=max_distance)
                    if d is not None and d <= max_distance:
                        found_freqs[hc] += w
                        
    except Exception as e:
        print(f"⚠️ Failed on {full_path}: {e}")
        return patient_id, {}

    ratios = {data['test']: (found_freqs.get(data['test'], 0.0) + epsilon) / 
              ((sum(found_freqs.get(c, 0.0) for c in data['controls']) / len(data['controls']) if data['controls'] else 0.0) + epsilon)
              for pid, data in pair_data.items()}

    return patient_id, ratios

# -------------------------------------------------------------------------
# 4) Orchestrator (Updated for Recursive Search)
# -------------------------------------------------------------------------
def main():
    cfg = load_config("burden.yaml")
    paths, settings, calc = cfg['paths'], cfg['settings'], cfg['calculation']
    
    pair_data, all_seqs = load_matched_pairs(paths['matched_sequences_file'])
    hc_buckets = defaultdict(list)
    for seq in all_seqs:
        hc_buckets[len(seq)].append(seq)
        
    if not os.path.exists(paths['input_dir']):
        raise FileNotFoundError(f"Input dir not found: {paths['input_dir']}")
    
    # --- RECURSIVE FILE SEARCH ---
    # This looks into all subfolders for any .parquet files
    search_pattern = os.path.join(paths['input_dir'], "**", "*.parquet")
    all_full_paths = glob.glob(search_pattern, recursive=True)
    
    # We pass paths relative to the input_dir to the worker
    file_names = [os.path.relpath(p, paths['input_dir']) for p in all_full_paths]
    
    if not file_names:
        print(f"⚠️ No parquet files found in {paths['input_dir']} or its subfolders!")
        return

    if settings['sample_n'] and settings['sample_n'] < len(file_names):
        random.seed(settings['random_seed'])
        file_names = random.sample(file_names, settings['sample_n'])
        print(f"🎲 Sampling {len(file_names)} files.")
    else:
        print(f"📂 Processing all {len(file_names)} files found in subfolders.")

    worker = partial(
        process_patient_ratios,
        input_dir=paths['input_dir'],
        sep=settings['sep'],
        seq_col=settings['seq_col'],
        freq_col=settings['freq_col'],
        max_distance=settings['max_distance'],
        hc_buckets=dict(hc_buckets),
        pair_data=pair_data,
        epsilon=float(calc['epsilon'])
    )

    results = []
    print(f"🚀 Starting parallel pool with {settings['n_workers']} workers...")
    
    with ProcessPoolExecutor(max_workers=settings['n_workers']) as exe:
        for idx, (patient_id, ratios) in enumerate(exe.map(worker, file_names), start=1):
            if idx % 10 == 0:
                print(f"   Processed {idx}/{len(file_names)}...")
            if not ratios: continue
            for seq, val in ratios.items():
                results.append({'patient_id': patient_id, 'hc_seq': seq, 'ratio': val})

    if results:
        df_long = pd.DataFrame(results)
        matrix = df_long.pivot(index='hc_seq', columns='patient_id', values='ratio')
        os.makedirs(os.path.dirname(paths['output_file']), exist_ok=True)
        matrix.to_csv(paths['output_file'])
        print(f"💾 Matrix saved to: {paths['output_file']} | Shape: {matrix.shape}")
    else:
        print("⚠️ No results generated.")

if __name__ == '__main__':
    main()