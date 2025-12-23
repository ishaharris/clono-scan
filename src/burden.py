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
from rapidfuzz.process import cdist

# -------------------------------------------------------------------------
# GLOBAL VARIABLES FOR WORKERS
# Using globals with an initializer prevents re-sending large data for every task.
# -------------------------------------------------------------------------
worker_hc_buckets = None
worker_pair_data = None

def init_worker(hc_buckets, pair_data):
    """Initializes global variables once per worker process."""
    global worker_hc_buckets, worker_pair_data
    worker_hc_buckets = hc_buckets
    worker_pair_data = pair_data

# -------------------------------------------------------------------------
# 1) Config Loader
# -------------------------------------------------------------------------
def load_config(config_path="burden.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# -------------------------------------------------------------------------
# 2) Data Loader
# -------------------------------------------------------------------------
def load_matched_pairs(filepath):
    print(f"📖 Loading matched sequences from {filepath}...")
    try:
        sep = '\t' if filepath.endswith('.tsv') else ','
        df = pd.read_csv(filepath, sep=sep)
        df.columns = df.columns.str.strip()
        
        pairs = defaultdict(lambda: {'test': None, 'controls': []})
        all_unique_seqs = set()

        for _, row in df.iterrows():
            pid = row['pair_id']
            seq = row['junction_aa']
            stype = row['sequence_type']
            
            if not isinstance(seq, str) or pd.isna(seq):
                continue
                
            all_unique_seqs.add(seq)
            if stype == 'test':
                pairs[pid]['test'] = seq
            elif stype == 'control':
                pairs[pid]['controls'].append(seq)
        
        valid_pairs = {k: v for k, v in pairs.items() if v['test'] is not None}
        print(f"✅ Loaded {len(valid_pairs)} valid pairs.")
        return valid_pairs, list(all_unique_seqs)
    except Exception as e:
        print(f"❌ Error loading matched file: {e}")
        raise

# -------------------------------------------------------------------------
# 3) Worker Function
# -------------------------------------------------------------------------
def process_patient_ratios(
    relative_path: str,
    input_dir: str,
    sep: str,
    seq_col: str,
    freq_col: str,
    max_distance: int
) -> tuple:
    """
    Processes a single file. Uses worker_hc_buckets from global scope.
    """
    global worker_hc_buckets
    full_path = os.path.join(input_dir, relative_path)
    
    # Extract Patient ID
    parts = relative_path.split(os.sep)
    patient_id = "unknown"
    for part in parts:
        if "sample_id=" in part:
            patient_id = part.split("=")[1]
            break
    if patient_id == "unknown":
        patient_id = os.path.splitext(os.path.basename(relative_path))[0]

    found_freqs = defaultdict(float)

    try:
        # 1. Load Data
        if full_path.endswith('.parquet'):
            rep = pd.read_parquet(full_path, columns=[seq_col, freq_col]).dropna()
        else:
            rep = pd.read_csv(full_path, sep=sep, usecols=[seq_col, freq_col], encoding='latin-1').dropna()
        
        # 2. Collapse duplicates (biological abundance)
        collapsed = rep.groupby(seq_col)[freq_col].sum().reset_index()
        collapsed['len'] = collapsed[seq_col].str.len()
        
        # 3. Block Processing by length
        for p_len, group in collapsed.groupby('len'):
            p_seqs = group[seq_col].tolist()
            p_weights = group[freq_col].to_numpy()
            
            candidates = []
            for tgt_len in range(p_len - max_distance, p_len + max_distance + 1):
                candidates.extend(worker_hc_buckets.get(tgt_len, []))
            
            if not candidates:
                continue
                
            # Vectorized distance calculation
            # Note: workers=1 here because we are already inside a process pool.
            dist_matrix = cdist(p_seqs, candidates, scorer=L.distance, score_cutoff=max_distance, workers=1)
            
            rows, cols = np.where(dist_matrix <= max_distance)
            for r, c in zip(rows, cols):
                candidate_seq = candidates[c]
                found_freqs[candidate_seq] += p_weights[r]

    except Exception as e:
        print(f"⚠️ Failed on {full_path}: {e}")
        return patient_id, {}

    return patient_id, dict(found_freqs)

# -------------------------------------------------------------------------
# 4) Orchestrator
# -------------------------------------------------------------------------
def main():
    cfg = load_config("burden.yaml")
    paths, settings, calc = cfg['paths'], cfg['settings'], cfg['calculation']
    
    pair_data, all_seqs = load_matched_pairs(paths['matched_sequences_file'])
    
    # Pre-bucket reference sequences by length
    hc_buckets = defaultdict(list)
    for seq in all_seqs:
        hc_buckets[len(seq)].append(seq)
    hc_buckets = dict(hc_buckets) # Convert to standard dict for pickling
        
    search_pattern = os.path.join(paths['input_dir'], "**", "*.parquet")
    all_full_paths = glob.glob(search_pattern, recursive=True)
    file_names = [os.path.relpath(p, paths['input_dir']) for p in all_full_paths]
    
    if not file_names:
        print(f"⚠️ No files found in {paths['input_dir']}!")
        return

    # Sampling logic
    if settings.get('sample_n') and settings['sample_n'] < len(file_names):
        random.seed(settings.get('random_seed', 42))
        file_names = random.sample(file_names, settings['sample_n'])
        print(f"🎲 Sampling {len(file_names)} files.")

    # Partial function for non-global arguments
    worker_func = partial(
        process_patient_ratios,
        input_dir=paths['input_dir'],
        sep=settings.get('sep', ','),
        seq_col=settings['seq_col'],
        freq_col=settings['freq_col'],
        max_distance=settings['max_distance']
    )

    patient_counts = defaultdict(lambda: defaultdict(float))
    n_workers = settings.get('n_workers', 1)
    chunk_size = max(1, len(file_names) // (n_workers * 2))

    print(f"🚀 Starting parallel pool with {n_workers} workers...")
    
    # Process Pool with Initializer
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(hc_buckets, pair_data)
    ) as exe:
        # Map the worker function over the file list
        for patient_id, freqs in exe.map(worker_func, file_names, chunksize=chunk_size):
            if not freqs: continue
            for seq, weight in freqs.items():
                patient_counts[patient_id][seq] += weight

    # Final Aggregation
    final_results = []
    epsilon = float(calc['epsilon'])

    print("📊 Calculating final ratios...")
    for patient_id, summed_freqs in patient_counts.items():
        for pid, data in pair_data.items():
            test_seq = data['test']
            control_seqs = data['controls']
            
            test_val = summed_freqs.get(test_seq, 0.0)
            control_vals = [summed_freqs.get(c, 0.0) for c in control_seqs]
            ctrl_avg = sum(control_vals) / len(control_vals) if control_vals else 0.0
            
            ratio = (test_val + epsilon) / (ctrl_avg + epsilon)
            final_results.append({
                'patient_id': patient_id, 
                'hc_seq': test_seq, 
                'ratio': ratio
            })

    if final_results:
        print("💾 Saving results...")
        df_long = pd.DataFrame(final_results)
        matrix = df_long.pivot_table(
            index='hc_seq', 
            columns='patient_id', 
            values='ratio'
        )
        os.makedirs(os.path.dirname(paths['output_file']), exist_ok=True)
        matrix.to_csv(paths['output_file'])
        print(f"✅ Success! Saved to {paths['output_file']} | Shape: {matrix.shape}")
    else:
        print("❌ No results generated.")

if __name__ == '__main__':
    main()