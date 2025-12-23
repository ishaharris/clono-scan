import os
import yaml
import random
import glob
import polars as pl
import numpy as np
from functools import partial
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from rapidfuzz.distance import Levenshtein as L
from rapidfuzz.process import cdist
from tqdm import tqdm  # For a better progress bar

# --- GLOBALS (To prevent IPC overhead/pickling) ---
G_HC_BUCKETS = None
G_PAIR_DATA = None

# -------------------------------------------------------------------------
# 1) Config & Data Loaders
# -------------------------------------------------------------------------
def load_config(config_path="burden.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_matched_pairs(filepath):
    print(f"📖 Loading matched sequences from {filepath}...")
    # Use Polars for fast loading
    df = pl.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    
    pairs = defaultdict(lambda: {'test': None, 'controls': []})
    all_unique_seqs = set()

    for row in df.to_dicts():
        pid = row['pair_id']
        seq = row['junction_aa']
        stype = row['sequence_type']
        
        if not seq or not isinstance(seq, str): continue
        
        all_unique_seqs.add(seq)
        if stype == 'test':
            pairs[pid]['test'] = seq
        elif stype == 'control':
            pairs[pid]['controls'].append(seq)
    
    valid_pairs = {k: v for k, v in pairs.items() if v['test'] is not None}
    return valid_pairs, list(all_unique_seqs)

# -------------------------------------------------------------------------
# 2) Optimized Worker
# -------------------------------------------------------------------------
def process_patient_ratios(
    relative_path: str,
    input_dir: str,
    seq_col: str,
    freq_col: str,
    max_distance: int,
    epsilon: float
) -> tuple:
    # Use Global data inherited from parent process (Copy-on-Write)
    global G_HC_BUCKETS, G_PAIR_DATA
    
    full_path = os.path.join(input_dir, relative_path)
    
    # Extract Patient ID
    patient_id = "unknown"
    if "sample_id=" in relative_path:
        patient_id = relative_path.split("sample_id=")[1].split(os.sep)[0]
    else:
        patient_id = os.path.splitext(os.path.basename(relative_path))[0]

    found_freqs = defaultdict(float)

    try:
        # 1. Faster Reading & Pre-collapsing with Polars
        if full_path.endswith('.parquet'):
            df = pl.read_parquet(full_path, columns=[seq_col, freq_col])
        else:
            df = pl.read_csv(full_path, columns=[seq_col, freq_col], ignore_errors=True)
        
        # Collapse duplicates immediately
        collapsed = (
            df.drop_nulls()
            .group_by(seq_col)
            .agg(pl.col(freq_col).sum())
            .with_columns(pl.col(seq_col).str.len_chars().alias("len"))
        )
        
        # 2. Block Processing by Length
        for p_len, group in collapsed.partition_by("len", as_dict=True).items():
            p_seqs = group[seq_col].to_list()
            p_weights = group[freq_col].to_numpy() # shape (N,)
            
            # Find candidate controls within length threshold
            candidates = []
            for tgt_len in range(p_len - max_distance, p_len + max_distance + 1):
                candidates.extend(G_HC_BUCKETS.get(tgt_len, []))
            
            if not candidates: continue
                
            # 3. Vectorized Distance + Weight Accumulation
            # cdist is the bottleneck; limit workers=1 here because parent is multi-processed
            dist_matrix = cdist(p_seqs, candidates, scorer=L.distance, score_cutoff=max_distance, workers=1)
            
            # Create a boolean mask where distance is within threshold
            match_mask = (dist_matrix <= max_distance).astype(np.float32)
            
            # Vectorized Dot Product: Sum weights for each candidate
            # (1 x N_patient_seqs) @ (N_patient_seqs x M_candidates) = (1 x M_candidates)
            summed_weights = p_weights @ match_mask
            
            # Map back to global freq map
            for i, total_weight in enumerate(summed_weights):
                if total_weight > 0:
                    found_freqs[candidates[i]] += total_weight

    except Exception as e:
        return patient_id, None

    # 4. Calculate Ratios
    ratios = {}
    for pid, data in G_PAIR_DATA.items():
        test_val = found_freqs.get(data['test'], 0.0)
        ctrl_sum = sum(found_freqs.get(c, 0.0) for c in data['controls'])
        ctrl_avg = ctrl_sum / len(data['controls']) if data['controls'] else 0.0
        
        ratios[data['test']] = (test_val + epsilon) / (ctrl_avg + epsilon)

    return patient_id, ratios

# -------------------------------------------------------------------------
# 3) Orchestrator
# -------------------------------------------------------------------------
def main():
    global G_HC_BUCKETS, G_PAIR_DATA
    
    cfg = load_config("burden.yaml")
    paths, settings, calc = cfg['paths'], cfg['settings'], cfg['calculation']
    
    # 1. Load reference pairs
    pair_data, all_seqs = load_matched_pairs(paths['matched_sequences_file'])
    
    # 2. Prepare Globals
    hc_buckets = defaultdict(list)
    for seq in all_seqs:
        hc_buckets[len(seq)].append(seq)
    
    G_HC_BUCKETS = dict(hc_buckets)
    G_PAIR_DATA = pair_data
    
    # 3. Collect files
    search_pattern = os.path.join(paths['input_dir'], "**", "*.parquet")
    file_names = [os.path.relpath(p, paths['input_dir']) for p in glob.glob(search_pattern, recursive=True)]
    
    if settings.get('sample_n') and settings['sample_n'] < len(file_names):
        random.seed(42)
        file_names = random.sample(file_names, settings['sample_n'])

    # 4. Setup Worker
    worker = partial(
        process_patient_ratios,
        input_dir=paths['input_dir'],
        seq_col=settings['seq_col'],
        freq_col=settings['freq_col'],
        max_distance=settings['max_distance'],
        epsilon=float(calc['epsilon'])
    )

    results = []
    num_workers = settings['n_workers']
    print(f"🚀 Processing {len(file_names)} files using {num_workers} workers...")

    # 5. Parallel Execution with tqdm progress bar
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        # Smaller chunksize (e.g., 10-20) ensures the progress bar updates frequently
        mapper = exe.map(worker, file_names, chunksize=15)
        
        for patient_id, ratios in tqdm(mapper, total=len(file_names), desc="Progress"):
            if ratios is None: continue
            for seq, val in ratios.items():
                results.append({'patient_id': patient_id, 'hc_seq': seq, 'ratio': val})

    # 6. Save Matrix
    if results:
        print("💾 Saving Results...")
        df_out = pl.DataFrame(results)
        # Polars pivot is extremely fast
        matrix = df_out.pivot(
            values="ratio",
            index="hc_seq",
            on="patient_id",
            aggregate_function="mean"
        )
        os.makedirs(os.path.dirname(paths['output_file']), exist_ok=True)
        matrix.write_csv(paths['output_file'])
        print(f"✅ Success! Matrix shape: {matrix.shape}")

if __name__ == '__main__':
    main()