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
from tqdm import tqdm

# --- GLOBALS (Placeholders) ---
G_HC_BUCKETS = None
G_PAIR_DATA = None

# --- INITIALIZER (Required for macOS/Windows) ---
def init_worker(buckets, pair_data):
    """This function runs once when each worker process starts."""
    global G_HC_BUCKETS, G_PAIR_DATA
    G_HC_BUCKETS = buckets
    G_PAIR_DATA = pair_data

# -------------------------------------------------------------------------
# 1) Optimized Worker
# -------------------------------------------------------------------------
def process_patient_ratios(
    relative_path: str,
    input_dir: str,
    seq_col: str,
    freq_col: str,
    max_distance: int,
    epsilon: float
) -> tuple:
    global G_HC_BUCKETS, G_PAIR_DATA
    
    # Debug: Check if globals exist
    if G_HC_BUCKETS is None or G_PAIR_DATA is None:
        return "error", None

    full_path = os.path.join(input_dir, relative_path)
    
    # Extract Patient ID
    patient_id = "unknown"
    if "sample_id=" in relative_path:
        patient_id = relative_path.split("sample_id=")[1].split(os.sep)[0]
    else:
        patient_id = os.path.splitext(os.path.basename(relative_path))[0]

    found_freqs = defaultdict(float)

    try:
        # Load and Collapse
        if full_path.endswith('.parquet'):
            df = pl.read_parquet(full_path, columns=[seq_col, freq_col])
        else:
            df = pl.read_csv(full_path, columns=[seq_col, freq_col], ignore_errors=True)
        
        collapsed = (
            df.drop_nulls()
            .group_by(seq_col)
            .agg(pl.col(freq_col).sum())
            .with_columns(pl.col(seq_col).str.len_chars().alias("len"))
        )
        
        for p_len, group in collapsed.partition_by("len", as_dict=True).items():
            p_seqs = group[seq_col].to_list()
            p_weights = group[freq_col].to_numpy()
            
            candidates = []
            for tgt_len in range(p_len - max_distance, p_len + max_distance + 1):
                candidates.extend(G_HC_BUCKETS.get(tgt_len, []))
            
            if not candidates: continue
                
            dist_matrix = cdist(p_seqs, candidates, scorer=L.distance, score_cutoff=max_distance, workers=1)
            match_mask = (dist_matrix <= max_distance).astype(np.float32)
            summed_weights = p_weights @ match_mask
            
            for i, total_weight in enumerate(summed_weights):
                if total_weight > 0:
                    found_freqs[candidates[i]] += total_weight

        # Calculate Ratios
        ratios = {}
        for pid, data in G_PAIR_DATA.items():
            test_val = found_freqs.get(data['test'], 0.0)
            ctrl_sum = sum(found_freqs.get(c, 0.0) for c in data['controls'])
            ctrl_avg = ctrl_sum / len(data['controls']) if data['controls'] else 0.0
            ratios[data['test']] = (test_val + epsilon) / (ctrl_avg + epsilon)

        return patient_id, ratios

    except Exception as e:
        # Print actual error for debugging
        print(f"Error in {relative_path}: {e}")
        return patient_id, None

# -------------------------------------------------------------------------
# 2) Orchestrator
# -------------------------------------------------------------------------
def main():
    with open("burden.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    paths, settings, calc = cfg['paths'], cfg['settings'], cfg['calculation']
    
    # Load reference pairs
    df_matched = pl.read_csv(paths['matched_sequences_file'])
    df_matched.columns = [c.strip() for c in df_matched.columns]
    
    pair_data = defaultdict(lambda: {'test': None, 'controls': []})
    all_seqs = set()
    for row in df_matched.to_dicts():
        seq, pid, stype = row['junction_aa'], row['pair_id'], row['sequence_type']
        if not seq: continue
        all_seqs.add(seq)
        if stype == 'test': pair_data[pid]['test'] = seq
        elif stype == 'control': pair_data[pid]['controls'].append(seq)
    
    valid_pair_data = {k: v for k, v in pair_data.items() if v['test'] is not None}
    hc_buckets = defaultdict(list)
    for seq in all_seqs:
        hc_buckets[len(seq)].append(seq)
    
    # Collect files
    search_pattern = os.path.join(paths['input_dir'], "**", "*.parquet")
    file_names = [os.path.relpath(p, paths['input_dir']) for p in glob.glob(search_pattern, recursive=True)]
    
    if settings.get('sample_n') and settings['sample_n'] < len(file_names):
        file_names = random.sample(file_names, settings['sample_n'])

    # Setup Worker
    worker_func = partial(
        process_patient_ratios,
        input_dir=paths['input_dir'],
        seq_col=settings['seq_col'],
        freq_col=settings['freq_col'],
        max_distance=settings['max_distance'],
        epsilon=float(calc['epsilon'])
    )

    results = []
    print(f"🚀 Starting processing {len(file_names)} files...")

    # USE INITIALIZER HERE
    # This sends the data once to each worker upon startup
    with ProcessPoolExecutor(
        max_workers=settings['n_workers'],
        initializer=init_worker,
        initargs=(dict(hc_buckets), valid_pair_data)
    ) as exe:
        mapper = exe.map(worker_func, file_names, chunksize=10)
        
        for patient_id, ratios in tqdm(mapper, total=len(file_names), desc="Progress"):
            if patient_id == "error":
                print("⚠️ Worker failed to initialize globals.")
                continue
            if ratios is None: continue
            for seq, val in ratios.items():
                results.append({'patient_id': patient_id, 'hc_seq': seq, 'ratio': val})

    if results:
        print(f"💾 Saving {len(results)} rows to {paths['output_file']}...")
        df_out = pl.DataFrame(results)
        matrix = df_out.pivot(values="ratio", index="hc_seq", on="patient_id", aggregate_function="mean")
        os.makedirs(os.path.dirname(paths['output_file']), exist_ok=True)
        matrix.write_csv(paths['output_file'])
        print(f"✅ Success! Saved to {paths['output_file']}")
    else:
        print("❌ No results generated! Check if column names in burden.yaml match your files.")

if __name__ == '__main__':
    main()