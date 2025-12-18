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
    Parses the input file and returns pairs + all unique sequences.
    """
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
        
        print(f"✅ Loaded {len(valid_pairs)} valid pairs from {len(df)} rows.")
        return valid_pairs, list(all_unique_seqs)

    except Exception as e:
        print(f"❌ Error loading matched file: {e}")
        raise

# -------------------------------------------------------------------------
# 3) Worker Function (Vectorized & Optimized)
# -------------------------------------------------------------------------
def process_patient_ratios(
    relative_path: str,
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
        
        # 2. OPTIMIZATION: Collapse duplicates first (groupby sequence, sum frequency)
        # This preserves the biological 'abundance' while reducing the row count for distance calc.
        # Even with only 20% repeats, this cleans the data structure for the next step.
        collapsed = rep.groupby(seq_col)[freq_col].sum().reset_index()
        
        # 3. Add Length Column for Blocking
        collapsed['len'] = collapsed[seq_col].str.len()
        
        # 4. BLOCK PROCESSING: Iterate by patient sequence length
        # This avoids comparing length 10 seqs against length 20 seqs.
        for p_len, group in collapsed.groupby('len'):
            
            # Get the patient sequences and weights for this length block
            p_seqs = group[seq_col].tolist()
            p_weights = group[freq_col].to_numpy()
            
            # Gather relevant reference sequences (within max_distance length)
            # Flatten the list of lists into a single list of candidate controls
            candidates = []
            for tgt_len in range(p_len - max_distance, p_len + max_distance + 1):
                candidates.extend(hc_buckets.get(tgt_len, []))
            
            if not candidates:
                continue
                
            # 5. VECTORIZED DISTANCE CALCULATION (C-Level Speed)
            # cdist computes the matrix of distances between all p_seqs and all candidates.
            # score_cutoff stops calculation early if dist > max_distance
            dist_matrix = cdist(p_seqs, candidates, scorer=L.distance, score_cutoff=max_distance, workers=1)
            
            # 6. Extract Matches and Accumulate Weights
            # cdist returns values where distance <= cutoff. 
            # We find indices where the distance is valid (not -1 or whatever strict cutoff indicator is used,
            # but usually cdist just returns the distance. If strictly > cutoff, it might optimize out, 
            # so we standard check <= max_distance).
            
            # Get indices (row=patient_idx, col=candidate_idx) where distance is valid
            rows, cols = np.where(dist_matrix <= max_distance)
            
            # If no matches, continue
            if len(rows) == 0:
                continue

            # Map back to frequencies
            # We iterate the match indices. 
            # Note: A single patient seq can match multiple candidates.
            # Note: Multiple patient seqs can match a single candidate.
            
            # Optimization: We can simply iterate the sparse matches found
            for r, c in zip(rows, cols):
                candidate_seq = candidates[c]
                weight = p_weights[r]
                found_freqs[candidate_seq] += weight

    except Exception as e:
        print(f"⚠️ Failed on {full_path}: {e}")
        # Return empty on failure to avoid crashing the whole pool
        return patient_id, {}

    # Calculate Ratios
    ratios = {data['test']: (found_freqs.get(data['test'], 0.0) + epsilon) / 
              ((sum(found_freqs.get(c, 0.0) for c in data['controls']) / len(data['controls']) if data['controls'] else 0.0) + epsilon)
              for pid, data in pair_data.items()}

    return patient_id, ratios

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
        
    if not os.path.exists(paths['input_dir']):
        raise FileNotFoundError(f"Input dir not found: {paths['input_dir']}")
    
    search_pattern = os.path.join(paths['input_dir'], "**", "*.parquet")
    all_full_paths = glob.glob(search_pattern, recursive=True)
    
    file_names = [os.path.relpath(p, paths['input_dir']) for p in all_full_paths]
    
    if not file_names:
        print(f"⚠️ No parquet files found in {paths['input_dir']}!")
        return

    if settings.get('sample_n') and settings['sample_n'] < len(file_names):
        random.seed(settings.get('random_seed', 42))
        file_names = random.sample(file_names, settings['sample_n'])
        print(f"🎲 Sampling {len(file_names)} files.")
    else:
        print(f"📂 Processing all {len(file_names)} files.")

    # Convert hc_buckets to dict for safe pickling/passing
    worker = partial(
        process_patient_ratios,
        input_dir=paths['input_dir'],
        sep=settings.get('sep', ','),
        seq_col=settings['seq_col'],
        freq_col=settings['freq_col'],
        max_distance=settings['max_distance'],
        hc_buckets=dict(hc_buckets),
        pair_data=pair_data,
        epsilon=float(calc['epsilon'])
    )

    results = []
    print(f"🚀 Starting parallel pool with {settings['n_workers']} workers...")
    
    # Chunksize optimization for smoother progress bars on large file counts
    chunk_size = max(1, len(file_names) // (settings['n_workers'] * 4))

    with ProcessPoolExecutor(max_workers=settings['n_workers']) as exe:
        for idx, (patient_id, ratios) in enumerate(exe.map(worker, file_names, chunksize=chunk_size), start=1):
            if idx % 10 == 0:
                print(f"   Processed {idx}/{len(file_names)}...", end='\r')
            if not ratios: continue
            for seq, val in ratios.items():
                results.append({'patient_id': patient_id, 'hc_seq': seq, 'ratio': val})
    
    print("") # Newline after progress

    if results:
        print("💾 Aggregating and saving results...")
        df_long = pd.DataFrame(results)
        
        # We use pivot_table instead of pivot to handle duplicate patient_id/hc_seq pairs.
        # aggfunc='mean' ensures that if a patient has 2 files (e.g. part1 and part2), 
        # their ratios are averaged together rather than causing a crash.
        matrix = df_long.pivot_table(
            index='hc_seq', 
            columns='patient_id', 
            values='ratio', 
            aggfunc='mean'
        )
        
        # Optional: If you'd rather SUM the ratios (depending on your math), change to aggfunc='sum'
        
        os.makedirs(os.path.dirname(paths['output_file']), exist_ok=True)
        matrix.to_csv(paths['output_file'])
        print(f"✅ Matrix saved to: {paths['output_file']} | Shape: {matrix.shape}")
    else:
        print("⚠️ No results generated.")

if __name__ == '__main__':
    main()