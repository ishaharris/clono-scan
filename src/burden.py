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

# --- GLOBALS ---
G_HC_BUCKETS = None
G_PAIR_DATA = None

def init_worker(buckets, pair_data):
    global G_HC_BUCKETS, G_PAIR_DATA
    G_HC_BUCKETS = buckets
    G_PAIR_DATA = pair_data

def process_patient_ratios(relative_path, input_dir, seq_col, freq_col, max_distance, epsilon):
    global G_HC_BUCKETS, G_PAIR_DATA
    if G_HC_BUCKETS is None: return "error", None

    full_path = os.path.join(input_dir, relative_path)
    
    # Extract Patient ID
    if "sample_id=" in relative_path:
        patient_id = relative_path.split("sample_id=")[1].split(os.sep)[0]
    else:
        patient_id = os.path.splitext(os.path.basename(relative_path))[0]

    found_freqs = defaultdict(float)

    try:
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
        
        for p_len_raw, group in collapsed.partition_by("len", as_dict=True).items():
            p_len = p_len_raw[0] if isinstance(p_len_raw, tuple) else p_len_raw
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

        ratios = []
        for pid, data in G_PAIR_DATA.items():
            test_val = found_freqs.get(data['test'], 0.0)
            ctrl_sum = sum(found_freqs.get(c, 0.0) for c in data['controls'])
            ctrl_avg = ctrl_sum / len(data['controls']) if data['controls'] else 0.0
            ratio_val = (test_val + epsilon) / (ctrl_avg + epsilon)
            # Return as a list of dicts for easy Polars conversion
            ratios.append({'patient_id': patient_id, 'hc_seq': data['test'], 'ratio': ratio_val})

        return patient_id, ratios

    except Exception as e:
        print(f"Error in {relative_path}: {e}")
        return patient_id, None

def main():
    with open("burden.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    paths, settings, calc = cfg['paths'], cfg['settings'], cfg['calculation']
    raw_output = paths['output_file'].replace(".csv", "_raw.csv")
    
    # 1. Load Reference Data
    df_matched = pl.read_csv(paths['matched_sequences_file'])
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
    
    # 2. Resuming Logic: Check what is already processed
    processed_patients = set()
    if os.path.exists(raw_output):
        print(f"🔄 Found existing progress at {raw_output}. Checking processed samples...")
        try:
            processed_patients = set(pl.scan_csv(raw_output).select("patient_id").collect().to_series().unique())
            print(f"⏭️ Skipping {len(processed_patients)} already processed patients.")
        except Exception:
            print("⚠️ Could not read progress file. Starting fresh.")

    # 3. Collect Files and Filter
    search_pattern = os.path.join(paths['input_dir'], "**", "*.parquet")
    all_files = [os.path.relpath(p, paths['input_dir']) for p in glob.glob(search_pattern, recursive=True)]
    
    # Filter out already processed patients
    file_names = []
    for f in all_files:
        p_id = f.split("sample_id=")[1].split(os.sep)[0] if "sample_id=" in f else os.path.splitext(os.path.basename(f))[0]
        if p_id not in processed_patients:
            file_names.append(f)

    if settings.get('sample_n') and settings['sample_n'] < len(file_names):
        file_names = random.sample(file_names, settings['sample_n'])

    if not file_names:
        print("✅ All patients already processed. Proceeding to final pivot.")
    else:
        worker_func = partial(
            process_patient_ratios,
            input_dir=paths['input_dir'],
            seq_col=settings['seq_col'],
            freq_col=settings['freq_col'],
            max_distance=settings['max_distance'],
            epsilon=float(calc['epsilon'])
        )

        print(f"🚀 Processing {len(file_names)} new files...")
        with ProcessPoolExecutor(
            max_workers=settings['n_workers'],
            initializer=init_worker,
            initargs=(dict(hc_buckets), valid_pair_data)
        ) as exe:
            mapper = exe.map(worker_func, file_names, chunksize=1) # Smaller chunksize for frequent saving
            
            for patient_id, batch_results in tqdm(mapper, total=len(file_names), desc="Progress"):
                if batch_results:
                    df_batch = pl.DataFrame(batch_results)
                    # Append to CSV (write header only if file doesn't exist)
                    file_exists = os.path.exists(raw_output)
                    with open(raw_output, "ab") as f:
                        df_batch.write_csv(f, include_header=not file_exists)

    # 4. Final Pivoting (Memory Efficient)
    if os.path.exists(raw_output):
        print("📊 Creating final pivot table...")
        # Use scan_csv for lazy processing to keep RAM low
        final_matrix = (
            pl.scan_csv(raw_output)
            .collect() # Pivot currently requires collecting, but we do it only once at the end
            .pivot(values="ratio", index="hc_seq", on="patient_id", aggregate_function="mean")
        )
        final_matrix.write_csv(paths['output_file'])
        print(f"✅ All done! Saved to {paths['output_file']}")
    else:
        print("❌ No data found to process.")

if __name__ == '__main__':
    main()