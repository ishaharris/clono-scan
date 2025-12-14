import argparse
import pandas as pd
import numpy as np
import Levenshtein
import time
import sys
import os
import math
import bisect
import yaml
import contextlib
from collections import defaultdict

# --- OLGA IMPORTS ---
try:
    import olga
    from olga.load_model import GenerativeModelVDJ, GenomicDataVDJ
    from olga.generation_probability import GenerationProbabilityVDJ
    from olga.sequence_generation import SequenceGenerationVDJ
    from importlib.resources import files
except ImportError:
    pass

# --- UTILS ---
@contextlib.contextmanager
def suppress_output():
    """
    Redirects C-level stdout/stderr to devnull to suppress 
    OLGA's 'Unfamiliar typed V usage mask' warnings.
    """
    try:
        null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        save_fds = [os.dup(1), os.dup(2)]
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)
        yield
    finally:
        os.dup2(save_fds[0], 1)
        os.dup2(save_fds[1], 2)
        for fd in null_fds + save_fds:
            os.close(fd)

class SmartDecoyGenerator:
    def __init__(self, species='human', chain='TRB'):
        print(f"Initializing OLGA for {species} {chain}...")
        
        self.paths = self._locate_model_files(species, chain)
        
        print("  - Loading Genomic Data...")
        self.genomic_data = GenomicDataVDJ()
        with suppress_output():
            self.genomic_data.load_igor_genomic_data(
                self.paths['params'], 
                self.paths['v_anchors'], 
                self.paths['j_anchors']
            )
        
        print("  - Loading Generative Model...")
        self.model = GenerativeModelVDJ()
        with suppress_output():
            self.model.load_and_process_igor_model(self.paths['marginals'])
        
        self.gen_prob = GenerationProbabilityVDJ(self.model, self.genomic_data)
        self.seq_gen = SequenceGenerationVDJ(self.model, self.genomic_data)
        
        self._init_gene_maps()
        self.pool = defaultdict(list)

    def _locate_model_files(self, species, chain):
        if species == 'human' and chain == 'TRB':
            folder_name = "human_T_beta"
        elif species == 'mouse' and chain == 'TRB':
            folder_name = "mouse_T_beta"
        else:
            raise ValueError("Script supports human/mouse TRB only.")

        try:
            model_dir = str(files("olga") / "default_models" / folder_name)
        except Exception:
            import olga as _olga
            model_dir = os.path.join(os.path.dirname(_olga.__file__), "default_models", folder_name)

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        print(f"  - Model directory: {model_dir}")

        def pick_anchor(basename):
            csv_path = os.path.join(model_dir, f"{basename}.csv")
            txt_path = os.path.join(model_dir, f"{basename}.txt")
            if os.path.exists(csv_path): return csv_path
            if os.path.exists(txt_path): return txt_path
            raise FileNotFoundError(f"Anchor file {basename} not found in {model_dir}")

        return {
            'params': os.path.join(model_dir, "model_params.txt"),
            'marginals': os.path.join(model_dir, "model_marginals.txt"),
            'v_anchors': pick_anchor("V_gene_CDR3_anchors"),
            'j_anchors': pick_anchor("J_gene_CDR3_anchors")
        }

    def _init_gene_maps(self):
        v_list = None
        j_list = None
        
        if hasattr(self.genomic_data, 'V_names'): v_list = self.genomic_data.V_names
        elif hasattr(self.genomic_data, 'V_segments'): v_list = self.genomic_data.V_segments
        
        if hasattr(self.genomic_data, 'J_names'): j_list = self.genomic_data.J_names
        elif hasattr(self.genomic_data, 'J_segments'): j_list = self.genomic_data.J_segments

        if not v_list:
            v_list = self._parse_gene_file(self.paths['v_anchors'])
        if not j_list:
            j_list = self._parse_gene_file(self.paths['j_anchors'])

        if not v_list or not j_list:
            raise AttributeError("Could not determine gene names from attributes or files.")

        self.v_map = {v.split('*')[0]: i for i, v in enumerate(v_list)}
        self.j_map = {j.split('*')[0]: i for i, j in enumerate(j_list)}

    def _parse_gene_file(self, filepath):
        names = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('gene_name'):
                        continue
                    cleaned = line.replace(';', ',').replace('\t', ',')
                    parts = cleaned.split(',')
                    if parts:
                        names.append(parts[0].strip())
        except Exception:
            pass
        return names

    def get_gene_index(self, gene_name, gene_type):
        if pd.isna(gene_name): return None
        clean_name = str(gene_name).split('*')[0]
        mapping = self.v_map if gene_type == 'V' else self.j_map
        
        if clean_name in mapping: return mapping[clean_name]
        
        if not clean_name.startswith('TR'):
            prefix = 'TRBV' if gene_type == 'V' else 'TRBJ'
            retry_name = prefix + clean_name
            if retry_name in mapping: return mapping[retry_name]
        return None

    def calculate_pgen(self, aa_seq, v_idx=None, j_idx=None):
        try:
            if v_idx is not None and j_idx is not None:
                return self.gen_prob.compute_aa_cdr3_pgen(aa_seq, v_idx, j_idx)
            else:
                 return self.gen_prob.compute_aa_cdr3_pgen(aa_seq)
        except (TypeError, AttributeError):
            try:
                return self.gen_prob.compute_aa_CDR3_pgen(aa_seq, v_idx, j_idx)
            except Exception:
                return 0.0

    def prime_pool(self, unique_lengths, min_candidates=1000):
        print(f"Priming pool for {len(unique_lengths)} unique lengths...")
        
        needed_lengths = set(unique_lengths)
        
        # --- STRATIFICATION SETTINGS (ADJUSTED) ---
        # High Pgen matches are rare. We lower the requirement to 5 to avoid
        # getting stuck in infinite loops for lengths that are naturally rare.
        high_threshold = 1e-07
        min_high_candidates = 5       # <--- CHANGED: Reduced from 50 to 5
        safety_cap_per_length = 2000  # <--- CHANGED: Reduced from 50,000 to 2,000
        
        # Track counts: Total vs High-Pgen
        counts_total = {l: 0 for l in needed_lengths}
        counts_high = {l: 0 for l in needed_lengths}

        target_total = min_candidates
        max_total_gen = 50_000_000 
        batch_size = 100  # Reduced batch size slightly for more responsive updates
        
        total_generated = 0
        total_stored = 0
        start_t = time.time()
        
        print(f"Goal per length: {target_total} total OR ({min_high_candidates} > {high_threshold})")

        while total_generated < max_total_gen:
            if not needed_lengths:
                print(f"\n[!] Success: All buckets satisfied.")
                break

            # --- SILENT BATCH GENERATION ---
            with suppress_output():
                batch_generated = 0
                while batch_generated < batch_size:
                    try:
                        item = self.seq_gen.gen_rnd_prod_CDR3()
                    except Exception:
                        continue 
                    
                    batch_generated += 1
                    total_generated += 1

                    if len(item) == 4: _, cdr3, v, j = item
                    else: cdr3, v, j = item[0], item[1], item[2]

                    l = len(cdr3)

                    # Only process if this length is still needed
                    if l in needed_lengths:
                        pgen = self.calculate_pgen(cdr3, v, j)
                        
                        if pgen > 0:
                            self.pool[l].append((pgen, cdr3))
                            counts_total[l] += 1
                            total_stored += 1
                            
                            # Track High Pgen separately
                            if pgen >= high_threshold:
                                counts_high[l] += 1
                            
                            # --- STOPPING LOGIC ---
                            # 1. Quality Met: We have enough TOTAL candidates AND enough HIGH Pgen candidates
                            quality_met = (counts_total[l] >= target_total and counts_high[l] >= min_high_candidates)
                            
                            # 2. Safety Valve: We have generated enough candidates to know 
                            #    that high-pgen versions of this length likely don't exist.
                            safety_met = (counts_total[l] >= safety_cap_per_length)
                            
                            if quality_met or safety_met:
                                needed_lengths.remove(l)

            # --- PROGRESS UPDATE ---
            elapsed = time.time() - start_t
            rate = total_generated / elapsed if elapsed > 0 else 0
            
            # Show "High Pgen" progress in the status bar
            total_high_found = sum(counts_high.values())
            
            sys.stdout.write(
                f"\r>> Gen: {total_generated:,} | "
                f"Pool: {total_stored:,} (High Pgen: {total_high_found}) | "
                f"Active Lens: {len(needed_lengths)} | "
                f"Rate: {int(rate)}/s"
            )
            sys.stdout.flush()

        sys.stdout.write("\n")
        print(f"Finished. Total stored: {total_stored:,}")
        
        # Sort pool by Pgen (Low to High) for Binary Search later
        for l in self.pool:
            self.pool[l].sort(key=lambda x: x[0])

    def find_pgen_matched_decoys(self, target_pgen, length, 
                                 n_decoys=5, tol_log10=0.5, 
                                 all_validated_seqs=set(), min_dist=2):
        candidates = self.pool[length]
        if not candidates: return []

        pgens = [x[0] for x in candidates]
        idx = bisect.bisect_left(pgens, target_pgen)
        found_decoys = []
        
        search_radius = n_decoys * 50 
        start = max(0, idx - search_radius)
        end = min(len(candidates), idx + search_radius)
        subset = candidates[start:end]
        
        target_log = math.log10(target_pgen) if target_pgen > 0 else -50
        subset.sort(key=lambda x: abs(math.log10(x[0]) - target_log) if x[0] > 0 else 999)

        for pgen, seq in subset:
            if len(found_decoys) >= n_decoys: break
            if pgen <= 0: continue
            
            log_diff = abs(math.log10(pgen) - target_log)
            if log_diff > tol_log10: continue 

            collision = False
            
            if seq in all_validated_seqs:
                collision = True
            
            if not collision and min_dist > 0:
                for val_seq in all_validated_seqs:
                    if abs(len(seq) - len(val_seq)) >= min_dist: continue
                    if Levenshtein.distance(seq, val_seq) < min_dist:
                        collision = True
                        break
            
            if not collision:
                found_decoys.append((seq, pgen))
                
        return found_decoys

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as exc:
        print(f"Error parsing config: {exc}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='decoys_config.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    
    input_file = config.get('input_file')
    output_file = config.get('output_file', 'decoys_output.csv')
    params = config.get('parameters', {})
    
    print(f"Reading {input_file}...")
    try:
        df = pd.read_csv(input_file, sep='\t')
        if df.shape[1] < 2: df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    try:
        generator = SmartDecoyGenerator(species=config.get('species', 'human'))
    except Exception as e:
        print(f"Generator Initialization Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("Analyzing input requirements...")
    required_lengths = set()
    row_metadata = [] 
    valid_indices = []
    
    for idx, row in df.iterrows():
        v_idx = generator.get_gene_index(row['v_call'], 'V')
        j_idx = generator.get_gene_index(row['j_call'], 'J')
        seq = row['junction_aa']
        
        if isinstance(seq, str):
            required_lengths.add(len(seq))
            row_metadata.append({'v': v_idx, 'j': j_idx, 'seq': seq})
            valid_indices.append(idx)
        else:
            row_metadata.append(None)

    generator.prime_pool(required_lengths, min_candidates=params.get('pool_prime_min_candidates', 50))
    
    validated_seqs = set(df['junction_aa'].dropna().unique())
    results = []
    L = params.get('levenshtein_distance_threshold', 1)

    print("Finding matched decoys (Matching on Length + Pgen only)...")
    start_time = time.time()
    
    matches_count = 0
    total_targets = len(valid_indices)

    for i, real_idx in enumerate(valid_indices):
        meta = row_metadata[real_idx]
        target_aa = meta['seq']
        
        # --- PROGRESS INDICATOR (Updates continuously) ---
        sys.stdout.write(f"\rProcessing {i+1}/{total_targets} | Total Matches: {matches_count}...")
        sys.stdout.flush()
        
        target_pgen = generator.calculate_pgen(target_aa, meta['v'], meta['j'])
        
        decoys = generator.find_pgen_matched_decoys(
            target_pgen, len(target_aa),
            n_decoys=params.get('decoys_per_sequence', 5),
            tol_log10=params.get('pgen_tolerance_log10', 1.0),
            all_validated_seqs=validated_seqs,
            min_dist=L + 1
        )
        
        if not decoys:
             results.append({
                 'target_id': real_idx,
                 'target_aa': target_aa,
                 'target_pgen': target_pgen,
                 'v_call': df.loc[real_idx, 'v_call'],
                 'j_call': df.loc[real_idx, 'j_call'],
                 'decoy_rank': None,
                 'decoy_aa': None,
                 'decoy_pgen': None,
                 'pgen_diff_log10': None
             })
             continue
        
        # --- MATCH FOUND PRINT ---
        matches_count += 1
        # \r clears the line so the log print starts fresh
        # \n at the end ensures the log line stays, and the progress bar redraws below it on next loop
        sys.stdout.write(f"\r[+] Target {i+1} ({target_aa}): Found {len(decoys)} decoys (Pgen: {target_pgen:.2e})\n")
        sys.stdout.flush()

        for k, (d_seq, d_pgen) in enumerate(decoys):
            results.append({
                'target_id': real_idx,
                'target_aa': target_aa,
                'target_pgen': target_pgen,
                'v_call': df.loc[real_idx, 'v_call'],
                'j_call': df.loc[real_idx, 'j_call'],
                'decoy_rank': k+1,
                'decoy_aa': d_seq,
                'decoy_pgen': d_pgen,
                'pgen_diff_log10': abs(math.log10(d_pgen) - math.log10(target_pgen)) if target_pgen > 0 and d_pgen > 0 else None
            })

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"\nDone. Saved to {output_file}. Total time: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    main()