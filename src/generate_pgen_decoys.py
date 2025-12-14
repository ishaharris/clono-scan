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
import pickle
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

    # --- POOL PERSISTENCE ---
    def save_pool(self, filepath):
        print(f"Saving pool to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(self.pool, f)
        print(f"Pool saved. Contains {sum(len(v) for v in self.pool.values())} sequences across {len(self.pool)} lengths.")

    def load_pool(self, filepath):
        if not os.path.exists(filepath):
            print(f"Pool file {filepath} not found. Starting with empty pool.")
            return False
        
        print(f"Loading pool from {filepath}...")
        try:
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)
                for k, v in loaded_data.items():
                    self.pool[k] = v
            print(f"Pool loaded. Total sequences: {sum(len(v) for v in self.pool.values())}")
            return True
        except Exception as e:
            print(f"Error loading pool: {e}")
            return False

    def prime_pool(self, unique_lengths, min_candidates=1000):
        needed_lengths = set()
        for l in unique_lengths:
            if len(self.pool[l]) < min_candidates:
                needed_lengths.add(l)
        
        if not needed_lengths:
            print("Pool is already sufficient for all requested lengths.")
            return

        print(f"Priming pool for {len(needed_lengths)} specific lengths...")
        
        counts_total = {l: len(self.pool[l]) for l in needed_lengths}
        target_total = min_candidates
        max_total_gen = 50_000_000 
        batch_size = 500 
        
        total_generated = 0
        total_added = 0
        start_t = time.time()
        
        while total_generated < max_total_gen and needed_lengths:
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

                    if l in needed_lengths:
                        pgen = self.calculate_pgen(cdr3, v, j)
                        if pgen > 0:
                            self.pool[l].append((pgen, cdr3))
                            counts_total[l] += 1
                            total_added += 1
                            
                            if counts_total[l] >= target_total:
                                needed_lengths.remove(l)
                                if not needed_lengths: break

            elapsed = time.time() - start_t
            rate = total_generated / elapsed if elapsed > 0 else 0
            
            sys.stdout.write(
                f"\r>> Gen: {total_generated:,} | "
                f"Added: {total_added:,} | "
                f"Remaining Lengths: {len(needed_lengths)} | "
                f"Rate: {int(rate)}/s"
            )
            sys.stdout.flush()

        # --- RESCUE PHASE ---
        if needed_lengths:
            print(f"\n[!] Warning: {len(needed_lengths)} lengths still under-filled. Attempting targeted rescue...")
            for l in list(needed_lengths):
                if len(self.pool[l]) > 0: continue 

                print(f"   -> Forcing generation for length {l}...")
                att = 0
                while len(self.pool[l]) < 50 and att < 200000:
                    att += 1
                    with suppress_output():
                        item = self.seq_gen.gen_rnd_prod_CDR3()
                        cdr3 = item[1] if len(item) == 4 else item[0]
                        if len(cdr3) == l:
                             v = item[2] if len(item) == 4 else item[1]
                             j = item[3] if len(item) == 4 else item[2]
                             pgen = self.calculate_pgen(cdr3, v, j)
                             self.pool[l].append((pgen, cdr3))

        sys.stdout.write("\n")
        print(f"Finished. Total sequences in memory: {sum(len(v) for v in self.pool.values()):,}")
        
        for l in self.pool:
            self.pool[l].sort(key=lambda x: x[0])       

    def find_pgen_matched_decoys(self, target_pgen, length, 
                                 n_decoys=5, 
                                 all_validated_seqs=set(), min_dist=2):
        """
        Returns: (list_of_decoys, failure_reason_string)
        """
        candidates = self.pool[length]
        
        # Reason 1: No pool
        if not candidates: 
            return [], f"Pool empty for length {length}"

        target_log = math.log10(target_pgen) if target_pgen > 0 else -50

        def get_log_diff(pgen_val):
            if pgen_val <= 0: return 999.0
            val_log = math.log10(pgen_val)
            return abs(val_log - target_log)

        # Sort by Pgen closeness
        sorted_candidates = sorted(candidates, key=lambda x: get_log_diff(x[0]))
        
        found_decoys = []
        filtered_count = 0
        
        # FIRST PASS: Strict Levenshtein
        for pgen, seq in sorted_candidates:
            if len(found_decoys) >= n_decoys: break
            if pgen <= 0: continue
            
            collision = False
            if seq in all_validated_seqs: collision = True
            
            if not collision and min_dist > 0:
                for val_seq in all_validated_seqs:
                    if abs(len(seq) - len(val_seq)) >= min_dist: continue
                    if Levenshtein.distance(seq, val_seq) < min_dist:
                        collision = True
                        break
            
            if not collision:
                found_decoys.append((seq, pgen))
            else:
                filtered_count += 1

        # SECOND PASS (Fallback): Relax Levenshtein if needed
        used_fallback = False
        if len(found_decoys) < n_decoys:
            needed = n_decoys - len(found_decoys)
            existing_seqs = set(x[0] for x in found_decoys)
            
            for pgen, seq in sorted_candidates:
                if len(found_decoys) >= n_decoys: break
                if seq in existing_seqs: continue
                if seq in all_validated_seqs: continue # Still reject exact real seqs
                
                found_decoys.append((seq, pgen))
                existing_seqs.add(seq)
                used_fallback = True

        # --- DIAGNOSE REASON ---
        if len(found_decoys) == n_decoys:
            if used_fallback:
                 return found_decoys, "Filled using relaxed Levenshtein fallback"
            else:
                 return found_decoys, "Full"
        else:
            # We still don't have enough
            total_avail = len(candidates)
            if total_avail < n_decoys:
                return found_decoys, f"Pool exhausted (only {total_avail} candidates existed)"
            else:
                return found_decoys, f"Filtered by validation set collision (rejected {filtered_count} candidates)"

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
    parser.add_argument('--pool_file', default=None, help='Path to .pkl file to save/load pool')
    args = parser.parse_args()
    config = load_config(args.config)
    
    input_file = config.get('input_file')
    output_file = config.get('output_file', 'decoys_output.csv')
    params = config.get('parameters', {})
    target_count = params.get('decoys_per_sequence', 5)
    
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
        sys.exit(1)
    
    if args.pool_file:
        generator.load_pool(args.pool_file)
    
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

    generator.prime_pool(required_lengths, min_candidates=params.get('pool_prime_min_candidates', 1000))
    
    if args.pool_file:
        generator.save_pool(args.pool_file)
    
    validated_seqs = set(df['junction_aa'].dropna().unique())
    results = []
    L = params.get('levenshtein_distance_threshold', 1)

    print("Finding 'best possible' matched decoys...")
    start_time = time.time()
    
    matches_count = 0
    total_targets = len(valid_indices)

    for i, real_idx in enumerate(valid_indices):
        meta = row_metadata[real_idx]
        target_aa = meta['seq']
        
        sys.stdout.write(f"\rProcessing {i+1}/{total_targets}...")
        sys.stdout.flush()
        
        target_pgen = generator.calculate_pgen(target_aa, meta['v'], meta['j'])
        
        decoys, reason = generator.find_pgen_matched_decoys(
            target_pgen, len(target_aa),
            n_decoys=target_count,
            all_validated_seqs=validated_seqs,
            min_dist=L + 1
        )
        
        matches_count += len(decoys)
        
        # If no decoys found, log the failure reason
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
                 'pgen_diff_log10': None,
                 'missing_reason': reason
             })
             continue

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
                'pgen_diff_log10': abs(math.log10(d_pgen) - math.log10(target_pgen)) if target_pgen > 0 and d_pgen > 0 else None,
                'missing_reason': reason if len(decoys) < target_count else "Full"
            })

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"\nDone. Saved to {output_file}. Total time: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    main()