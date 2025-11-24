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
import random
import copy
from collections import defaultdict

# --- OLGA IMPORTS ---
try:
    import olga
    from olga.load_model import GenerativeModelVDJ, GenomicDataVDJ
    from olga.generation_probability import GenerationProbabilityVDJ
    from olga.sequence_generation import SequenceGenerationVDJ
    from importlib.resources import files
except ImportError:
    print("Warning: OLGA not installed or found.")
    pass

class OutputSilencer:
    """
    Context manager to suppress stdout and stderr from C-level libraries (like OLGA).
    """
    def __init__(self):
        self._original_stdout_fd = None
        self._original_stderr_fd = None
        self._devnull = None

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            self._original_stdout_fd = os.dup(sys.stdout.fileno())
            self._original_stderr_fd = os.dup(sys.stderr.fileno())
        except Exception:
            return
        os.dup2(self._devnull, sys.stdout.fileno())
        os.dup2(self._devnull, sys.stderr.fileno())

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._original_stdout_fd is None:
            return
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._original_stdout_fd, sys.stdout.fileno())
        os.dup2(self._original_stderr_fd, sys.stderr.fileno())
        os.close(self._original_stdout_fd)
        os.close(self._original_stderr_fd)
        os.close(self._devnull)

class SmartDecoyGenerator:
    def __init__(self, species='human', chain='TRB'):
        # We keep initialization prints so you know the script started
        print(f"Initializing OLGA for {species} {chain}...")
        
        self.paths = self._locate_model_files(species, chain)
        
        print("  - Loading Genomic Data...")
        self.genomic_data = GenomicDataVDJ()
        self.genomic_data.load_igor_genomic_data(
            self.paths['params'], 
            self.paths['v_anchors'], 
            self.paths['j_anchors']
        )
        
        print("  - Loading Generative Model...")
        self.model = GenerativeModelVDJ()
        self.model.load_and_process_igor_model(self.paths['marginals'])
        
        self.gen_prob = GenerationProbabilityVDJ(self.model, self.genomic_data)
        self.seq_gen = SequenceGenerationVDJ(self.model, self.genomic_data)
        
        self._init_gene_maps()
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

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

        if not v_list: v_list = self._parse_gene_file(self.paths['v_anchors']) 
        if not j_list: j_list = self._parse_gene_file(self.paths['j_anchors'])

        self.v_map = {v.split('*')[0]: i for i, v in enumerate(v_list)}
        self.j_map = {j.split('*')[0]: i for i, j in enumerate(j_list)}

    def _parse_gene_file(self, filepath):
        names = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    cleaned = line.strip().replace(';', ',').replace('\t', ',')
                    if not cleaned or cleaned.startswith(('#', 'gene')): continue
                    parts = cleaned.split(',')
                    if parts: names.append(parts[0].strip())
        except Exception: pass
        return names

    def get_gene_index(self, gene_name, gene_type):
        if pd.isna(gene_name): return None
        clean_name = str(gene_name).split('*')[0]
        mapping = self.v_map if gene_type == 'V' else self.j_map
        if clean_name in mapping: return mapping[clean_name]
        if not clean_name.startswith('TR'):
            retry = ('TRBV' if gene_type == 'V' else 'TRBJ') + clean_name
            if retry in mapping: return mapping[retry]
        return None

    def calculate_pgen(self, aa_seq, v_idx, j_idx):
        if v_idx is None or j_idx is None: return 0.0
        try:
            with OutputSilencer():
                return self.gen_prob.compute_aa_CDR3_pgen(aa_seq, v_idx, j_idx)
        except Exception: return 0.0

    def generate_mutated_decoys(self, row_id, target_seq, target_pgen, v_idx, j_idx, 
                                n_decoys=5, tol_log10=0.5, min_dist=2, max_iter=10000,
                                n_anchor=2, max_mutations=3):
        """
        Generates decoys using a 'Ladder' strategy: 
        It attempts the MAXIMUM mutations first. If it fails to find a Pgen match 
        after 'patience' tries, it lowers the mutation count by 1.
        """
        # Safety Checks
        if len(target_seq) < (n_anchor * 2) + 1: return []
        if target_pgen <= 0: return []

        target_log = math.log10(target_pgen)
        
        # Identify middle
        prefix = target_seq[:n_anchor]
        suffix = target_seq[-n_anchor:]
        original_middle = list(target_seq[n_anchor:-n_anchor])
        middle_len = len(original_middle)

        if middle_len < 1: return []
        
        # Cap max_mutations at the actual length of the middle section
        actual_max_muts = min(max_mutations, middle_len)
        
        # --- LADDER LOGIC INIT ---
        current_n_muts = actual_max_muts
        fails_at_current_level = 0
        patience = 50  # How many times to try a mutation count before giving up and lowering it
        # -------------------------

        found_decoys = []
        seen_seqs = set()
        seen_seqs.add(target_seq)
        
        attempts = 0

        while len(found_decoys) < n_decoys and attempts < max_iter:
            attempts += 1
            
            # 1. Mutation Logic (Fixed Count based on Ladder)
            candidate_middle = copy.copy(original_middle)
            
            # Select indices to mutate
            idxs_to_mutate = random.sample(range(middle_len), current_n_muts)
            
            for idx in idxs_to_mutate:
                curr_aa = candidate_middle[idx]
                new_aa = curr_aa
                # Force a change
                while new_aa == curr_aa:
                    new_aa = random.choice(self.amino_acids)
                candidate_middle[idx] = new_aa
            
            candidate_seq = prefix + "".join(candidate_middle) + suffix
            
            # 2. Filter Logic
            valid_candidate = True
            
            # Check 1: Duplicate
            if candidate_seq in seen_seqs: 
                valid_candidate = False
            
            # Check 2: Levenshtein (Global check)
            if valid_candidate:
                dist = Levenshtein.distance(candidate_seq, target_seq)
                if dist < min_dist:
                    valid_candidate = False
            
            # Check 3: Pgen
            match_found = False
            if valid_candidate:
                cand_pgen = self.calculate_pgen(candidate_seq, v_idx, j_idx)
                if cand_pgen > 0:
                    cand_log = math.log10(cand_pgen)
                    diff = abs(cand_log - target_log)
                    
                    if diff <= tol_log10:
                        match_found = True
                        seen_seqs.add(candidate_seq)
                        print(f"[Row {row_id}] Match {len(found_decoys)+1}/{n_decoys}: {candidate_seq} (Diff: {diff:.2f} | Muts: {current_n_muts})")
                        found_decoys.append((candidate_seq, cand_pgen))
                        
                        # SUCCESS! Reset ladder to top to try to get another high-diff decoy
                        current_n_muts = actual_max_muts
                        fails_at_current_level = 0

            # 3. Ladder Adjustment
            if not match_found:
                fails_at_current_level += 1
                # If we failed too many times at this difficulty level, lower the difficulty
                if fails_at_current_level > patience:
                    current_n_muts -= 1
                    fails_at_current_level = 0
                    print(f"[Row {row_id}] Lowering mutation count to {current_n_muts} after {patience} failed attempts.")
                    
                    # If we drop below the minimum required distance, reset to top
                    # (This prevents getting stuck at 0 mutations)
                    if current_n_muts < min_dist:
                        current_n_muts = actual_max_muts

        return found_decoys

def load_config(config_path):
    try:
        with open(config_path, 'r') as f: return yaml.safe_load(f)
    except Exception: return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    input_file = config.get('input_file', 'input.tsv')
    output_file = config.get('output_file', 'decoys_mutated.csv')
    params = config.get('parameters', {})
    
    n_decoys = params.get('decoys_per_sequence', 5)
    tol_log10 = params.get('pgen_tolerance_log10', 0.5)
    max_iter = params.get('max_scramble_iterations', 10000) # Increased default since we want it to keep trying
    anchor_len = params.get('anchor_length', 2)
    max_muts = params.get('max_mutations_per_attempt', 3)
    min_dist = params.get('levenshtein_distance_threshold', 2)

    print(f"Reading {input_file}...")
    try:
        df = pd.read_csv(input_file, sep='\t')
        if df.shape[1] < 2: df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("Input file not found."); sys.exit(1)
    
    try:
        generator = SmartDecoyGenerator(species=config.get('species', 'human'))
    except Exception as e:
        print(f"Init Failed: {e}"); sys.exit(1)
    
    results = []
    print("\n--- Starting Generation (Silent Mode) ---\n")
    
    for i, row in df.iterrows():
        seq = row.get('junction_aa')
        v_call = row.get('v_call')
        j_call = row.get('j_call')
        
        if pd.isna(seq) or pd.isna(v_call) or pd.isna(j_call): continue
            
        v_idx = generator.get_gene_index(v_call, 'V')
        j_idx = generator.get_gene_index(j_call, 'J')
        
        if v_idx is None or j_idx is None: continue

        # Note: Removed the "Processing Row..." print here.
        target_pgen = generator.calculate_pgen(seq, v_idx, j_idx)
        
        decoys = generator.generate_mutated_decoys(
            row_id=i, # Passed row ID for logging
            target_seq=seq, target_pgen=target_pgen,
            v_idx=v_idx, j_idx=j_idx,
            n_decoys=n_decoys, tol_log10=tol_log10,
            min_dist=min_dist, max_iter=max_iter,
            n_anchor=anchor_len, max_mutations=max_muts
        )
        
        if not decoys:
             results.append({
                 'target_id': i, 'target_aa': seq, 'target_pgen': target_pgen,
                 'v_call': v_call, 'j_call': j_call,
                 'decoy_rank': 0, 'decoy_aa': None, 'decoy_pgen': None, 'pgen_diff_log10': None
             })
        else:
            for k, (d_seq, d_pgen) in enumerate(decoys):
                results.append({
                    'target_id': i, 'target_aa': seq, 'target_pgen': target_pgen,
                    'v_call': v_call, 'j_call': j_call,
                    'decoy_rank': k+1, 'decoy_aa': d_seq, 'decoy_pgen': d_pgen,
                    'pgen_diff_log10': abs(math.log10(d_pgen) - math.log10(target_pgen))
                })

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"\nDone. Saved to {output_file}.")

if __name__ == "__main__":
    main()