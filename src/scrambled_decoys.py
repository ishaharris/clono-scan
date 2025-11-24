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
        # Flush Python buffers first to ensure order
        sys.stdout.flush()
        sys.stderr.flush()

        # Open null device
        self._devnull = os.open(os.devnull, os.O_WRONLY)

        # Save original file descriptors
        try:
            self._original_stdout_fd = os.dup(sys.stdout.fileno())
            self._original_stderr_fd = os.dup(sys.stderr.fileno())
        except Exception:
            # If running in an environment without real FDs (e.g. some IDE consoles),
            # we cannot suppress C-level output. Just pass.
            return

        # Redirect stdout/stderr to devnull
        os.dup2(self._devnull, sys.stdout.fileno())
        os.dup2(self._devnull, sys.stderr.fileno())

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._original_stdout_fd is None:
            return

        # Flush low-level buffers
        sys.stdout.flush()
        sys.stderr.flush()

        # Restore original file descriptors
        os.dup2(self._original_stdout_fd, sys.stdout.fileno())
        os.dup2(self._original_stderr_fd, sys.stderr.fileno())

        # Clean up
        os.close(self._original_stdout_fd)
        os.close(self._original_stderr_fd)
        os.close(self._devnull)

class SmartDecoyGenerator:
    def __init__(self, species='human', chain='TRB'):
        print(f"Initializing OLGA for {species} {chain}...")
        
        # 1. Locate Files
        self.paths = self._locate_model_files(species, chain)
        
        # 2. Load Genomic Data
        print("  - Loading Genomic Data...")
        self.genomic_data = GenomicDataVDJ()
        self.genomic_data.load_igor_genomic_data(
            self.paths['params'], 
            self.paths['v_anchors'], 
            self.paths['j_anchors']
        )
        
        # 3. Initialize Generative Model
        print("  - Loading Generative Model...")
        self.model = GenerativeModelVDJ()
        self.model.load_and_process_igor_model(self.paths['marginals'])
        
        # 4. Initialize Engines
        self.gen_prob = GenerationProbabilityVDJ(self.model, self.genomic_data)
        self.seq_gen = SequenceGenerationVDJ(self.model, self.genomic_data)
        
        # 5. Map Gene Names
        self._init_gene_maps()

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
        except Exception as e:
            print(f"    Warning: Failed to parse {filepath}: {e}")
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

    def calculate_pgen(self, aa_seq, v_idx, j_idx):
        """
        Calculates Pgen forcing the specific V and J indices.
        Silences OLGA's C++ 'Unfamiliar typed usage mask' warnings.
        """
        if v_idx is None or j_idx is None:
            return 0.0
        
        try:
            with OutputSilencer():
                val = self.gen_prob.compute_aa_CDR3_pgen(aa_seq, v_idx, j_idx)
            return val
        except Exception:
            return 0.0

    def generate_scrambled_decoys(self, target_seq, target_pgen, v_idx, j_idx, 
                                  n_decoys=5, tol_log10=0.5, min_dist=2, max_iter=500,
                                  n_anchor=1):
        """
        Scrambles central amino acids. 
        n_anchor: Number of residues to keep fixed at start and end.
        """
        # Sanity check: Sequence must be long enough to have a middle to scramble
        if len(target_seq) < (n_anchor * 2) + 2:
            actual_anchor = 1
            if len(target_seq) < 4: 
                 print(f"    [Skip] Sequence too short to scramble: {target_seq}")
                 return []
        else:
            actual_anchor = n_anchor

        if target_pgen <= 0:
            print(f"    [Skip] Target Pgen is 0, cannot match.")
            return []

        target_log = math.log10(target_pgen)
        
        # Slice based on anchors
        prefix = target_seq[:actual_anchor]
        suffix = target_seq[-actual_anchor:]
        middle_chars = list(target_seq[actual_anchor:-actual_anchor])

        found_decoys = []
        seen_seqs = set()
        seen_seqs.add(target_seq)
        
        attempts = 0
        
        print(f"    > Target: {target_seq} (log10 Pgen: {target_log:.2f}) | Anchors: {actual_anchor}")
        
        while len(found_decoys) < n_decoys and attempts < max_iter:
            attempts += 1
            
            # 1. Scramble
            random.shuffle(middle_chars)
            candidate_seq = prefix + "".join(middle_chars) + suffix
            
            # 2. Basic Filters
            if candidate_seq in seen_seqs:
                continue
            seen_seqs.add(candidate_seq)
            
            # 3. Levenshtein Check
            dist = Levenshtein.distance(candidate_seq, target_seq)
            if dist < min_dist:
                continue
                
            # 4. Calculate Pgen
            cand_pgen = self.calculate_pgen(candidate_seq, v_idx, j_idx)
            
            if cand_pgen <= 0:
                continue
                
            cand_log = math.log10(cand_pgen)
            diff = abs(cand_log - target_log)
            
            # 5. Check Tolerance
            if diff <= tol_log10:
                print(f"      [Match {len(found_decoys)+1}] {candidate_seq} | Pgen: {cand_pgen:.2e} (Diff: {diff:.2f}) | Iter: {attempts}")
                found_decoys.append((candidate_seq, cand_pgen))
            else:
                if attempts % 100 == 0:
                    print(f"      [Searching...] Iter {attempts}: {candidate_seq} (Diff: {diff:.2f})")

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
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print("Config file not found. Using defaults/dummy values for testing.")
        config = {}
    else:
        config = load_config(args.config)
    
    input_file = config.get('input_file', 'input.tsv')
    output_file = config.get('output_file', 'decoys_scrambled.csv')
    params = config.get('parameters', {})
    
    # Default Params
    n_decoys = params.get('decoys_per_sequence', 5)
    tol_log10 = params.get('pgen_tolerance_log10', 2.0)
    max_scramble_iter = params.get('max_scramble_iterations', 2000)
    anchor_len = params.get('anchor_length', 3) # Default to 3 if not in config
    
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
    
    results = []
    L = params.get('levenshtein_distance_threshold', 1)

    print("\n--- Starting Scramble Generation ---\n")
    start_time = time.time()
    
    for i, row in df.iterrows():
        # Extract info
        seq = row.get('junction_aa')
        v_call = row.get('v_call')
        j_call = row.get('j_call')
        
        if pd.isna(seq) or pd.isna(v_call) or pd.isna(j_call):
            continue
            
        v_idx = generator.get_gene_index(v_call, 'V')
        j_idx = generator.get_gene_index(j_call, 'J')
        
        if v_idx is None or j_idx is None:
            print(f"Skipping Row {i}: Could not map V/J genes ({v_call}, {j_call})")
            continue

        print(f"Processing Row {i}: {seq}")
        
        # Calculate Target Pgen
        target_pgen = generator.calculate_pgen(seq, v_idx, j_idx)
        
        # Generate Decoys via Scrambling
        decoys = generator.generate_scrambled_decoys(
            target_seq=seq,
            target_pgen=target_pgen,
            v_idx=v_idx,
            j_idx=j_idx,
            n_decoys=n_decoys,
            tol_log10=tol_log10,
            min_dist=L + 1,
            max_iter=max_scramble_iter,
            n_anchor=anchor_len  # Pass the anchor length here
        )
        
        if not decoys:
             results.append({
                 'target_id': i,
                 'target_aa': seq,
                 'target_pgen': target_pgen,
                 'v_call': v_call,
                 'j_call': j_call,
                 'decoy_rank': 0,
                 'decoy_aa': None,
                 'decoy_pgen': None,
                 'pgen_diff_log10': None
             })
        else:
            for k, (d_seq, d_pgen) in enumerate(decoys):
                results.append({
                    'target_id': i,
                    'target_aa': seq,
                    'target_pgen': target_pgen,
                    'v_call': v_call,
                    'j_call': j_call,
                    'decoy_rank': k+1,
                    'decoy_aa': d_seq,
                    'decoy_pgen': d_pgen,
                    'pgen_diff_log10': abs(math.log10(d_pgen) - math.log10(target_pgen))
                })

    # Save Output
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"\nDone. Saved to {output_file}. Total time: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    main()