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
        
        # 5. Map Gene Names (Robust Fallback Added)
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
        """Extracts V/J names, falling back to parsing the anchor files if attributes are missing."""
        v_list = None
        j_list = None
        
        # 1. Try Attribute Access
        if hasattr(self.genomic_data, 'V_names'): v_list = self.genomic_data.V_names
        elif hasattr(self.genomic_data, 'V_segments'): v_list = self.genomic_data.V_segments
        
        if hasattr(self.genomic_data, 'J_names'): j_list = self.genomic_data.J_names
        elif hasattr(self.genomic_data, 'J_segments'): j_list = self.genomic_data.J_segments

        # 2. Fallback: Parse the files directly
        if not v_list:
            print("  - V_names attribute missing. Parsing V_anchor file directly...")
            v_list = self._parse_gene_file(self.paths['v_anchors'])
            
        if not j_list:
            print("  - J_names attribute missing. Parsing J_anchor file directly...")
            j_list = self._parse_gene_file(self.paths['j_anchors'])

        if not v_list or not j_list:
            raise AttributeError("Could not determine gene names from attributes or files.")

        self.v_map = {v.split('*')[0]: i for i, v in enumerate(v_list)}
        self.j_map = {j.split('*')[0]: i for i, j in enumerate(j_list)}

    def _parse_gene_file(self, filepath):
        """Parses gene names from CSV/TXT anchor files."""
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

    def calculate_pgen(self, aa_seq, v_idx=None, j_idx=None):
        try:
            return self.gen_prob.compute_aa_cdr3_pgen(aa_seq)
        except (TypeError, AttributeError):
            try:
                return self.gen_prob.compute_aa_CDR3_pgen(aa_seq, None, None)
            except Exception:
                return 0.0

    def prime_pool(self, unique_combinations, min_candidates=100):
        print(f"Priming pool for {len(unique_combinations)} unique V-J-L combinations...")
        needed_keys = set(unique_combinations)
        total_generated = 0
        total_stored = 0
        
        # Cap generation - increased to find rare sequences
        max_total_gen = min(min_candidates * len(unique_combinations) * 200, 50000000)
        
        print(f"Generating optimized background pool (Cap: {max_total_gen} seqs)...")
        
        report_step = 5000 
        
        while total_generated < max_total_gen:
            for _ in range(report_step):
                if total_generated >= max_total_gen: break
                
                try:
                    item = self.seq_gen.gen_rnd_prod_CDR3()
                except TypeError:
                    item = self.seq_gen.gen_rnd_prod_CDR3()
                
                if len(item) == 4:
                    _, cdr3, v, j = item
                else:
                    cdr3, v, j = item[0], item[1], item[2]

                l = len(cdr3)
                key = (v, j, l)
                
                if key in needed_keys:
                    pgen = self.calculate_pgen(cdr3, v, j)
                    if pgen > 0:
                        self.pool[key].append((pgen, cdr3))
                        total_stored += 1
                
                total_generated += 1
            
            print(f"Generated {total_generated} sequences... (Stored {total_stored} candidates)", end='\r')

        print("")
        for key in self.pool:
            self.pool[key].sort(key=lambda x: x[0]) 

    def find_pgen_matched_decoys(self, target_pgen, v_idx, j_idx, length, 
                                 n_decoys=5, tol_log10=0.5, 
                                 all_validated_seqs=set(), min_dist=2):
        key = (v_idx, j_idx, length)
        candidates = self.pool[key]
        if not candidates: return []

        pgens = [x[0] for x in candidates]
        idx = bisect.bisect_left(pgens, target_pgen)
        found_decoys = []
        
        search_radius = n_decoys * 20 
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
    parser.add_argument('--config', default='config.yaml')
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
    required_combos = set()
    row_metadata = [] 
    valid_indices = []
    
    for idx, row in df.iterrows():
        v_idx = generator.get_gene_index(row['v_call'], 'V')
        j_idx = generator.get_gene_index(row['j_call'], 'J')
        seq = row['junction_aa']
        
        if v_idx is not None and j_idx is not None and isinstance(seq, str):
            required_combos.add((v_idx, j_idx, len(seq)))
            row_metadata.append({'v': v_idx, 'j': j_idx, 'seq': seq})
            valid_indices.append(idx)
        else:
            row_metadata.append(None)

    generator.prime_pool(required_combos, min_candidates=params.get('pool_prime_min_candidates', 50))
    validated_seqs = set(df['junction_aa'].dropna().unique())
    results = []
    L = params.get('levenshtein_distance_threshold', 1)

    print("Finding matched decoys...")
    start_time = time.time()
    for i, real_idx in enumerate(valid_indices):
        meta = row_metadata[real_idx]
        target_aa = meta['seq']
        target_pgen = generator.calculate_pgen(target_aa, meta['v'], meta['j'])
        
        decoys = generator.find_pgen_matched_decoys(
            target_pgen, meta['v'], meta['j'], len(target_aa),
            n_decoys=params.get('decoys_per_sequence', 5),
            tol_log10=params.get('pgen_tolerance_log10', 1.0),
            all_validated_seqs=validated_seqs,
            min_dist=L + 1
        )
        
        # --- FIXED LOGIC: Output row even if no decoys found ---
        if not decoys:
             # Append a placeholder row so the sequence isn't lost
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
        if i % 50 == 0: print(f"Processed {i}/{len(valid_indices)}...", end='\r')

    # Save Output
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"\nDone. Saved to {output_file}. Total time: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    main()