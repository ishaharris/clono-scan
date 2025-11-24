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

        if not v_list or not j_list:
            raise AttributeError("Could not determine gene names.")

        self.v_map = {v.split('*')[0]: i for i, v in enumerate(v_list)}
        self.j_map = {j.split('*')[0]: i for i, j in enumerate(j_list)}
        self.v_list = v_list
        self.j_list = j_list

    def _parse_gene_file(self, filepath):
        names = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('gene_name'): continue
                    cleaned = line.replace(';', ',').replace('\t', ',')
                    parts = cleaned.split(',')
                    if parts: names.append(parts[0].strip())
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

    def get_gene_name(self, index, gene_type):
        try:
            return self.v_list[index] if gene_type == 'V' else self.j_list[index]
        except (IndexError, TypeError):
            return "Unknown"

    def calculate_pgen(self, aa_seq):
        """
        Calculates the Marginal Pgen (Probability of generating this AA sequence
        summed over all possible V/J/Recombination events).
        This is the standard metric for 'Generation Probability'.
        """
        try:
            return self.gen_prob.compute_aa_cdr3_pgen(aa_seq)
        except (TypeError, AttributeError):
            try:
                return self.gen_prob.compute_aa_CDR3_pgen(aa_seq, None, None)
            except Exception:
                return 0.0

    def find_decoy_with_diagnostics(self, target_v_idx, target_j_idx, target_len, target_pgen, 
                                   max_attempts=1000, tol_log10=0.5, validated_seqs=set(), min_dist=2):
        
        fail_gene_strict = 0
        fail_len = 0
        fail_pgen = 0
        collision_count = 0
        
        target_log = math.log10(target_pgen) if target_pgen > 0 else -50
        relaxed_candidate = None

        for attempt_idx in range(int(max_attempts)):
            try:
                item = self.seq_gen.gen_rnd_prod_CDR3()
            except TypeError:
                continue

            if len(item) == 4: _, cdr3, v, j = item
            else: cdr3, v, j = item[0], item[1], item[2]

            # 1. CRITICAL V CHECK (V must always match)
            if v != target_v_idx:
                fail_gene_strict += 1
                continue
            
            # 2. Check Length
            if len(cdr3) != target_len:
                fail_len += 1
                continue

            # 3. Check Pgen (Calculated using the generated sequence)
            pgen = self.calculate_pgen(cdr3)
            if pgen <= 0:
                fail_pgen += 1
                continue
                
            log_diff = abs(math.log10(pgen) - target_log)
            if log_diff > tol_log10:
                fail_pgen += 1
                continue

            # 4. Check Collision
            collision = False
            for val_seq in validated_seqs:
                if abs(len(cdr3) - len(val_seq)) >= min_dist: continue
                if Levenshtein.distance(cdr3, val_seq) < min_dist:
                    collision = True; break
            
            if collision:
                collision_count += 1
                continue

            # --- SUCCESS ---
            
            # Case A: Strict Match
            if j == target_j_idx:
                return {
                    'success': True,
                    'decoy': cdr3,
                    'decoy_pgen': pgen,
                    'decoy_v_idx': v,
                    'decoy_j_idx': j,
                    'attempts': attempt_idx + 1,
                    'match_type': 'Strict (V+J)',
                    'bottleneck': 'None'
                }
            
            # Case B: Relaxed Match (Backup)
            if relaxed_candidate is None:
                relaxed_candidate = {
                    'success': True,
                    'decoy': cdr3,
                    'decoy_pgen': pgen,
                    'decoy_v_idx': v,
                    'decoy_j_idx': j,
                    'attempts': attempt_idx + 1,
                    'match_type': 'Relaxed (V only)',
                    'bottleneck': 'Relaxed J fallback'
                }

        # End of loop
        if relaxed_candidate is not None:
            return relaxed_candidate

        # Failure
        total_len_ok = max_attempts - fail_gene_strict - fail_len
        reason = "Unknown"
        if fail_gene_strict / max_attempts > 0.95: reason = "V Gene Rare"
        elif (fail_len / (max_attempts - fail_gene_strict + 1)) > 0.95: reason = "Length Mismatch"
        elif total_len_ok > 0 and (fail_pgen / total_len_ok > 0.95): reason = "Pgen Mismatch"
        elif collision_count > 0: reason = "Collision"

        return {
            'success': False,
            'decoy': None, 'decoy_pgen': None,
            'decoy_v_idx': None, 'decoy_j_idx': None,
            'match_type': 'None',
            'attempts': max_attempts,
            'bottleneck': reason,
            'stats': f"GV:{fail_gene_strict}|L:{fail_len}|P:{fail_pgen}"
        }

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
    output_file = config.get('output_file', 'diagnostic_decoys.csv')
    params = config.get('parameters', {})
    
    max_samples = int(params.get('diagnostic_samples', 20000))
    max_input_seqs_raw = params.get('max_input_sequences', None)
    L_dist = int(params.get('levenshtein_distance_threshold', 1))
    pgen_tol = float(params.get('pgen_tolerance_log10', 0.5))

    print(f"Reading {input_file}...")
    try:
        df = pd.read_csv(input_file, sep='\t')
        if df.shape[1] < 2: df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    if max_input_seqs_raw is not None:
        try:
            limit = int(max_input_seqs_raw)
            if limit > 0:
                print(f"Limiting input to the first {limit} sequences.")
                df = df.head(limit)
        except ValueError:
            pass
    
    try:
        generator = SmartDecoyGenerator(species=config.get('species', 'human'))
    except Exception as e:
        print(f"Generator Initialization Failed: {e}")
        sys.exit(1)
    
    print(f"\n--- Starting Diagnostic Generation ---")
    print(f"Attempts per sequence: {max_samples}")
    print("Looping through input list...")
    
    validated_seqs = set(df['junction_aa'].dropna().unique())
    results = []

    df['v_idx'] = df['v_call'].apply(lambda x: generator.get_gene_index(x, 'V'))
    df['j_idx'] = df['j_call'].apply(lambda x: generator.get_gene_index(x, 'J'))

    start_time = time.time()

    for idx, row in df.iterrows():
        v_idx = row['v_idx']
        j_idx = row['j_idx']
        seq = row['junction_aa']
        
        if pd.isna(v_idx) or pd.isna(j_idx) or not isinstance(seq, str):
            results.append({
                'target_id': idx, 'target_aa': seq, 'status': 'Invalid Input'
            })
            continue

        # --- KEY CHANGE: Calculate Pgen for Input Sequence ---
        target_pgen = generator.calculate_pgen(seq)
        
        # --- KEY CHANGE: Calculate Theoretical Attempts ---
        if target_pgen > 0:
            theoretical_attempts = 1.0 / target_pgen
        else:
            theoretical_attempts = float('inf')

        diag = generator.find_decoy_with_diagnostics(
            target_v_idx=v_idx, target_j_idx=j_idx, target_len=len(seq), target_pgen=target_pgen,
            max_attempts=max_samples, tol_log10=pgen_tol, validated_seqs=validated_seqs, min_dist=L_dist + 1
        )
        
        # --- PREPARE OUTPUT ROW ---
        res_row = {
            'target_id': idx,
            'target_aa': seq,
            
            # The Mathematical Evidence
            'target_pgen': target_pgen,
            'theoretical_attempts_needed': theoretical_attempts,
            
            'target_v_call': row['v_call'],
            'target_j_call': row['j_call'],
            
            'status': 'Found' if diag['success'] else 'Failed',
            'match_type': diag['match_type'],
            'bottleneck': diag['bottleneck'],
            
            'decoy_aa': diag['decoy'],
            'decoy_pgen': diag['decoy_pgen'], 
            'attempts_used': diag['attempts'],
        }
        
        if diag['success'] and diag['decoy_v_idx'] is not None:
            res_row['decoy_v_call'] = generator.get_gene_name(diag['decoy_v_idx'], 'V')
            res_row['decoy_j_call'] = generator.get_gene_name(diag['decoy_j_idx'], 'J')
        else:
            res_row['decoy_v_call'] = None
            res_row['decoy_j_call'] = None

        results.append(res_row)

        if idx % 10 == 0:
            stat_msg = diag['match_type'] if diag['success'] else diag['bottleneck']
            print(f"Processed {idx+1}/{len(df)} | Last: {stat_msg}", end='\r')

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"\n\nDone. Saved to {output_file}.")

if __name__ == "__main__":
    main()