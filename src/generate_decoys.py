#!/usr/bin/env python3
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
    def __init__(self, species='human', chain='TRB', debug_dir='debug_logs'):
        print(f"Initializing OLGA for {species} {chain}...")
        self.debug_dir = debug_dir
        os.makedirs(self.debug_dir, exist_ok=True)

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

        # 5. Map Gene Names (FIXED LOGIC)
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

    def _init_gene_maps(self):
        """Extracts V/J names with improved allele handling and adds debugging output."""

        v_list = None
        j_list = None

        # 1. Try Attribute Access
        if hasattr(self.genomic_data, 'V_names'):
            v_list = self.genomic_data.V_names
        elif hasattr(self.genomic_data, 'V_segments'):
            v_list = self.genomic_data.V_segments

        if hasattr(self.genomic_data, 'J_names'):
            j_list = self.genomic_data.J_names
        elif hasattr(self.genomic_data, 'J_segments'):
            j_list = self.genomic_data.J_segments

        # 2. Fallback: Parse anchor files
        if not v_list:
            print("  - V_names attribute missing. Parsing V_anchor file directly...")
            v_list = self._parse_gene_file(self.paths['v_anchors'])

        if not j_list:
            print("  - J_names attribute missing. Parsing J_anchor file directly...")
            j_list = self._parse_gene_file(self.paths['j_anchors'])

        if not v_list or not j_list:
            raise AttributeError("Could not determine gene names from attributes or files.")

        # --- FIXED MAPPING LOGIC ---
        # Build V map
        self.v_map = {}
        for i, v in enumerate(v_list):
            v_clean = v.strip()
            self.v_map[v_clean] = i    # exact match

            base = v_clean.split('*')[0]
            if base not in self.v_map:
                self.v_map[base] = i

        # Build J map
        self.j_map = {}
        for i, j in enumerate(j_list):
            j_clean = j.strip()
            self.j_map[j_clean] = i

            base = j_clean.split('*')[0]
            if base not in self.j_map:
                self.j_map[base] = i

        # --- Store lists for debugging / reverse lookup ---
        self.v_list_debug = v_list
        self.j_list_debug = j_list

        # === DEBUG OUTPUT SAVED TO FILES (instead of printing each line) ===
        v_log = os.path.join(self.debug_dir, 'debug_olga_v_list.tsv')
        j_log = os.path.join(self.debug_dir, 'debug_olga_j_list.tsv')
        with open(v_log, 'w') as vf:
            vf.write("idx\tv_name\n")
            for i, v in enumerate(self.v_list_debug):
                vf.write(f"{i}\t{v}\n")
        with open(j_log, 'w') as jf:
            jf.write("idx\tj_name\n")
            for i, j in enumerate(self.j_list_debug):
                jf.write(f"{i}\t{j}\n")
        print(f"Saved OLGA V/J lists to: {v_log}, {j_log}")

    def get_gene_index(self, gene_name, gene_type):
        """
        Returns INTEGER index for a gene name.
        """
        if pd.isna(gene_name): return None
        name_str = str(gene_name).strip()

        mapping = self.v_map if gene_type == 'V' else self.j_map

        # 1. Try Exact Match
        if name_str in mapping:
            return mapping[name_str]

        # 2. Try Base Match (strip allele)
        base_name = name_str.split('*')[0]
        if base_name in mapping:
            return mapping[base_name]

        # 3. Try adding TRBV/TRBJ prefix
        if not base_name.startswith('TR'):
            prefix = 'TRBV' if gene_type == 'V' else 'TRBJ'
            retry_name = prefix + base_name
            if retry_name in mapping:
                return mapping[retry_name]

        return None

    def calculate_pgen(self, aa_seq, v_idx=None, j_idx=None):
        # Try conditioned computation when possible
        try:
            if v_idx is not None and j_idx is not None:
                # many OLGA versions accept v/j params; attempt that first
                try:
                    return float(self.gen_prob.compute_aa_cdr3_pgen(aa_seq, v_idx, j_idx))
                except TypeError:
                    # try alternate case (older API signatures)
                    pass
            # fallback call (no v/j)
            try:
                return float(self.gen_prob.compute_aa_cdr3_pgen(aa_seq))
            except Exception:
                return float(self.gen_prob.compute_aa_CDR3_pgen(aa_seq, None, None))
        except Exception as exc:
            # log minimal error info to a file
            errfile = os.path.join(self.debug_dir, 'debug_pgen_errors.tsv')
            with open(errfile, 'a') as ef:
                ef.write(f"{aa_seq}\t{v_idx}\t{j_idx}\t{exc}\n")
            return 0.0

    def prime_pool(self, unique_combinations, min_candidates=100):
        print(">>> ENTERED prime_pool()")
        print(">>> debug_dir =", self.debug_dir)

        """
        Full debug version that logs details to files and writes a concise summary on completion.
        """
        print(f"Priming pool for {len(unique_combinations)} unique V-J-L combinations...")
        needed_keys = set(unique_combinations)
        total_generated = 0
        total_stored = 0

        # --- DIAGNOSTIC COUNTERS ---
        matches_found = 0
        pgen_zero_count = 0
        out_of_range_count = 0

        # File handles for debug logs
        samples_path = os.path.join(self.debug_dir, 'debug_generated_samples.tsv')
        first500_path = os.path.join(self.debug_dir, 'debug_first500.tsv')
        pgen_zero_path = os.path.join(self.debug_dir, 'debug_pgen_zero.tsv')
        out_of_range_path = os.path.join(self.debug_dir, 'debug_out_of_range.tsv')
        required_keys_path = os.path.join(self.debug_dir, 'debug_required_keys.tsv')
        pool_cov_path = os.path.join(self.debug_dir, 'pool_coverage.csv')

        # Open files and write headers
                # Open files and write headers (wrap in try/except to catch permission / path errors)
        try:
            samples_f = open(samples_path, 'w')
            samples_f.write("sample_idx\traw_item_repr\n")

            first500_f = open(first500_path, 'w')
            first500_f.write("gen_idx\tv_idx\tj_idx\tlen\tcdr3\tkey_in_needed\n")

            pgen_zero_f = open(pgen_zero_path, 'w')
            pgen_zero_f.write("gen_idx\tv_idx\tj_idx\tlen\tcdr3\tpgen\n")

            out_of_range_f = open(out_of_range_path, 'w')
            out_of_range_f.write("gen_idx\tv_idx\tj_idx\tlen\tcdr3\tnote\n")
        except Exception as e:
            print("ERROR: failed to open debug files in", self.debug_dir, "-", e)
            raise


        first500_f = open(first500_path, 'w')
        first500_f.write("gen_idx\tv_idx\tj_idx\tlen\tcdr3\tkey_in_needed\n")

        pgen_zero_f = open(pgen_zero_path, 'w')
        pgen_zero_f.write("gen_idx\tv_idx\tj_idx\tlen\tcdr3\tpgen\n")

        out_of_range_f = open(out_of_range_path, 'w')
        out_of_range_f.write("gen_idx\tv_idx\tj_idx\tlen\tcdr3\tnote\n")

        required_f = open(required_keys_path, 'w')
        required_f.write("v_idx\tj_idx\tlength\tv_name\tj_name\n")
        for k in sorted(needed_keys):
            vname = self.v_list_debug[k[0]] if 0 <= k[0] < len(self.v_list_debug) else "OUT_OF_RANGE"
            jname = self.j_list_debug[k[1]] if 0 <= k[1] < len(self.j_list_debug) else "OUT_OF_RANGE"
            required_f.write(f"{k[0]}\t{k[1]}\t{k[2]}\t{vname}\t{jname}\n")
        required_f.close()

        # Cap generation (safe ceiling)
        max_total_gen = min(min_candidates * len(unique_combinations) * 200, 10000000)
        print(f"Generating optimized background pool (Cap: {max_total_gen} seqs)...")
        report_step = 5000

        # Main generation loop
        while total_generated < max_total_gen:
            for _ in range(report_step):
                if total_generated >= max_total_gen: break

                try:
                    item = self.seq_gen.gen_rnd_prod_CDR3()
                except TypeError:
                    item = self.seq_gen.gen_rnd_prod_CDR3()

                # SAMPLE RAW ITEM (first N)
                if total_generated < 200:
                    samples_f.write(f"{total_generated}\t{repr(item)}\n")

                # Robust unpacking: detect common formats
                # - some OLGA versions return (prod, cdr3, v_idx, j_idx)
                # - some return (cdr3, v_idx, j_idx)
                # - some return (v_idx, j_idx, cdr3)
                # - fallback: inspect types/lengths
                v = j = None
                cdr3 = None

                if isinstance(item, (list, tuple)):
                    if len(item) == 4:
                        # variant: (something, cdr3, v, j) or (prod, cdr3, v, j)
                        # find the string-looking CDR3
                        # Heuristic: the element of type str and length > 2
                        str_idxs = [i for i, x in enumerate(item) if isinstance(x, str)]
                        if len(str_idxs) >= 1:
                            # assume first string is CDR3
                            c_idx = str_idxs[0]
                            cdr3 = item[c_idx]
                            # set v and j as the two integer-like of remaining
                            others = [item[i] for i in range(len(item)) if i != c_idx]
                            int_candidates = [x for x in others if isinstance(x, (int, np.integer))]
                            if len(int_candidates) >= 2:
                                v, j = int(int_candidates[-2]), int(int_candidates[-1])
                        else:
                            # fallback positions
                            try:
                                cdr3 = item[0]
                                v = int(item[1])
                                j = int(item[2])
                            except Exception:
                                cdr3 = str(item[0])
                    elif len(item) == 3:
                        # could be (cdr3, v, j) or (v, j, cdr3)
                        if isinstance(item[0], str):
                            cdr3, v, j = item[0], int(item[1]), int(item[2])
                        elif isinstance(item[2], str):
                            v, j, cdr3 = int(item[0]), int(item[1]), item[2]
                        else:
                            # fallback: assume (cdr3, v, j)
                            try:
                                cdr3, v, j = item[0], int(item[1]), int(item[2])
                            except Exception:
                                # last resort
                                cdr3 = str(item[0])
                                v = int(item[1]) if isinstance(item[1], (int, np.integer)) else None
                                j = int(item[2]) if isinstance(item[2], (int, np.integer)) else None
                    else:
                        # unknown structure: cast best-effort
                        # pick last two ints as v/j, first long string as cdr3
                        ints = [x for x in item if isinstance(x, (int, np.integer))]
                        strs = [x for x in item if isinstance(x, str)]
                        if strs:
                            cdr3 = strs[0]
                        if len(ints) >= 2:
                            v, j = int(ints[-2]), int(ints[-1])
                else:
                    # item not a sequence; coerce to string and skip
                    cdr3 = str(item)

                # Safe-cast indices and compute length
                try:
                    v = int(v) if v is not None else None
                except Exception:
                    v = None
                try:
                    j = int(j) if j is not None else None
                except Exception:
                    j = None
                l = int(len(cdr3)) if cdr3 is not None else None

                # detect out-of-range
                if v is None or j is None or l is None:
                    out_of_range_count += 1
                    out_of_range_f.write(f"{total_generated}\t{v}\t{j}\t{l}\t{cdr3}\tMALFORMED\n")
                    total_generated += 1
                    continue

                if v < 0 or v >= len(self.v_list_debug) or j < 0 or j >= len(self.j_list_debug):
                    out_of_range_count += 1
                    out_of_range_f.write(f"{total_generated}\t{v}\t{j}\t{l}\t{cdr3}\tOUT_OF_RANGE\n")

                key = (v, j, l)

                # If this is within the first 500, log details
                if total_generated < 500:
                    key_in_needed = key in needed_keys
                    first500_f.write(f"{total_generated}\t{v}\t{j}\t{l}\t{cdr3}\t{key_in_needed}\n")

                # If the key is needed, compute pgen and possibly store
                if key in needed_keys:
                    matches_found += 1
                    pgen = self.calculate_pgen(cdr3, v, j)
                    if pgen <= 0:
                        pgen_zero_count += 1
                        # log sample of pgen==0
                        if pgen_zero_count <= 1000000:  # avoid runaway logs
                            pgen_zero_f.write(f"{total_generated}\t{v}\t{j}\t{l}\t{cdr3}\t{pgen}\n")
                    else:
                        self.pool[key].append((pgen, cdr3))
                        total_stored += 1

                total_generated += 1

            # Dynamic Status Report (kept compact)
            sys.stdout.write(f"\rGenerated {total_generated} | Matches found {matches_found} | Stored {total_stored}")
            sys.stdout.flush()

        # Close files
        samples_f.close()
        first500_f.close()
        pgen_zero_f.close()
        out_of_range_f.close()

        # --- DIAGNOSTIC REPORT: pool coverage ---
        with open(pool_cov_path, 'w') as pf:
            pf.write("v_idx,j_idx,length,count,v_name,j_name\n")
            for k in sorted(needed_keys):
                cnt = len(self.pool.get(k, []))
                vname = self.v_list_debug[k[0]] if 0 <= k[0] < len(self.v_list_debug) else "OUT"
                jname = self.j_list_debug[k[1]] if 0 <= k[1] < len(self.j_list_debug) else "OUT"
                pf.write(f"{k[0]},{k[1]},{k[2]},{cnt},{vname},{jname}\n")

        # Small summary print for user convenience
        missing = [k for k in sorted(needed_keys) if len(self.pool.get(k, [])) == 0]
        summary = {
            'generated': total_generated,
            'matches_found': matches_found,
            'stored': total_stored,
            'pgen_zero_logged': pgen_zero_count,
            'out_of_range_logged': out_of_range_count,
            'missing_required_keys': len(missing)
        }
        print("\n\n=== prime_pool summary ===")
        print(f"Generated: {summary['generated']}")
        print(f"Matches (keys encountered while generating): {summary['matches_found']}")
        print(f"Stored candidates (pgen>0): {summary['stored']}")
        print(f"Pgen==0 logged: {summary['pgen_zero_logged']}")
        print(f"Out-of-range / malformed items logged: {summary['out_of_range_logged']}")
        print(f"Required keys with zero candidates: {summary['missing_required_keys']}")
        print(f"Debug files written to: {self.debug_dir}")
        print("Pool generation complete.\n")

        # Final: sort pools by pgen ascending for efficient bisect
        for key in self.pool:
            self.pool[key].sort(key=lambda x: x[0])

    def find_pgen_matched_decoys(self, target_pgen, v_idx, j_idx, length,
                                 n_decoys=5, tol_log10=0.5,
                                 all_validated_seqs=set(), min_dist=2):

        # FORCE INTEGERS for lookup
        key = (int(v_idx), int(j_idx), int(length))

        candidates = self.pool.get(key, [])
        if not candidates:
            # quick early-exit, but log a tiny hint file
            no_cand_path = os.path.join(self.debug_dir, 'debug_no_candidates.tsv')
            with open(no_cand_path, 'a') as nc:
                nc.write(f"{key}\n")
            return []

        pgens = [x[0] for x in candidates]
        idx = bisect.bisect_left(pgens, target_pgen)
        found_decoys = []

        search_radius = n_decoys * 20
        start = max(0, idx - search_radius)
        end = min(len(candidates), idx + search_radius)
        subset = candidates[start:end]

        target_log = math.log10(target_pgen) if target_pgen > 0 else -50
        subset.sort(key=lambda x: abs(math.log10(x[0]) - target_log) if x[0] > 0 else 999)

        reject_collision = 0
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

            if collision:
                reject_collision += 1
                continue

            found_decoys.append((seq, pgen))

        # minimal logging of rejections
        if reject_collision > 0:
            rc_path = os.path.join(self.debug_dir, 'debug_rejections.tsv')
            with open(rc_path, 'a') as rf:
                rf.write(f"{key}\trequested:{n_decoys}\trejected_collision:{reject_collision}\n")

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

        # --- read debug config block properly ---
    debug_cfg = config.get('debug', {}) or {}
    resolved_debug_dir = debug_cfg.get('debug_dir', 'debug_logs')
    # ensure resolved_debug_dir is a string and strip trailing spaces
    if not resolved_debug_dir:
        resolved_debug_dir = 'debug_logs'
    resolved_debug_dir = str(resolved_debug_dir).rstrip('/')

    print("Resolved debug_dir:", resolved_debug_dir)
    try:
        generator = SmartDecoyGenerator(
            species=config.get('species', 'human'),
            debug_dir=resolved_debug_dir
        )
    except Exception as e:
        print(f"Generator Initialization Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    except Exception as e:
        print(f"Generator Initialization Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("Analyzing input requirements...")
    required_combos = set()
    row_metadata = []
    valid_indices = []

    # mapping debug list
    mapping_rows = []

    for idx, row in df.iterrows():
        v_idx = generator.get_gene_index(row['v_call'], 'V')
        j_idx = generator.get_gene_index(row['j_call'], 'J')
        seq = row['junction_aa']

        # capture mapping info for later CSV dump
        mapping_rows.append({
            'input_idx': idx,
            'v_call_raw': row.get('v_call', ''),
            'mapped_v_idx': v_idx,
            'mapped_v_name': generator.v_list_debug[v_idx] if (v_idx is not None and 0 <= v_idx < len(generator.v_list_debug)) else None,
            'j_call_raw': row.get('j_call', ''),
            'mapped_j_idx': j_idx,
            'mapped_j_name': generator.j_list_debug[j_idx] if (j_idx is not None and 0 <= j_idx < len(generator.j_list_debug)) else None,
            'junction_aa': seq
        })

        if v_idx is not None and j_idx is not None and isinstance(seq, str):
            v_idx = int(v_idx)
            j_idx = int(j_idx)
            l_seq = int(len(seq))

            required_combos.add((v_idx, j_idx, l_seq))
            row_metadata.append({'v': v_idx, 'j': j_idx, 'seq': seq})
            valid_indices.append(idx)
        else:
            row_metadata.append(None)

    # dump mapping debug
    mapping_debug_path = os.path.join(generator.debug_dir, 'mapping_debug.csv')
    pd.DataFrame(mapping_rows).to_csv(mapping_debug_path, index=False)
    pd.DataFrame(mapping_rows).to_csv(mapping_debug_path, index=False)
    print(f"Saved mapping debug to: {mapping_debug_path}")
    print("Check existence:", os.path.exists(mapping_debug_path))


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
            tol_log10=params.get('pgen_tolerance_log10', 2.0),
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

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"\nDone. Saved to {output_file}. Total time: {time.time()-start_time:.2f}s")
    print(f"All debug logs are in: {generator.debug_dir}")


if __name__ == "__main__":
    main()
