import pandas as pd
import os
import glob
import yaml 
from collections import defaultdict

class RepertoirePipeline:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.input_dir = self.config['paths']['input_dir']
        self.output_dir = self.config['paths']['output_dir']

    def extract_sample_id(self, filename):
        # Logic to get ID from filename
        return os.path.splitext(filename)[0]

    def process_files(self):
        # 1. Find Files
        file_pattern = self.config['parsing']['file_extension']
        files = glob.glob(os.path.join(self.input_dir, file_pattern))
        
        print(f"Found {len(files)} files in {self.input_dir}...")

        # 2. Process Each File
        for file_path in files:
            filename = os.path.basename(file_path)
            sample_id = self.extract_sample_id(filename)
            
            # --- NEW: Logic to check and skip ---
            # Define the expected output path for this sample
            save_path = os.path.join(self.output_dir, f"sample_id={sample_id}")
            
            # Check if it exists
            if os.path.exists(save_path):
                print(f"Skipping: {filename} (Output folder {save_path} already exists)")
                continue
            # ------------------------------------
            
            print(f"\nProcessing: {filename} -> ID: {sample_id}")
            
            # --- Initialize counters for progress tracking ---
            total_rows_processed = 0 
            report_interval = 20000 
            
            try:
                chunk_size = self.config['parsing']['chunk_size']
                df_iter = pd.read_csv(file_path, sep='\t', chunksize=chunk_size)
                
                # Create the directory now that we know we need to write to it
                os.makedirs(save_path, exist_ok=True)
                
                chunk_counter = 0
                for chunk in df_iter:
                    
                    # --- PROGRESS CHECK ---
                    total_rows_processed += len(chunk)
                    if total_rows_processed % report_interval < chunk_size:
                        print(f" Rows processed for {sample_id}: {total_rows_processed:,}")
                    
                    # A. Determine Columns to Load
                    cols_to_use = self.config['columns']['keep'].copy()
                    meta_config = self.config['metadata_parsing']
                    meta_col = meta_config['col_name']
                    if meta_col not in cols_to_use:
                        cols_to_use.append(meta_col)

                    valid_cols = [c for c in cols_to_use if c in chunk.columns]
                    df_clean = chunk[valid_cols].copy()
                    
                    # B. Inject Sample ID
                    df_clean['sample_id'] = sample_id

                    # C. Parse The Long Metadata String
                    if meta_col in df_clean.columns:
                        tag_string = df_clean[meta_col].iloc[0]
                        
                        if pd.notna(tag_string) and isinstance(tag_string, str):
                            delimiter = meta_config['delimiter']
                            kv_sep = meta_config['kv_sep']
                            
                            parsed_data = defaultdict(list)
                            items = tag_string.split(delimiter)
                            
                            for item in items:
                                if kv_sep in item:
                                    key, val = item.split(kv_sep, 1)
                                    k_clean, v_clean = key.strip(), val.strip()
                                    if k_clean and v_clean:
                                        parsed_data[k_clean].append(v_clean)
                            
                            for k, v_list in parsed_data.items():
                                df_clean[k] = "; ".join(v_list)

                        df_clean.drop(columns=[meta_col], inplace=True)

                    # D. Rename Columns (Standardize)
                    rename_map = self.config['columns'].get('rename_map', {})
                    if rename_map:
                        df_clean.rename(columns=rename_map, inplace=True)

                    # E. Optimize Types (Categoricals)
                    cat_cols = self.config['types']['categoricals']
                    for c in cat_cols:
                        if c in df_clean.columns:
                            df_clean[c] = df_clean[c].astype('category')

                    # F. Save to Parquet
                    base_name = os.path.splitext(filename)[0]
                    output_file = os.path.join(save_path, f"{base_name}_part{chunk_counter}.parquet")
                    
                    df_clean.to_parquet(output_file, index=False, engine='pyarrow')
                    chunk_counter += 1
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                # Optional: Remove the folder if processing failed so it retries next time
                # import shutil
                # if os.path.exists(save_path): shutil.rmtree(save_path)
                
            print(f"File {sample_id} processing complete. Total rows: {total_rows_processed:,}")

        print("\nPipeline Finished.")

if __name__ == "__main__":
    pipeline = RepertoirePipeline("config.yaml") 
    pipeline.process_files()