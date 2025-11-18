import pandas as pd
import os
import glob
import re

class RepertoirePipeline:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # 1. CONFIGURATION
        # Columns strictly required for immune analysis
        self.seq_cols = [
            'nucleotide', 'aminoAcid', 'count (reads)', 
            'frequencyCount (%)', 'vGeneName', 'jGeneName'
        ]
        
        # If metadata exists inside the TSV, map it to a standard name here
        # Format: {'Old_Name_In_TSV': 'New_Standard_Name'}
        self.internal_meta_map = {
            'sample_tags': 'clinical_tag',
            'guest_lab_id': 'lab_id'
        }

    def extract_sample_id(self, filename):
        # CUSTOMIZE THIS: Regex to pull ID from filename
        # Example: "2023_05_Patient_A12_TCR.tsv" -> "Patient_A12"
        match = re.search(r'(Patient_[A-Z0-9]+)', filename)
        if match:
            return match.group(1)
        return os.path.splitext(filename)[0] # Fallback to whole filename

    def process_files(self, external_metadata_path=None):
        # Load external metadata if provided
        meta_df = None
        if external_metadata_path:
            meta_df = pd.read_csv(external_metadata_path)
            # Ensure sample_id is string for merging
            meta_df['sample_id'] = meta_df['sample_id'].astype(str)

        files = glob.glob(os.path.join(self.input_dir, '*.tsv'))
        
        print(f"Processing {len(files)} files...")

        for file_path in files:
            filename = os.path.basename(file_path)
            sample_id = self.extract_sample_id(filename)
            
            # 1. Read TSV (Sequence columns + potential internal metadata columns)
            # We read slightly more columns than we need to check for internal metadata
            try:
                # Read only necessary columns to save memory
                # We assume the standard cols exist, plus we check for mapped cols
                df_iter = pd.read_csv(file_path, sep='\t', chunksize=10000)
                
                for chunk in df_iter:
                    # Keep only sequence cols that exist in this file
                    valid_seq_cols = [c for c in self.seq_cols if c in chunk.columns]
                    df_clean = chunk[valid_seq_cols].copy()
                    
                    # 2. HANDLE METADATA INSIDE TSV
                    for tsv_col, std_col in self.internal_meta_map.items():
                        if tsv_col in chunk.columns:
                            # Renaming to standard name
                            df_clean[std_col] = chunk[tsv_col]
                    
                    # 3. INJECT SAMPLE ID (The Link Key)
                    df_clean['sample_id'] = sample_id
                    
                    # 4. OPTIONAL: INJECT EXTERNAL METADATA
                    # Only do this if you want "Baked In" metadata. 
                    # Usually, it is better to join later (see next section).
                    if meta_df is not None:
                        # Merge specific row metadata if needed
                        pass 

                    # 5. OPTIMIZE TYPES
                    cats = ['vGeneName', 'jGeneName', 'aminoAcid']
                    for c in cats:
                        if c in df_clean.columns:
                            df_clean[c] = df_clean[c].astype('category')

                    # 6. SAVE TO PARQUET (Partitioned by Sample ID)
                    # Partitioning by ID makes it easy to grab specific patients later
                    save_path = os.path.join(self.output_dir, f"sample_id={sample_id}")
                    os.makedirs(save_path, exist_ok=True)
                    
                    # We append chunks to the partition
                    output_file = os.path.join(save_path, f"{filename.replace('.tsv','')}.parquet")
                    df_clean.to_parquet(output_file, index=False)
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        print("Pipeline Finished.")

# --- RUNNING IT ---
pipeline = RepertoirePipeline(input_dir='raw_data/', output_dir='processed_dataset/')
pipeline.process_files() # Run without external meta to keep it clean