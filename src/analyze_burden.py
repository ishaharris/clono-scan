import os
import pandas as pd
import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from statsmodels.stats.multitest import multipletests

# --- CONFIGURATION ---
# Replace these with your actual file paths
METADATA_FILE = "/Users/ishaharris/Projects/TCR/TCR-Isha/data/Repertoires/Cohort01_whole_metadata.tsv" 
MATRIX_FILE = "/Users/ishaharris/Projects/TCR/clono-scan/data/burden/AC02_Emerson_burden_matrix.csv"
OUTPUT_DIR = "results_plots"

def load_and_prep_data(matrix_path, metadata_path):
    print("📂 [1/5] Loading and Aligning Data...")
    
    # 1. Load Ratio Matrix with Polars for speed
    df_ratios = pl.read_csv(matrix_path)
    seq_names = df_ratios["hc_seq"].to_list()
    patient_ids = df_ratios.columns[1:]
    
    # 2. Transpose and Log-Transform
    # Using np.log1p handles outliers and zero-inflation
    print(f"   - Processing {len(patient_ids)} patients and {len(seq_names)} sequences.")
    data_values = df_ratios.drop("hc_seq").to_numpy().T
    X_df = pd.DataFrame(np.log1p(data_values), index=patient_ids, columns=seq_names)
    X_df = X_df.fillna(0.0)

    # 3. Load Metadata
    meta = pd.read_csv(metadata_path, sep="\t")
    filtered = meta[
        meta['sample_name'].str.startswith('P') &
        meta['sample_tags'].str.contains(r'Cytomegalovirus\s+[+-]', case=False, na=False)
    ]
    
    y_map = {row['sample_name']: (1 if 'Cytomegalovirus +' in row['sample_tags'] else 0) 
             for _, row in filtered.iterrows()}

    # 4. Align Matrix with Labels
    common_ids = [pid for pid in X_df.index if pid in y_map]
    X_final = X_df.loc[common_ids]
    y_final = np.array([y_map[pid] for pid in common_ids])
    
    print(f"✅ Data Aligned: {len(common_ids)} patients matching labels.")
    return X_final, y_final

def plot_pca(X, y, save_path):
    print("🎨 [2/5] Generating PCA Plot...")
    # Standardization is critical before PCA
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette={0: 'blue', 1: 'red'}, style=y, s=100)
    plt.title(f'PCA: Global separation by CMV Status\nExplained Var: {pca.explained_variance_ratio_.sum():.2%}')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   - Saved: {save_path}")

def run_differential_abundance(X, y, csv_path, plot_path):
    print("🔬 [3/5] Performing Differential Abundance (Volcano Plot)...")
    results = []
    pos_group = X[y == 1]
    neg_group = X[y == 0]
    
    for seq in X.columns:
        u, p = mannwhitneyu(pos_group[seq], neg_group[seq], alternative='two-sided')
        results.append({'sequence': seq, 'p_value': p, 'log_diff': pos_group[seq].mean() - neg_group[seq].mean()})
        
    res_df = pd.DataFrame(results)
    # Benjamini-Hochberg FDR correction
    res_df['p_adj'] = multipletests(res_df['p_value'], method='fdr_bh')[1]
    res_df.to_csv(csv_path, index=False)
    
    # Volcano Plot
    plt.figure(figsize=(10, 6))
    colors = res_df['p_adj'].apply(lambda x: 'red' if x < 0.05 else 'grey')
    plt.scatter(res_df['log_diff'], -np.log10(res_df['p_adj']), c=colors, alpha=0.5, s=10)
    plt.axhline(-np.log10(0.05), color='black', linestyle='--')
    plt.title('Volcano Plot: Significant CMV sequences')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"   - Stats saved to {csv_path}")

def run_lasso_model(X, y):
    print("🤖 [4/5] Starting Lasso Logistic Regression Pipeline...")
    
    # 1. Feature Pre-filtering
    print("   [STEP 1] Filtering top 10% most informative features to save time...")
    selector = SelectPercentile(f_classif, percentile=10)
    X_reduced = selector.fit_transform(X, y)
    feat_names = X.columns[selector.get_support()]
    print(f"   ---> Reduced feature set from {X.shape[1]} to {X_reduced.shape[1]} sequences.")

    # 2. Standardization
    print("   [STEP 2] Standardizing features...")
    X_scaled = StandardScaler().fit_transform(X_reduced)
    
    # 3. Model Training
    print("   [STEP 3] Running 3-fold Cross-Validation with SAGA solver...")
    print("   (This uses all available CPU cores via n_jobs=-1)")
    
    clf = LogisticRegressionCV(
        Cs=10,               # Number of regularization strengths to test
        cv=5,               # 3-fold CV for speed
        penalty='l1', 
        solver='saga',      # Faster for large datasets with L1
        max_iter=1000,
        tol=1e-2,           # Slightly higher tolerance for speed
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42
    )
    
    clf.fit(X_scaled, y)
    
    print(f"   [STEP 4] Model Fit Complete. Average AUC: {clf.scores_[1].mean():.3f}")
    
    # 4. Extracting Results
    coefs = pd.Series(clf.coef_[0], index=feat_names)
    important = coefs[coefs != 0].sort_values(ascending=False)
    print(f"   [STEP 5] Extracted {len(important)} sequences with non-zero predictive value.")
    
    return important

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        X, y = load_and_prep_data(MATRIX_FILE, METADATA_FILE)
        
        plot_pca(X, y, os.path.join(OUTPUT_DIR, "pca_log_transformed.png"))
        
        run_differential_abundance(
            X, y, 
            os.path.join(OUTPUT_DIR, "differential_stats.csv"),
            os.path.join(OUTPUT_DIR, "volcano_plot.png")
        )
        
        biomarkers = run_lasso_model(X, y)
        
        print("\n🏆 [5/5] Finalizing Results...")
        if not biomarkers.empty:
            biomarkers.to_csv(os.path.join(OUTPUT_DIR, "predictive_biomarkers.csv"))
            print(f"✅ Success! Found {len(biomarkers)} predictive sequences.")
            print(f"Top Predictor: {biomarkers.index[0]} (Coef: {biomarkers.iloc[0]:.4f})")
        else:
            print("⚠️ No predictive sequences found (Regularization too high).")

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")