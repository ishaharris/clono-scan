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
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.multitest import multipletests

# --- CONFIGURATION ---
MATRIX_FILE = "/Users/ishaharris/Projects/TCR/clono-scan/data/burden/AC02_Emerson_burden_matrix.csv"  # Output from your previous script
METADATA_FILE = "/Users/ishaharris/Projects/TCR/TCR-Isha/data/Repertoires/Cohort01_whole_metadata.tsv"            # Your patient metadata
OUTPUT_DIR = "/Users/ishaharris/Projects/TCR/clono-scan/data/results_plots"              # Where to save images/tables

def load_and_prep_data(matrix_path, metadata_path):
    print("📂 Loading data...")
    
    # 1. Load Ratio Matrix
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Could not find {matrix_path}")
    
    df_ratios = pl.read_csv(matrix_path)
    seq_names = df_ratios["hc_seq"].to_list()
    # Transpose: Rows=Patients, Cols=Sequences
    data_values = df_ratios.drop("hc_seq").to_numpy().T
    patient_ids = df_ratios.columns[1:]
    
    # 2. Create DataFrame & Log-Transform
    # FIX: We apply log1p (log(x+1)) immediately. 
    # This fixes the outlier plot issue and helps the Lasso model converge.
    print(f"   - Found {len(patient_ids)} patients and {len(seq_names)} sequences.")
    print("   - Applying Log1p transformation to fix outliers...")
    X_df = pd.DataFrame(np.log1p(data_values), index=patient_ids, columns=seq_names)
    X_df = X_df.fillna(0.0)

    # 3. Load Metadata
    meta = pd.read_csv(metadata_path, sep="\t") # Change sep="," if csv
    
    # Filter for relevant samples
    filtered = meta[
        meta['sample_name'].str.startswith('P') &
        meta['sample_tags'].str.contains(r'Cytomegalovirus\s+[+-]', case=False, na=False)
    ]
    
    y_map = {}
    for _, row in filtered.iterrows():
        pid = row['sample_name']
        tag = row['sample_tags']
        if 'Cytomegalovirus +' in tag:
            y_map[pid] = 1
        elif 'Cytomegalovirus -' in tag:
            y_map[pid] = 0

    # 4. Align X and y
    common_ids = [pid for pid in X_df.index if pid in y_map]
    
    if not common_ids:
        raise ValueError("No matching patient IDs found between Matrix and Metadata!")

    X_final = X_df.loc[common_ids]
    y_final = np.array([y_map[pid] for pid in common_ids])
    
    print(f"✅ Data Ready: {X_final.shape[0]} patients (Matches: {len(common_ids)}), {X_final.shape[1]} sequences.")
    print(f"   - Class Balance: {sum(y_final)} CMV+ / {len(y_final) - sum(y_final)} CMV-")
    
    return X_final, y_final

def plot_pca(X, y, save_path):
    print("🎨 Generating PCA Plot...")
    
    # PCA is sensitive to scale, so we Standardize the already Log-transformed data
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette={0: 'blue', 1: 'red'}, style=y, s=100, alpha=0.8)
    
    plt.title(f'PCA of TCR Ratios (Log-Transformed)\nExplained Variance: {pca.explained_variance_ratio_.sum():.2f}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.legend(title='CMV Status', labels=['CMV-', 'CMV+'])
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   - Saved to {save_path}")

def run_differential_abundance(X, y, save_csv_path, save_plot_path):
    print("🔬 Running Differential Abundance (Volcano Analysis)...")
    
    pos_group = X[y == 1]
    neg_group = X[y == 0]
    
    results = []
    # Loop through sequences
    # Note: X is already Log-Transformed, so difference of means = Log Fold Change
    for seq in X.columns:
        # Use Mann-Whitney (Non-parametric test)
        u_stat, p_val = mannwhitneyu(pos_group[seq], neg_group[seq], alternative='two-sided')
        
        mean_pos = pos_group[seq].mean()
        mean_neg = neg_group[seq].mean()
        
        # Since X is log-transformed, Mean(Pos) - Mean(Neg) is approximately Log2FC
        log2fc = mean_pos - mean_neg 
        
        results.append({
            'sequence': seq,
            'p_value': p_val,
            'log2fc': log2fc
        })
        
    res_df = pd.DataFrame(results)
    
    # Multiple Testing Correction (Benjamini-Hochberg)
    res_df['p_adj'] = multipletests(res_df['p_value'], method='fdr_bh')[1]
    res_df.to_csv(save_csv_path, index=False)
    print(f"   - Stats saved to {save_csv_path}")
    
    # Plotting Volcano
    plt.figure(figsize=(10, 6))
    res_df['neg_log_p'] = -np.log10(res_df['p_adj'])
    colors = res_df['p_adj'].apply(lambda x: 'red' if x < 0.05 else 'grey')
    
    plt.scatter(res_df['log2fc'], res_df['neg_log_p'], c=colors, alpha=0.6, s=15)
    plt.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=0.8, label='FDR=0.05')
    plt.axvline(0, color='black', linewidth=0.8)
    
    plt.title('Volcano Plot: Differential Abundance')
    plt.xlabel('Log Difference (Approx Log2FC)')
    plt.ylabel('-Log10 Adjusted P-value')
    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=300)
    plt.close()
    print(f"   - Volcano plot saved to {save_plot_path}")

def run_lasso_model(X, y):
    print("🤖 Training Lasso Logistic Regression...")
    
    # 1. Feature Filtering (SPEED OPTIMIZATION)
    # Remove features with 0 variance (constant columns) to speed up fitting
    print("   - Removing constant features...")
    selector = VarianceThreshold(threshold=0)
    X_reduced = selector.fit_transform(X)
    feat_names = X.columns[selector.get_support()]
    print(f"   - Features reduced from {X.shape[1]} to {X_reduced.shape[1]}")
    
    # 2. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)
    
    # 3. Model Training (SPEED OPTIMIZATION)
    # n_jobs=-1 uses all CPU cores
    # tol=1e-3 relaxes convergence slightly for speed
    print("   - Fitting Cross-Validation Model (this may take a moment)...")
    clf = LogisticRegressionCV(
        cv=5, 
        penalty='l1', 
        solver='liblinear', 
        max_iter=5000,
        tol=1e-3,       # Faster convergence
        scoring='roc_auc',
        n_jobs=-1,      # PARALLEL PROCESSING
        random_state=42
    )
    
    clf.fit(X_scaled, y)
    
    print(f"✅ Model Trained! Average AUC: {clf.scores_[1].mean():.3f}")
    
    # 4. Extract Top Features
    coefs = pd.Series(clf.coef_[0], index=feat_names)
    important = coefs[coefs != 0].sort_values(ascending=False)
    
    print(f"   - Selected {len(important)} predictive sequences out of {len(feat_names)}.")
    return important

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # 1. Load Data (now includes Log transformation)
        X, y = load_and_prep_data(MATRIX_FILE, METADATA_FILE)
        
        # 2. PCA
        plot_pca(X, y, os.path.join(OUTPUT_DIR, "pca_plot.png"))
        
        # 3. Stats
        run_differential_abundance(
            X, y, 
            os.path.join(OUTPUT_DIR, "stats.csv"),
            os.path.join(OUTPUT_DIR, "volcano.png")
        )
        
        # 4. ML Model
        top_features = run_lasso_model(X, y)
        
        # Save top features
        if not top_features.empty:
            save_path = os.path.join(OUTPUT_DIR, "predictive_biomarkers.csv")
            top_features.to_csv(save_path)
            print(f"📜 Top predictive sequences saved to {save_path}")
            print("\nTop 5 Positive Predictors (Higher in CMV+):")
            print(top_features.head(5))
        else:
            print("⚠️ Model dropped all features (Lasso regularization was too strong).")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()