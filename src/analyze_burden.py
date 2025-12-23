import pandas as pd
import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from statsmodels.stats.multitest import multipletests

def load_and_prep_data(matrix_path, metadata_path):
    # 1. Load the Ratio Matrix (Output from your main script)
    # Shape: Rows=Sequences, Cols=Patients
    df_ratios = pl.read_csv(matrix_path)
    
    # Transpose to Shape: Rows=Patients, Cols=Sequences (Standard for ML)
    # We drop the 'hc_seq' column first, then transpose
    seq_names = df_ratios["hc_seq"].to_list()
    data_values = df_ratios.drop("hc_seq").to_numpy().T
    patient_ids_in_matrix = df_ratios.columns[1:] # Skip 'hc_seq'
    
    # Create a clean Pandas DataFrame for analysis
    X_df = pd.DataFrame(data_values, index=patient_ids_in_matrix, columns=seq_names)
    
    # Fill NaNs with 0 (Assuming NaN means the sequence wasn't found, i.e., 0 burden)
    X_df = X_df.fillna(0.0)

    # 2. Load Metadata and Map Labels
    meta = pd.read_csv(metadata_path, sep="\t")
    
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
            y_map[pid] = 1 # Positive
        elif 'Cytomegalovirus -' in tag:
            y_map[pid] = 0 # Negative

    # 3. Align X (Data) and y (Labels)
    # Only keep patients that exist in both the matrix and the metadata
    common_ids = [pid for pid in X_df.index if pid in y_map]
    
    X_final = X_df.loc[common_ids]
    y_final = np.array([y_map[pid] for pid in common_ids])
    
    print(f"Data Prepared: {X_final.shape[0]} patients, {X_final.shape[1]} sequences.")
    print(f"Class Balance: {sum(y_final)} CMV+ / {len(y_final) - sum(y_final)} CMV-")
    
    return X_final, y_final

def run_differential_abundance(X, y):
    print("\n--- Running Differential Abundance (Volcano Analysis) ---")
    results = []
    
    # Separate groups
    pos_group = X[y == 1]
    neg_group = X[y == 0]
    
    for seq in X.columns:
        # Mann-Whitney U Test
        u_stat, p_val = mannwhitneyu(pos_group[seq], neg_group[seq], alternative='two-sided')
        
        # Calculate Log2 Fold Change (add small epsilon to avoid div by zero)
        mean_pos = pos_group[seq].mean() + 1e-6
        mean_neg = neg_group[seq].mean() + 1e-6
        log2fc = np.log2(mean_pos / mean_neg)
        
        results.append({
            'sequence': seq,
            'p_value': p_val,
            'log2fc': log2fc,
            'mean_pos': mean_pos,
            'mean_neg': mean_neg
        })
        
    res_df = pd.DataFrame(results)
    
    # Multiple testing correction (Benjamini-Hochberg)
    res_df['p_adj'] = multipletests(res_df['p_value'], method='fdr_bh')[1]
    
    # Sort by significance
    top_hits = res_df.sort_values('p_adj').head(10)
    print("Top 10 Differentiating Sequences:")
    print(top_hits[['sequence', 'log2fc', 'p_adj']])
    
    return res_df

def run_predictive_modeling(X, y):
    print("\n--- Running Lasso Logistic Regression ---")
    
    # Standardize features (important for Lasso)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Logistic Regression with L1 penalty (Lasso) and Cross-Validation
    # solver='liblinear' handles L1 well for smaller datasets
    clf = LogisticRegressionCV(
        cv=5, 
        penalty='l1', 
        solver='liblinear', 
        max_iter=5000,
        scoring='roc_auc'
    )
    
    clf.fit(X_scaled, y)
    
    print(f"Model AUC Score (CV average): {clf.scores_[1].mean():.3f}")
    
    # Extract coefficients
    coefs = pd.Series(clf.coef_[0], index=X.columns)
    important_feats = coefs[coefs != 0].sort_values(ascending=False)
    
    print(f"Number of non-zero features selected: {len(important_feats)}")
    if not important_feats.empty:
        print("Top Predictive Sequences (Positive = predictive of CMV+):")
        print(important_feats.head(5))
        print(important_feats.tail(5))
        
    return clf, important_feats

def plot_pca(X, y):
    print("\n--- Generating PCA Plot ---")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette={0: 'blue', 1: 'red'}, style=y, s=100)
    plt.title(f'PCA of TCR Ratios (Red=CMV+, Blue=CMV-)\nExplained Variance: {pca.explained_variance_ratio_.sum():.2f}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.legend(title='CMV Status', labels=['CMV-', 'CMV+'])
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Update these paths
    MATRIX_FILE = "/Users/ishaharris/Projects/TCR/clono-scan/data/burden/AC02_Emerson_burden_matrix.csv" 
    METADATA_FILE = "/Users/ishaharris/Projects/TCR/TCR-Isha/data/Repertoires/Cohort01_whole_metadata.tsv"
    
    try:
        X, y = load_and_prep_data(MATRIX_FpILE, METADATA_FILE)
        
        print("Data Loaded Successfully.")
        # 1. Unsupervised Check
        plot_pca(X, y)
        
        # 2. Statistical Check (Which specific sequences differ?)
        stats_df = run_differential_abundance(X, y)
        
        # 3. Predictive Check (Can we predict status?)
        model, important_features = run_predictive_modeling(X, y)
        
    except Exception as e:
        print(f"Analysis failed: {e}")