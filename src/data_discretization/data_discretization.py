import pandas as pd
import pyarrow.parquet as pq
import os

def discretize_and_save(n_bins=5):
    """
    Reads a Parquet file, discretizes continuous numeric variables, and saves the result.
    
    Args:
        n_bins (int): Number of bins (groups) for the continuous variables.
    """
    input_path = 'data/features/selected_features_df.parquet'
    output_path = 'data/discrete/discrete_df.parquet'

    print(f"Reading input file: {input_path}")
    
    # 1. Read the data
    try:
        df = pq.read_table(input_path).to_pandas()
    except FileNotFoundError:
        print("Error: Input file not found.")
        return

    # 2. Identify columns to process, gets only numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Columns we do NOT want to touch (strings and the target 'label')
    # Even though 'label' is numeric, it is not a feature to discretize; it is the target.
    excluded_cols = ['label'] 
    
    for col in numeric_cols:
        if col in excluded_cols:
            continue
            
        # If the column only has 0s and 1s (or only one unique value), we don't discretize it.
        unique_vals = df[col].dropna().unique()
        is_binary = all(val in [0, 1] for val in unique_vals)
        
        if is_binary:
            print(f"Skipping binary: {col}")
            continue
            
        # If it has too few unique values (fewer than the bins), don't force qcut
        if len(unique_vals) < n_bins:
             print(f"Skipping low variance: {col} (has {len(unique_vals)} unique values)")
             continue

        # 3. Discretization (Q-cut)
        # We use qcut (quantile cut) to try to get roughly the same number of samples in each bin.
        try:
            # labels=False returns 0, 1, 2... instead of ranges like "(0.5, 0.9]"
            # duplicates = 'drop' handles cases where many values repeat (e.g., many 0s)
            df[col] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
            print(f"Discretized: {col} -> {n_bins} bins")
        except ValueError as e:
            print(f"Error discretizing {col}: {e}")

    # 5. Ensure the data are integers (less memory usage we can remove it if needed)
    for col in numeric_cols:
        if col != 'label':
             df[col] = df[col].fillna(-1).astype(int)

    # 6. Save the data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)    
    df.to_parquet(output_path)
    print(f"Successfully saved to: {output_path}")