import pandas as pd
import pyarrow.parquet as pq
import os


class Discretizer:
    def __init__(self, n_bins=5):
        self.n_bins = n_bins
        self.bins = {} 
        self.drop_cols = ['label', 'URL', 'Domain', 'TLD', 'Title'] 

    def fit_transform(self):
        """
        Computes the bin boundaries (fit) and discretizes the training data.
        """
        input_path = 'data/features/selected_features_df.parquet'
        output_path = 'data/discrete/discrete_df.parquet'

        print(f"Starting discretization (n_bins={self.n_bins})")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")
            
        df = pq.read_table(input_path).to_pandas()
        
        # Select relevant numeric columns
        numeric_cols = [c for c in df.select_dtypes(include=['number']).columns 
                        if c not in self.drop_cols]

        for col in numeric_cols:
            # Check for binary columns (0/1)
            unique_vals = df[col].dropna().unique()
            is_binary = all(val in [0, 1] for val in unique_vals)

            if is_binary:
                df[col] = df[col].astype(int)
                continue
            
            # If unique values are less than n_bins, skip discretization
            if len(unique_vals) < self.n_bins:
                df[col] = df[col].astype(int)
                continue

            # Compute bins and discretize (qcut)
            try:
                # retbins=True returns the exact bin edges
                discretized_series, boundaries = pd.qcut(
                    df[col], 
                    q=self.n_bins, 
                    labels=False, 
                    retbins=True, 
                    duplicates='drop'
                )
                
                # Store bin edges for later transforms
                self.bins[col] = boundaries
                
                # Update the dataframe
                df[col] = discretized_series.fillna(-1).astype(int)
                
            except Exception as e:
                print(f"Warning for '{col}': {e}")

        # Save discretized parquet
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path)
        print(f"Discretized data saved to: {output_path}")
    
    def transform(self, new_data_df):
        """
        Extra helper: applies the learned bin edges to new data.
        Uses pd.cut instead of qcut, based on self.bins.
        """
        df_copy = new_data_df.copy()
        
        for col, boundaries in self.bins.items():
            if col in df_copy.columns:
                # Use pd.cut with the stored bin edges
                # include_lowest=True ensures the minimum value is included
                df_copy[col] = pd.cut(
                    df_copy[col], 
                    bins=boundaries, 
                    labels=False, 
                    include_lowest=True
                ).fillna(-1).astype(int)
        
        return df_copy