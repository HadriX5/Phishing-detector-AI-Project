import pandas as pd
import numpy as np

class Discretizer:
    def __init__(self, n_bins=5):
        self.n_bins = n_bins
        self.bins = {} 
        # Columnes a ignorar (Target i Text)
        self.drop_cols = ['label', 'URL', 'Domain', 'TLD', 'Title'] 

    def fit(self, df):
        """
        Aprèn els bins basant-se només en el DataFrame passat (hauria de ser X_train).
        NO modifica el df, només guarda els llindars a self.bins.
        """
        print(f"Fitting Discretizer (n_bins={self.n_bins})...")
        
        # Seleccionem només numèriques rellevants
        numeric_cols = [c for c in df.select_dtypes(include=['number']).columns 
                        if c not in self.drop_cols]

        for col in numeric_cols:
            # 1. Detectar binaris
            unique_vals = df[col].dropna().unique()
            is_binary = all(val in [0, 1] for val in unique_vals)

            if is_binary or len(unique_vals) < self.n_bins:
                self.bins[col] = None # Marquem com "No Discretitzar"
                continue

            # 2. Calcular bins (qcut)
            try:
                # retbins=True ens dona els límits. 
                # Només volem aprendre els límits (boundaries), no transformar encara.
                _, boundaries = pd.qcut(
                    df[col], 
                    q=self.n_bins, 
                    retbins=True, 
                    duplicates='drop'
                )
                self.bins[col] = boundaries
            except Exception as e:
                print(f"Warning fitting '{col}': {e}")
                self.bins[col] = None

    def transform(self, df):
        """
        Aplica els bins apresos a un DataFrame (Train o Test).
        """
        df_copy = df.copy()
        
        # Iterem només sobre les columnes que hem après
        for col in df_copy.columns:
            if col in self.bins:
                if self.bins[col] is not None:
                    # USEM PD.CUT amb els límits apresos (self.bins)
                    df_copy[col] = pd.cut(
                        df_copy[col], 
                        bins=self.bins[col], 
                        labels=False, 
                        include_lowest=True
                    ).fillna(0).astype(int) # Els nuls o fora de rang (si n'hi ha) al 0
                else:
                    # Si era binari o pocs valors, només assegurar int
                    df_copy[col] = df_copy[col].fillna(0).astype(int)
        
        return df_copy