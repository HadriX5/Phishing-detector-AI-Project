import pandas as pd
import numpy as np
import os

# Rutes
OUTPUT_DIR = 'notebooks'
OUTPUT_FILE = 'large_dummy_dataset.parquet'

def create_dummy_data():
    # ESCALA REAL: 200.000 files
    n_rows = 200000 
    
    print(f"Generant estructura per a {n_rows} files...")
    
    # Inicialitzem a 0 (int8 per estalviar RAM si volguessim, però int64 és més segur per ara)
    df = pd.DataFrame(0, index=range(n_rows), columns=[
        'URLCharProb', 'TLDLegitimateProb', 'CharContinuationRate', 'DomainTitleMatchScore',
        'NoOfEmptyRef', 'NoOfiFrame', 'NoOfDegitsInURL', 'NoOfJS', 'LargestLineLength',
        'NoOfImage', 'NoOfExternalRef', 'NoOfSelfRef', 'URLLength', 'NoOfSubDomain',
        'LineOfCode', 'HasPasswordField', 'HasCopyrightInfo', 'HasSocialNet',
        'HasDescription', 'IsHTTPS', 'label'
    ])

    # Calculem índexs
    end_phishing = int(n_rows * 0.4)      # 0 a 80.000
    end_legit = end_phishing + int(n_rows * 0.4) # 80.000 a 160.000
    # La resta (160k a 200k) serà soroll

    print("1. Creant exèrcit de CLONS PHISHING (40%)...")
    # Patró: No HTTPS + URL Llarga + Molts Dígits -> Label 1
    df.loc[0:end_phishing, 'IsHTTPS'] = 0
    df.loc[0:end_phishing, 'URLLength'] = 4
    df.loc[0:end_phishing, 'NoOfDegitsInURL'] = 4
    df.loc[0:end_phishing, 'HasPasswordField'] = 1
    df.loc[0:end_phishing, 'label'] = 1 
    
    print("2. Creant exèrcit de CLONS LEGÍTIMS (40%)...")
    # Patró: HTTPS + URL Curta + Sense Dígits -> Label 0
    df.loc[end_phishing:end_legit, 'IsHTTPS'] = 1
    df.loc[end_phishing:end_legit, 'URLLength'] = 0
    df.loc[end_phishing:end_legit, 'NoOfDegitsInURL'] = 0
    df.loc[end_phishing:end_legit, 'HasCopyrightInfo'] = 1
    df.loc[end_phishing:end_legit, 'label'] = 0

    print("3. Generant SOROLL ALEATORI (20%)...")
    remaining_idx = range(end_legit, n_rows)
    for col in df.columns:
        if col != 'label':
            # Valors random de 0 a 4 (simulant bins)
            df.loc[remaining_idx, col] = np.random.randint(0, 5, len(remaining_idx))
    
    # Assignem labels aleatoris al soroll
    df.loc[remaining_idx, 'label'] = np.random.randint(0, 2, len(remaining_idx))

    print("4. Barrejant dades (Shuffle)...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Assegurar tipus enters
    df = df.astype(int)

    # Guardar
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df.to_parquet(output_path, index=False)
    
    print(f"Dataset creat a: {output_path}")
    print(f"Dimensions: {df.shape}")

if __name__ == "__main__":
    create_dummy_data()