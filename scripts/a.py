import pandas as pd
import tldextract
import json
import os

# CONFIGURACIÓ
PARQUET_FILE = "data/features/selected_features_df.parquet"
OUTPUT_JSON = "data/others/tld_data.json"

def generate_tld_json():
    print(f"📖 Llegint {PARQUET_FILE}...")
    try:
        df = pd.read_parquet(PARQUET_FILE)
    except Exception as e:
        print(f"❌ Error llegint el parquet: {e}")
        return

    print("⚙️  Calculant probabilitats per TLD...")
    
    # 1. Extreure TLDs
    df['temp_tld'] = df['URL'].apply(lambda x: tldextract.extract(x).suffix.lower())
    
    # 2. Calcular mitjanes
    tld_map = df.groupby('temp_tld')['TLDLegitimateProb'].mean().to_dict()
    global_mean = df['TLDLegitimateProb'].mean()
    
    # 3. Preparar l'estructura final
    data_to_save = {
        "default_prob": global_mean,
        "tld_map": tld_map
    }

    # 4. Assegurar que el directori existeix
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    # 5. Guardar JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4)
        
    print(f"✅ Fitxer JSON generat correctament a: {OUTPUT_JSON}")

if __name__ == "__main__":
    generate_tld_json()