import pandas as pd

def clean_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Neteja i prepara el conjunt de dades per a un processament posterior o per a l’entrenament d’un model.

    Paràmetres:
        raw_data (pd.DataFrame): Les dades de característiques en brut obtingudes del conjunt de dades.
        y (pd.DataFrame o pd.Series): Les etiquetes objectiu corresponents.

    Retorna:
        pd.DataFrame: El conjunt de dades netejat amb les files duplicades i amb valors
                      mancants eliminades, i amb les etiquetes objectiu afegides com
                      una nova columna 'label'. El conjunt de dades netejat també es
                      desa com a fitxer CSV.
    """


    processed_data = raw_data.drop_duplicates().dropna()

    # IsHTTPS mismatch with URL
    https_mismatch_indices = processed_data[
        (processed_data['URL'].str.startswith('https') & (processed_data['IsHTTPS'] == 0)) |
        (~processed_data['URL'].str.startswith('https') & (processed_data['IsHTTPS'] == 1))
    ]

    processed_data = processed_data.drop(https_mismatch_indices.index)

    # Drop any inconsistent ip address rows
    octet_pattern = r'(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)'
    
    # We combine it: ^ (Group . Group . Group . Group) $
    strict_ip_regex = (r'^' + 
        octet_pattern + r'\.' + 
        octet_pattern + r'\.' + 
        octet_pattern + r'\.' + 
        octet_pattern + r'$'
    )

    # Check matches
    is_valid_ip = processed_data['Domain'].str.match(strict_ip_regex)
    

    # If IsDomainIP=1, domain MUST be a valid IP address
    # If IsDomainIP=0, domain MUST NOT be a valid IP address
    processed_data = processed_data[
        ((processed_data['IsDomainIP'] == 1) & is_valid_ip) |
        ((processed_data['IsDomainIP'] == 0) & ~is_valid_ip)
    ]

    # URL Length mismatch
    processed_data = processed_data[
        processed_data['URL'].str.len() == processed_data['URLLength']
    ]
    
    # Title mismatch
    has_content = processed_data['Title'].str.strip().astype(bool)
    processed_data = processed_data[
        ((processed_data['HasTitle'] == 1) & has_content) |
        ((processed_data['HasTitle'] == 0) & ~has_content)
    ]

    # Impossible Ratios (0 to 1)
    ratio_columns = [
        'URLSimilarityIndex', 'CharContinuationRate', 'TLDLegitimateProb', 
        'URLCharProb', 'ObfuscationRatio', 'LetterRatioInURL', 
        'DegitRatioInURL', 'SpacialCharRatioInURL'
    ]
    for col in ratio_columns:
        processed_data = processed_data[
            (processed_data[col] >= 0) & (processed_data[col] <= 1)
        ]

    output_path = 'data/processed/processed_dataset.parquet'
    processed_data.to_parquet(output_path, index = False)

    try:
        pd.read_parquet(output_path)
    except Exception as e:
        raise IOError("Error al desar el fitxer Parquet: " + str(e))
    
    print("Dades netejades desades a:", output_path)
    return 0
