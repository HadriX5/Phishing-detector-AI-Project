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

    # Normalize all caracters (`, ´, etc.)
    processed_data['label'] = y
    processed_data.to_csv('../data/processed/processed_dataset.csv', index=False)

    # Verificar que s'ha desat correctament i llençar error si no
    try:
        pd.read_csv('../data/processed/processed_dataset.csv')
    except Exception as e:
        raise IOError("Error al desar el fitxer CSV: " + str(e))

    return 0
