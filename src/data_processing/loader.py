from ucimlrepo import fetch_ucirepo 

def load_raw_data(id: int) -> tuple:
    """
    Carrega el conjunt de característiques i les etiquetes objectiu d’un conjunt
    de dades del UCI Machine Learning Repository.

    Paràmetres:
        id (int): Identificador únic del conjunt de dades al repositori de la UCI.

    Retorna:
        tuple: Un parell (característiques, objectius) on:
            - característiques (pd.DataFrame): DataFrame que conté les característiques d'entrada del conjunt de dades.
            - objectius (pd.DataFrame o pd.Series): Etiquetes objectiu associades al conjunt de dades.
    """

    data = fetch_ucirepo(id = id) 
    raw_data = data.data.features 
    y = data.data.targets
    return raw_data, y

    # MIRAR SI TARGETS ÉS DATAFRAME O SERIES
