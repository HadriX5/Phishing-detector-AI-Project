from ucimlrepo import fetch_ucirepo 
import pandas as pd

def load_raw_data(id: int = 967) -> pd.DataFrame:
    """
    Carrega el conjunt de característiques i les etiquetes objectiu d’un conjunt
    de dades del UCI Machine Learning Repository.

    Paràmetres:
        id (int): Identificador únic del conjunt de dades al repositori de la UCI.

    Retorna:
        pd.DataFrame: Un DataFrame que conté tant les característiques com les 
                      etiquetes objectiu del conjunt de dades.
    """

    # Fetch the data from the repository
    data = fetch_ucirepo(id)

    # Complete the dataset for cleaning
    raw_data = pd.concat([data.data.features, data.data.targets], axis = 1)
    return raw_data
