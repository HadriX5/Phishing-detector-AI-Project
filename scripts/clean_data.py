import pandas as pd
from src import load_raw_data, clean_data

def main():
    """
    Punt d'entrada principal per a la neteja i preparació del conjunt de dades.
    Carrega les dades en brut, les neteja i les desa com a fitxer CSV processat.
    """
    raw_data, y = load_raw_data(id=967)
    try:
        processed_data = clean_data(raw_data, y)
    except IOError as e:
        print(e)