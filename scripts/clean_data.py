from src import load_raw_data, clean_data, repair_encoding, ammend_columns

def main():
    """
    Punt d'entrada principal per a la neteja i preparació del conjunt de dades.
    Carrega les dades en brut, les repara, les neteja i les desa com a fitxer .parquet processat.
    """

    # -- Load raw data --
    raw_data = load_raw_data(967)
    
    # -- Repair encoding issues --
    repaired_data = raw_data.copy()

    # Identify the columns which have str's
    string_columns = repaired_data.select_dtypes(include=['object']).columns

    for col in string_columns:
        repaired_data[col] = repaired_data[col].apply(repair_encoding)

    # -- Ammend columns --
    ammended_data = ammend_columns(repaired_data)

    # -- Clean data --
    try:
        clean_data(ammended_data)
    except IOError as e:
        print(e)

if __name__ == "__main__":
    main()