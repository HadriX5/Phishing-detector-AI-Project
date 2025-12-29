from src import load_raw_data, clean_data, repair_encoding, ammend_columns

def main():
    """
    Main entry point for cleaning and preparing the dataset.
    Loads the raw data, repairs it, cleans it, and saves it as a processed .parquet file.

    Returns:
        None
    """
    raw_data = load_raw_data(967)
    
    # -- Repair encoding issues --
    repaired_data = raw_data.copy()

    # Identify the columns which have str's
    string_columns = repaired_data.select_dtypes(include=['object']).columns

    # Apply encoding repair from src module
    for col in string_columns:
        repaired_data[col] = repaired_data[col].apply(repair_encoding)

    ammended_data = ammend_columns(repaired_data)

    try:
        clean_data(ammended_data)
    except IOError as e:
        print(e)

if __name__ == "__main__":
    main()