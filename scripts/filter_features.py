from src import select_features, filter_correlated_features

def main():
    """
    Main entry point for filtering features.
    Calls the feature filtering functions from the feature_selection module.

    Returns:
        None
    """
    reduced_df, df = filter_correlated_features()
    select_features(reduced_df, df)

if __name__ == "__main__":
    main()