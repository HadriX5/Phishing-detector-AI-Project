from src import select_features, filter_correlated_features

def main():
    reduced_df, df = filter_correlated_features()
    select_features(reduced_df, df)

if __name__ == "__main__":
    main()