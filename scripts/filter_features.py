from src import select_features, filter_correlated_features

def main():
    df = filter_correlated_features()
    select_features(df)

if __name__ == "__main__":
    main()