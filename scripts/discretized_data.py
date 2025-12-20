from src import discretize_and_save

def main():
    """
    Main entry point for discretizing data.
    Calls the discretization function from the data_discretization module.
    """
    discretize_and_save(n_bins=5)

if __name__ == "__main__":
    main()
