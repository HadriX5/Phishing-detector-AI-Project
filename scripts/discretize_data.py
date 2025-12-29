import os
import pickle

from src import Discretizer

def main():
    """
    Main entry point for discretizing data.
    Calls the discretization function from the data_discretization module.

    Returns:
        None
    """
    discretizer = Discretizer(n_bins=5)
    discretizer.fit_transform()

    OUTPUT_MODEL = 'data/objects/discretizer_model.pkl'
    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
    
    with open(OUTPUT_MODEL, 'wb') as f:
        pickle.dump(discretizer, f)


if __name__ == "__main__":
    main()
