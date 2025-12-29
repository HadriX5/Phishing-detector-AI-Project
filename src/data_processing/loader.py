from ucimlrepo import fetch_ucirepo 
import pandas as pd

def load_raw_data(id: int = 967) -> pd.DataFrame:
    """
    Loads the feature set and target labels from a dataset in the UCI Machine Learning Repository.
    Parameters:
        id (int): Unique identifier of the dataset in the UCI repository.
    Returns:
        pd.DataFrame: A DataFrame containing both the features and target labels of the dataset.
    """

    # Fetch the data from the repository
    data = fetch_ucirepo(id = id)

    # Complete the dataset for cleaning
    raw_data = pd.concat([data.data.features, data.data.targets], axis = 1)
    return raw_data
