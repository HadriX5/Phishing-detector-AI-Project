import os
import pickle
import pandas as pd

from src import Discretizer

def main():
    """
    Main entry point for discretizing data.
    Calls the discretization function from the data_discretization module.
    """
    discretizer = Discretizer(n_bins=5)
    discretizer.fit_transform()

    OUTPUT_MODEL = 'data/objects/discretizer_model.pkl'
    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
    
    with open(OUTPUT_MODEL, 'wb') as f:
        pickle.dump(discretizer, f)

    # Example of loading and using the saved discretizer and transforming new data
    # using the saved model.
    # with open('data/discrete/discretizer_model.pkl', 'rb') as f:
    #     discretizer_reload = pickle.load(f)

    # new_data = pd.DataFrame([{
    #     "URLCharProb":0.050207214,"NoOfEmptyRef":0,
    #     "HasPasswordField":0,"NoOfiFrame":0,"TLDLegitimateProb":0.0326503,
    #     "NoOfDegitsInURL":0,"HasCopyrightInfo":1,"CharContinuationRate":0.666666667,
    #     "NoOfJS":8,"HasSocialNet":1,"LargestLineLength":9381,"HasDescription":0,
    #     "NoOfImage":50,"DomainTitleMatchScore":55.55555556,"NoOfExternalRef":217,
    #     "NoOfSelfRef":39,"URLLength":24,"NoOfSubDomain":1,"IsHTTPS":1,"LineOfCode":618,
    #     "label":1,"URL":"https://www.uni-mainz.de","Domain":"www.uni-mainz.de","TLD":"de",
    #     "Title":"johannes gutenberg-universität mainz"}
    # ])

    # state_discrete = discretizer_reload.transform(new_data)
    # print(state_discrete)


if __name__ == "__main__":
    main()
