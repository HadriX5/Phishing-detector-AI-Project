# --------------------------------------------------------------------------------------------------
# predict.py
# --------------------------------------------------------------------------------------------------
# Script per fer inferència (prediccions) sobre noves URLs utilitzant els models entrenats.
# --------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import pickle
import os
import sys

# Model paths
AGENT_PATH = "data/objects/q_agent_knn.pkl"
DISC_PATH = "data/objects/discretizer.pkl"

def load_models():
    """
    Loads the discretizer and the Q-learning agent with KNN model from disk.

    Returns:
        discretizer: The loaded Discretizer object.
        agent: The loaded QLearningAgentKNN object.
    """
    if not os.path.exists(AGENT_PATH) or not os.path.exists(DISC_PATH):
        raise FileNotFoundError("Models no trobats.")

    print("Loading models...")
    
    # 1. Load Discretizer from disk
    with open(DISC_PATH, 'rb') as f:
        discretizer = pickle.load(f)
    
    # 2. Load Q-learning Agent with KNN from disk
    with open(AGENT_PATH, 'rb') as f:
        agent = pickle.load(f)
    
    return discretizer, agent

def predict_single_url(features_dict, discretizer, agent):
    """
    Predicts whether a single URL is phishing or legitimate using the provided discretizer and agent.

    Args:
        features_dict (dict): A dictionary containing the features of the URL.
        discretizer: The Discretizer object for feature transformation.
        agent: The QLearningAgentKNN object for making predictions.
    Returns:
        action (int): The predicted label (0 for legitimate, 1 for phishing).
        method (str): The method used for prediction ("Q-Table (Memòria)" or "KNN (Similitud)").
    """
    # 1. Convert input dictionary to DataFrame for Discretizer
    df_new = pd.DataFrame([features_dict])
    
    # 2. Drop unnecessary columns if present
    cols_to_drop = ['URL', 'Domain', 'TLD', 'Title', 'label']
    df_clean = df_new.drop(
        columns = [c for c in cols_to_drop if c in df_new.columns], errors = 'ignore')

    # 3. Discretize features
    df_discrete = discretizer.transform(df_clean)
    
    # 4. Convert to the state format expected by the agent (NumPy array)
    state = df_discrete.values[0] # Take the first row as an array
    
    # 5. Agent prediction
    # is_test = True forces the use of either the Q-Table or the KNN.
    q_hits = agent.q_hits
    action = agent.choose_action(state, is_test = True)
    
    # Extra info for debugging
    method = "Q-Table (Memory)" if agent.q_hits >= q_hits else "KNN (Similarity)"
    
    return action, method

if __name__ == "__main__":
    
    # 1. Load models
    try:
        discretizer, agent = load_models()
        print("Models loaded successfully.\n")
    except Exception as e:
        print(e)
        sys.exit(1)

    # 2. Simulate a new URL with feature values

    # CASE A: An example that looks like PHISHING (suspicious values)
    sample_phishing = {
        "URLCharProb": 0.05812,       # Similar (0.06)
        "NoOfEmptyRef": 1,            # Slightly higher
        "HasPasswordField": 1,        # Phishing usually has password
        "NoOfiFrame": 2,              # A couple of iframes
        "NoOfDegitsInURL": 0,         # No digits, looks clean
        "TLDLegitimateProb": 0.51,    # Common TLD (.com/.net)
        "HasCopyrightInfo": 1,        # Tries to look real
        "CharContinuationRate": 0.95, # Very high
        "NoOfJS": 25,                 # Lots of JS (similar to 28)
        "HasSocialNet": 1,            # Puts FB/Twitter logos to deceive
        "LargestLineLength": 8500,    # Minified or obfuscated code (similar to 9381)
        "HasDescription": 1,          # Has meta description
        "NoOfImage": 30,              # Many images (similar to 34)
        "DomainTitleMatchScore": 0.0, # Title has nothing to do with the domain
        "NoOfExternalRef": 115,       # Many external refs (similar to 124)
        "NoOfSelfRef": 105,           # Many internal refs
        "URLLength": 35,              # Average length
        "NoOfSubDomain": 1,           # www
        "IsHTTPS": 1,                 # Has padlock (HTTPS)
        "LineOfCode": 600,            # Lots of code 
        "label": 1,                   # Target
        # Metadata (do not affect the model, only visual)
        "URL": "https://www.secure-update-account-verify.net",
        "Domain": "www.secure-update-account-verify.net",
        "TLD": "net",
        "Title": "Welcome to our Service - Please Login to continue" 
    }
    
    # CASE B: NEW LEGITIMATE EXAMPLE (Similar to 'xsph.ru')
    sample_legit = {
        "URLCharProb": 0.019,         # Low
        "NoOfEmptyRef": 0,             
        "HasPasswordField": 0,
        "NoOfiFrame": 0,
        "NoOfDegitsInURL": 4,           # Some numbers
        "TLDLegitimateProb": 0.02,      # Strange TLD
        "HasCopyrightInfo": 0,          # Shabby web
        "CharContinuationRate": 0.85, 
        "NoOfJS": 0,                    # Zero JS
        "HasSocialNet": 0,
        "LargestLineLength": 50,        # Short lines
        "HasDescription": 0,
        "NoOfImage": 1,                 # Maybe a broken icon
        "DomainTitleMatchScore": 100.0, # Match (often default)
        "NoOfExternalRef": 1,           # Almost no links
        "NoOfSelfRef": 0,
        "URLLength": 29,                # Short
        "NoOfSubDomain": 2,             # Technical subdomain
        "IsHTTPS": 0,                   # Not secure (HTTP)
        "LineOfCode": 15,               # Almost empty
        "label": 0,                     # Target
        # Metadata
        "URL": "http://test01.server-arch.test.xyz",
        "Domain": "test01.server-arch.test.xyz",
        "TLD": "xyz",
        "Title": "Index of /pub/archive"
    }

    # Ensure all features are present in the samples
    all_columns = list(discretizer.bins.keys()) 
    
    full_sample_phishing = {col: sample_phishing.get(col, 0) for col in all_columns}
    full_sample_legit = {col: sample_legit.get(col, 0) for col in all_columns}

    
    print("--- TEST 1: Suspicious URL ---")
    # Reset debug counters before predicting
    agent.q_hits = 0; agent.knn_hits = 0 
    
    pred, method = predict_single_url(full_sample_phishing, discretizer, agent)
    label = "PHISHING" if pred == 1 else "LEGITIMATE"
    print(f"Result: {label}")
    print(f"Method used: {method}")
    
    print("\n--- TEST 2: Secure URL ---")
    agent.q_hits = 0; agent.knn_hits = 0

    pred, method = predict_single_url(full_sample_legit, discretizer, agent)
    label = "PHISHING" if pred == 1 else "LEGITIMATE"
    print(f"Result: {label}")
    print(f"MMethod used: {method}")