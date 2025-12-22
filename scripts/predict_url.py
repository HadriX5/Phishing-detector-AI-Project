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

from src import QLearningAgent, Discretizer

# Rutes dels models entrenats
AGENT_PATH = "data/objects/q_agent_knn.pkl"
DISC_PATH = "data/objects/discretizer.pkl"

def load_models():
    """Carrega el Discretitzador i l'Agent (objectes sencers)."""
    if not os.path.exists(AGENT_PATH) or not os.path.exists(DISC_PATH):
        raise FileNotFoundError("Models no trobats.")

    print("Loading models...")
    
    # 1. Carregar Discretitzador
    with open(DISC_PATH, 'rb') as f:
        discretizer = pickle.load(f)
    
    # 2. Carregar Agent (Ara és directa!)
    # Com que hem guardat l'objecte, pickle ens retorna la instància amb tots els mètodes i atributs
    with open(AGENT_PATH, 'rb') as f:
        agent = pickle.load(f)
    
    return discretizer, agent

def predict_single_url(features_dict, discretizer, agent):
    """
    Fa la predicció per a una sola URL (passada com a diccionari de features).
    """
    # 1. Convertim el diccionari a DataFrame (format que espera el discretitzador)
    df_new = pd.DataFrame([features_dict])
    
    # 2. Eliminem columnes de text si venen (per seguretat)
    cols_to_drop = ['URL', 'Domain', 'TLD', 'Title', 'label']
    df_clean = df_new.drop(
        columns = [c for c in cols_to_drop if c in df_new.columns], errors = 'ignore')

    # 3. Discretització
    # Això converteix els floats (0.045) en enters (bin 0, 1, 2...) usant els rangs apresos.
    df_discrete = discretizer.transform(df_clean)
    
    # 4. Convertim a l'estat que espera l'agent (Array de NumPy)
    state = df_discrete.values[0] # Agafem la primera fila com a array
    
    # 5. Predicció de l'Agent
    # is_test = True força a fer servir la Q-Table o el KNN.
    q_hits = agent.q_hits
    action = agent.choose_action(state, is_test = True)
    
    # Informació extra per depurar
    method = "Q-Table (Memòria)" if agent.q_hits >= q_hits else "KNN (Similitud)"
    
    return action, method

# --------------------------------------------------------------------------------------------------
# EXEMPLE D'ÚS
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Carreguem models
    try:
        discretizer, agent = load_models()
        print("Models carregats correctament.\n")
    except Exception as e:
        print(e)
        sys.exit(1)

    # 2. Simulem una URL nova amb valors de features

    # CAS A: Un exemple que sembla PHISHING (valors sospitosos)
    sample_phishing = {
        "URLCharProb": 0.05812,       # Similar (0.06)
        "NoOfEmptyRef": 1,            # Lleugerament superior
        "HasPasswordField": 1,        # Phishing sol tenir password (a diferència de l'anterior, aquest és més agressiu)
        "NoOfiFrame": 2,              # Un parell d'iframes
        "NoOfDegitsInURL": 0,         # Sense dígits, sembla net
        "TLDLegitimateProb": 0.51,    # TLD comú (.com/.net)
        "HasCopyrightInfo": 1,        # Intenta semblar real
        "CharContinuationRate": 0.95, # Molt alt
        "NoOfJS": 25,                 # Molt JS (similar a 28)
        "HasSocialNet": 1,            # Posa logos de FB/Twitter per enganyar
        "LargestLineLength": 8500,    # Codi minificat o ofuscat (similar a 9381)
        "HasDescription": 1,          # Té meta description
        "NoOfImage": 30,              # Moltes imatges (similar a 34)
        "DomainTitleMatchScore": 0.0, # El títol no té res a veure amb el domini
        "NoOfExternalRef": 115,       # Moltes refs externes (similar a 124)
        "NoOfSelfRef": 105,           # Moltes refs internes
        "URLLength": 35,              # Longitud mitjana
        "NoOfSubDomain": 1,           # www
        "IsHTTPS": 1,                 # Té cadenat (el perill actual)
        "LineOfCode": 600,            # Pàgina pesada
        "label": 1,                   # Target
        # Metadades (no afecten al model, només visual)
        "URL": "https://www.secure-update-account-verify.net",
        "Domain": "www.secure-update-account-verify.net",
        "TLD": "net",
        "Title": "Welcome to our Service - Please Login to continue" 
    }
    
    # -------------------------------------------------------------------------
    # CAS B: NOU EXEMPLE LEGÍTIM (Similars a 'xsph.ru')
    sample_legit = {
        "URLCharProb": 0.019,         # Baix
        "NoOfEmptyRef": 0,
        "HasPasswordField": 0,
        "NoOfiFrame": 0,
        "NoOfDegitsInURL": 4,         # Alguns números (similar a 7)
        "TLDLegitimateProb": 0.02,    # TLD estrany
        "HasCopyrightInfo": 0,        # Web cutre
        "CharContinuationRate": 0.85, 
        "NoOfJS": 0,                  # Zero JS
        "HasSocialNet": 0,
        "LargestLineLength": 50,      # Línies curtes
        "HasDescription": 0,
        "NoOfImage": 1,               # Potser una icona trencada
        "DomainTitleMatchScore": 100.0, # Coincidència (sovint per defecte)
        "NoOfExternalRef": 1,         # Gairebé cap enllaç
        "NoOfSelfRef": 0,
        "URLLength": 29,              # Curta
        "NoOfSubDomain": 2,           # Subdomini tècnic
        "IsHTTPS": 0,                 # No segur (HTTP)
        "LineOfCode": 15,             # Gairebé buida (similar a 9)
        "label": 0,                   # Target
        # Metadades
        "URL": "http://test01.server-arch.test.xyz",
        "Domain": "test01.server-arch.test.xyz",
        "TLD": "xyz",
        "Title": "Index of /pub/archive"
    }

    # Assegurar-nos que tenim totes les columnes necessàries (tot a 0 per defecte per la prova)
    # Aquest pas és només per fer la demo sense tenir totes les 20 columnes a mà
    all_columns = list(discretizer.bins.keys()) 
    
    full_sample_phishing = {col: sample_phishing.get(col, 0) for col in all_columns}
    full_sample_legit = {col: sample_legit.get(col, 0) for col in all_columns}

    # 3. Fem les prediccions
    
    print("--- PROVA 1: URL Sospitosa ---")
    # Reset debug counters abans de predir
    agent.q_hits = 0; agent.knn_hits = 0 
    pred, method = predict_single_url(full_sample_phishing, discretizer, agent)
    label = "PHISHING" if pred == 1 else "LEGÍTIMA"
    print(f"Resultat: {label}")
    print(f"Mètode utilitzat: {method}")
    
    print("\n--- PROVA 2: URL Segura ---")
    agent.q_hits = 0; agent.knn_hits = 0
    pred, method = predict_single_url(full_sample_legit, discretizer, agent)
    label = "PHISHING" if pred == 1 else "LEGÍTIMA"
    print(f"Resultat: {label}")
    print(f"Mètode utilitzat: {method}")