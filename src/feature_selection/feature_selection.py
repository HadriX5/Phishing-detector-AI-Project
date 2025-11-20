import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt


def feature_selection():
    """
    Realitza la selecció de característiques utilitzant un classificador Random Forest
    i la importància per permutació. Desa les característiques més importants en un fitxer
    i mostra un gràfic de les importàncies.
    """
    X = pd.read_csv('../data/processed/processed_dataset.csv')

    # Drop label as we don't need it to select features
    y = X.drop(['label'], axis = 1)
    features = X.columns

    # -- TRAINING RANDOM_FOREST_CLASSIFIER --

    # Separate df in train - test to evade overfitting
    # Parameters:
    #     - test_size: Sample remaining in test
    #     - random_state: Usual int for repeatibility
    #     - stratify: Assure balancing in both datasets:
    #         (Meaning, if 10% OG dataset is class X, 
    #         there will be 10% of class X in train, 
    #         regardless of nº of samples in OG)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.2, 
                                                        random_state = 42, 
                                                        stratify = y)

    # Define forest
    rf = RandomForestClassifier(n_estimators = 500, 
                                random_state = 42, 
                                n_jobs = -1)

    # Apply the forest to our dataset
    rf.fit(X_train, y_train)

    # -- FEATURE IMPORTANCE --
    # Repeat the rf 20 times with X_test and y_test so we have an std to observe
    result = permutation_importance(rf, X_test, y_test,
                                n_repeats = 20,
                                random_state=42,
                                    n_jobs=-1,
                                    scoring='accuracy')

    # -- RESULTS --

    # Get the features indexes
    sorted_indexes = result.importances_mean.argsort()

    # Get the top 20
    features = features[sorted_indexes][-20:]
    features_values = result.importances_mean[sorted_idx][-20:]
    features_std = result.importances_std[sorted_idx][-20:]

    # Save the features
    top_features_list = [feature_names[i] for i in sorted_idx[::-1][:15]]
    np.save('../models/features.npy', features_list)

    # 
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.boxplot(
        result.importances[sorted_indexes][-20:].T,
        vert = False,
        labels = features_names
    )
    ax.set_title("Permutation Importance (Test Set)")
    ax.set_xlabel("Disminució de la precisió del model en permutar la variable")
    plt.tight_layout()
    plt.show()