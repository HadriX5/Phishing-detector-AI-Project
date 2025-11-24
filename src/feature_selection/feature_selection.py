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
    df = pd.read_parquet('data/processed/processed_dataset.parquet')

    # Drop label as we don't need it to select features
    X = df.drop('label', axis = 1)
    y = df['label']

    feature_names = X.columns

    # -- TRAINING RANDOM_FOREST_CLASSIFIER --

    # Separate df in train - test to evade overfitting
    # Parameters:
    #     - test_size: Sample remaining in test
    #     - random_state: Usual int for repeatibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.2, 
                                                        random_state = 42)

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
                                random_state = 42,
                                n_jobs = -1,
                                scoring = 'accuracy')

    # -- RESULTS --

    # Get the sorted indexes by importance (from lower to higher)
    sorted_indexes = result.importances_mean.argsort()

    # Aisle the top 20 for visualization
    top_20_idx = sorted_indexes[-20:]

    # Create the graphic
    fig, ax = plt.subplots(figsize = (12, 8 ))
    ax.boxplot(result.importances[top_20_idx].T,
               vert = False,
               labels = feature_names[top_20_idx])
    ax.set_title("Permutation Importances (test set)")
    ax.set_xlabel("Decrease in accuracy score when feature is permuted")
    plt.tight_layout()
    plt.show()

    # Save the top 15 features to a .npy
    top_15_idx = sorted_indexes[-15:][::-1]
    top_15_features = feature_names[top_15_idx]

    print(f"Top 15 features selected: {top_15_features}")
    np.save('../data/features/features.npy', top_15_features)