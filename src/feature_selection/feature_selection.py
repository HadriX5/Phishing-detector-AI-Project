import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

def select_features(numeric_df: pd.DataFrame, og_df: pd.DataFrame):
    """
    ENG:
    Performs feature selection using a Random Forest classifier and permutation importance.
    Saves the most important features to a file and displays a plot of the importances.
    Args:
        numeric_df (pd.DataFrame): DataFrame containing only numeric features.
        og_df (pd.DataFrame): Original DataFrame with all features.
    Returns:
        pd.DataFrame: DataFrame with selected numeric features and non-numeric features.
    """

    # Drop columns that are essentially cheats
    blacklist = ['label', 'URLSimilarityIndex'] 
    
    feature_cols = [c for c in numeric_df.columns if c not in blacklist]
    X = numeric_df[feature_cols]
    y = numeric_df['label']

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
    top_20_names = X.columns[top_20_idx]
    print(f"Top 20 features selected: {list(top_20_names)}")

    # Create the graphic
    top_20_data = result.importances[top_20_idx]      # For Boxplot (all repeats)
    top_20_means = result.importances_mean[top_20_idx] # For Bar Chart (average)
    
    fig, ax = plt.subplots(1, 2,figsize = (12, 8 ))
    
    ax[0].boxplot(top_20_data.T,
               vert = False,
               labels = top_20_names)
    ax[0].set_title("Permutation Importances (test set)")
    ax[0].set_xlabel("Decrease in accuracy score when feature is permuted")

    ax[1].barh(top_20_names, 
               top_20_means)
    ax[1].set_title("Top 20 Features (Mean Importance)")
    ax[1].set_xlabel("Mean Decrease in Accuracy")
    
    plt.tight_layout()
    plt.show()

    cols_to_keep = list(top_20_names) + ['label']
    df_filtered = numeric_df[cols_to_keep].copy()

    # Get the non-numeric columns from the original df
    non_numeric_df = og_df.select_dtypes(exclude=[np.number])

    # Concatenate the filtered numeric df with the non-numeric columns
    df_final = pd.concat([df_filtered, non_numeric_df], axis=1)

    # Ensure the output directory exists
    os.makedirs('data/features', exist_ok = True)

    # Save the filtered DataFrame to a Parquet file
    output_path = 'data/features/selected_features_df.parquet'
    df_final.to_parquet(output_path, index=False)
    
    print(f"DataFrame filtrat guardat a: {output_path}")
    print(f"Dimensions finals: {df_final.shape}")

    return df_final

def filter_correlated_features():
    # Pearson correlation to be implemented

    # Create a df without no-numberic columns
    df = pq.read_table('data/processed/processed_dataset.parquet').to_pandas()

    # Generic numeric filter
    numeric_df = df.select_dtypes(include = [np.number])

    correlation__matrix = numeric_df.corr('pearson').abs()

    # We set a threshold to drop correlated features
    threshold = 0.9

    # Since the correlation matrix is symmetric, we only need to check one triangle
    upper_triangle = correlation__matrix.where(
        np.triu(np.ones(correlation__matrix.shape), k = 1).astype(bool))

    # We look for columns with some value higher than the threshold
    to_drop = [column for column in upper_triangle.columns if 
               any(upper_triangle[column] > threshold)]
    
    # Protect the label column from being dropped just in case it has 
    # high correlation with some feature
    if 'label' in to_drop:
        to_drop.remove('label')

    # Drop the correlated features
    reduced_df = numeric_df.drop(columns = to_drop)
    print(reduced_df)
    return reduced_df, df