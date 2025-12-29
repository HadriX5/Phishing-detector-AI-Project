import pandas as pd
import numpy as np

def ammend_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ammend column values as the encoding repair process might have altered them.

    Parameters:
    df (pd.DataFrame): The input DataFrame with original column values.

    Returns:
    pd.DataFrame: The DataFrame with ammended columns.
    """
    df = df.copy()
    # --- BASIC LENGTHS ---
    df['URLLength'] = df['URL'].str.len()
    df['DomainLength'] = df['Domain'].str.len()

    # --- COUNTS ---
    # Count letters
    df['NoOfLettersInURL'] = df['URL'].str.count(r'[a-zA-Z]')

    # Count digits (respecting the original typo 'Degits')
    df['NoOfDegitsInURL'] = df['URL'].str.count(r'[0-9]')
    
    # Count specific symbols
    df['NoOfEqualsInURL'] = df['URL'].str.count('=')
    df['NoOfQMarkInURL'] = df['URL'].str.count(r'\?')
    df['NoOfAmpersandInURL'] = df['URL'].str.count('&')
    
    # Other special characters (neither letter nor number)
    df['NoOfOtherSpecialCharsInURL'] = df['URLLength'] - (df['NoOfLettersInURL'] + 
                                                          df['NoOfDegitsInURL'])

    # --- RATIOS ---

    safe_den = df['URLLength'].replace(0, np.nan)  # Avoid division by zero

    df['LetterRatioInURL'] = df['NoOfLettersInURL'] / safe_den
    df['DegitRatioInURL'] = df['NoOfDegitsInURL'] / safe_den
    df['SpacialCharRatioInURL'] = df['NoOfOtherSpecialCharsInURL'] / safe_den

    return df