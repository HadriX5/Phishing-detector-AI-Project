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
    # Comptar lletres
    df['NoOfLettersInURL'] = df['URL'].str.count(r'[a-zA-Z]')

    # Comptar dígits (respectant el nom original amb typo 'Degits')
    df['NoOfDegitsInURL'] = df['URL'].str.count(r'[0-9]')
    
    # Comptar símbols específics
    df['NoOfEqualsInURL'] = df['URL'].str.count('=')
    df['NoOfQMarkInURL'] = df['URL'].str.count(r'\?')
    df['NoOfAmpersandInURL'] = df['URL'].str.count('&')
    
    # Altres caràcters especials (ni lletra ni número)
    df['NoOfOtherSpecialCharsInURL'] = df['URLLength'] - (df['NoOfLettersInURL'] + 
                                                          df['NoOfDegitsInURL'])

    # --- RATIOS ---

    safe_den = df['URLLength'].replace(0, np.nan)  # Evitar divisió per zero

    df['LetterRatioInURL'] = df['NoOfLettersInURL'] / safe_den
    df['DegitRatioInURL'] = df['NoOfDegitsInURL'] / safe_den
    df['SpacialCharRatioInURL'] = df['NoOfOtherSpecialCharsInURL'] / safe_den

    return df