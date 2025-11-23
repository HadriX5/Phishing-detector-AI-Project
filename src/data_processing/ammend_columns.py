import pandas as pd

def ammend_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ammend column values as the encoding repair process might have altered them.

    Parameters:
    df (pd.DataFrame): The input DataFrame with original column values.

    Returns:
    pd.DataFrame: The DataFrame with ammended columns.
    """

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
    df['LetterRatioInURL'] = df['NoOfLettersInURL'] / df['URLLength']
    df['DegitRatioInURL'] = df['NoOfDegitsInURL'] / df['URLLength']
    df['SpacialCharRatioInURL'] = df['NoOfOtherSpecialCharsInURL'] / df['URLLength']

    return df