# code cleared
# ml_helpers/preprocessing.py

import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple cleaning:
    - Fill numeric NaNs with mean
    - Fill non-numeric NaNs with 'Unknown'
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna("Unknown")
    return df
