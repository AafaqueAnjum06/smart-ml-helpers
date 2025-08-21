# ml_helpers/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: numeric → mean, categorical → 'Unknown'."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna("Unknown")
    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all categorical columns."""
    return pd.get_dummies(df)

def scale_features(df: pd.DataFrame, method="standard") -> pd.DataFrame:
    """Scale numeric features using StandardScaler or MinMaxScaler."""
    df = df.copy()
    num_cols = df.select_dtypes(include='number').columns
    if method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def split_features_target(df: pd.DataFrame, target_col: str):
    """Split DataFrame into X (features) and y (target)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
