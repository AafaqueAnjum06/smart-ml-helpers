# ml_helpers/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def clean_data(df: pd.DataFrame, drop_thresh=0.5) -> pd.DataFrame:
    """
    Clean dataset:
    - Drop columns with > drop_thresh missing values
    - Fill numeric NaNs with mean
    - Fill categorical NaNs with mode
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isna().mean() > drop_thresh:
            df.drop(columns=[col], inplace=True)
        elif df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode categorical columns."""
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale numeric features only."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def split_features_target(df: pd.DataFrame, target_col: str):
    """Split dataframe into X, y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def automated_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame for automated EDA."""
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'missing': df.isna().sum(),
        'unique': df.nunique(),
        'mean': df.mean(numeric_only=True),
        'std': df.std(numeric_only=True),
        'min': df.min(numeric_only=True),
        'max': df.max(numeric_only=True)
    })
    return summary

def suggest_ml_prep(df: pd.DataFrame) -> dict:
    """Suggest ML preprocessing steps based on data types."""
    suggestions = {}
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            suggestions[col] = ['scale', 'check_outliers']
        else:
            suggestions[col] = ['encode_categorical']
    return suggestions
