import pytest
import pandas as pd
import numpy as np
from ml_helpers import preprocessing

@pytest.fixture
def df():
    return pd.DataFrame({
        'age': [25, 30, np.nan, 40],
        'salary': [50000, 60000, 55000, np.nan],
        'category': ['A', 'B', 'A', 'B'],
        'target': [1, 0, 1, 0]
    })

def test_clean_data(df):
    df_clean = preprocessing.clean_data(df)
    assert df_clean.isna().sum().sum() == 0  # no missing values

def test_encode_categorical(df):
    df_encoded = preprocessing.encode_categorical(df)
    # check that category column is numeric
    assert np.issubdtype(df_encoded['category'].dtype, np.number)

def test_scale_features(df):
    df_encoded = preprocessing.encode_categorical(df)
    df_scaled = preprocessing.scale_features(df_encoded)
    # check mean ~0 and std ~1 for numeric columns
    numeric_cols = df_scaled.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        assert abs(df_scaled[col].mean()) < 1e-6
        assert abs(df_scaled[col].std(ddof=0) - 1) < 1e-6

def test_split_features_target(df):
    X, y = preprocessing.split_features_target(df, 'target')
    assert 'target' not in X.columns
    assert y.name == 'target'

def test_automated_eda(df):
    summary = preprocessing.automated_eda(df)
    assert 'dtype' in summary.columns
    assert 'missing' in summary.columns

def test_suggest_ml_prep(df):
    suggestions = preprocessing.suggest_ml_prep(df)
    assert 'age' in suggestions
    assert 'category' in suggestions
