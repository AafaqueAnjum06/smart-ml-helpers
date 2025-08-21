# tests/conftest.py
import pytest
import pandas as pd
import numpy as np

# Generic DataFrame for most tests
@pytest.fixture
def df():
    return pd.DataFrame({
        "age": [25, 30, np.nan, 40],
        "salary": [50000, 60000, 55000, np.nan],
        "category": ["A", "B", "A", "B"],
        "city": ["LA", "NY", "SF", "LA"],
        "target": [1, 0, 1, 0],
        "x": [1, 2, 3, 4],
        "y": [4, 3, 2, 1]
    })

# Fixture for column name in feature_viz tests
@pytest.fixture
def column():
    return "age"

# Fixture for columns list in interactive tests
@pytest.fixture
def columns():
    return ["age", "salary", "x", "y"]

# Fixtures for model_viz tests
@pytest.fixture
def y_true():
    return np.array([0, 1, 0, 1])

@pytest.fixture
def y_pred():
    return np.array([0, 1, 1, 0])
