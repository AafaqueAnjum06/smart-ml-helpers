# tests/test_feature_viz.py
import pytest
import pandas as pd
import numpy as np
from ml_helpers import feature_viz

# Sample DataFrame
@pytest.fixture
def df():
    return pd.DataFrame({
        'age': [25, 30, 35, 40],
        'salary': [50000, 60000, 55000, 65000],
        'category': ['A', 'B', 'A', 'B']
    })

def test_plot_histogram(df):
    feature_viz.plot_histogram(df['age'])
    feature_viz.plot_histogram(df['salary'])

def test_plot_correlation_matrix(df):
    feature_viz.plot_correlation_matrix(df[['age', 'salary']])

def test_plot_feature_importance():
    importances = [0.2, 0.5, 0.3]
    feature_names = ['f1', 'f2', 'f3']
    feature_viz.plot_feature_importance(importances, feature_names)

def test_plot_boxplot(df):
    feature_viz.plot_boxplot(df, 'age')
    feature_viz.plot_boxplot(df, 'salary')

def test_plot_pairplot(df):
    feature_viz.plot_pairplot(df, hue='category')

def test_plot_interactive_histogram(df):
    # Should not open GUI during tests
    feature_viz.plot_interactive_histogram(df, 'age')
