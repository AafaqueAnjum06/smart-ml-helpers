# ml_helpers/feature_viz.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_histogram(series, bins=10):
    """Histogram for a single numeric column."""
    plt.figure(figsize=(6,4))
    plt.hist(series, bins=bins, edgecolor='black')
    plt.title(f"Histogram of {series.name}")
    plt.xlabel(series.name)
    plt.ylabel("Frequency")
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame):
    """Correlation heatmap for numeric features."""
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

def plot_feature_importance(importances, feature_names):
    """Bar chart of feature importance."""
    plt.figure(figsize=(8,5))
    sns.barplot(x=importances, y=feature_names)
    plt.title("Feature Importance")
    plt.show()

def plot_boxplot(df: pd.DataFrame, column: str):
    """Boxplot for a single numeric column."""
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.xlabel(column)
    plt.show()

def plot_pairplot(df: pd.DataFrame, hue=None):
    """Pairplot for visualizing relationships between numeric features."""
    sns.pairplot(df, hue=hue)
    plt.suptitle("Pairplot of Features", y=1.02)
    plt.show()

def plot_distribution(series, kind='hist', bins=10):
    """Plot distribution of a numeric series."""
    plt.figure(figsize=(6,4))
    if kind == 'hist':
        plt.hist(series, bins=bins, edgecolor='black')
        plt.title(f"Histogram of {series.name}")
    elif kind == 'kde':
        sns.kdeplot(series, shade=True)
        plt.title(f"Kernel Density Estimate of {series.name}")
    else:
        raise ValueError("kind must be 'hist' or 'kde'")
    
    plt.xlabel(series.name)
    plt.ylabel("Density")
    plt.show()

def plot_scatter_matrix(df: pd.DataFrame, hue=None):
    """Scatter matrix for visualizing pairwise relationships."""
    pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10), diagonal='kde', c=hue)
    plt.suptitle("Scatter Matrix of Features", y=1.02)
    plt.show()

def plot_feature_distribution(df: pd.DataFrame, column: str):
    """Plot distribution of a specific feature."""
    plt.figure(figsize=(6,4))
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

def plot_categorical_distribution(df: pd.DataFrame, column: str):
    """Bar plot for categorical feature distribution."""
    plt.figure(figsize=(8,5))
    sns.countplot(y=df[column], order=df[column].value_counts().index)
    plt.title(f"Distribution of {column}")
    plt.xlabel("Count")
    plt.ylabel(column)
    plt.show()

def plot_numeric_distribution(df: pd.DataFrame, column: str):
    """Histogram for numeric feature distribution."""
    plt.figure(figsize=(6,4))
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()