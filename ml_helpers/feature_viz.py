import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px

# Detect if running in Jupyter notebook
try:
    from IPython import get_ipython
    ipython_shell = get_ipython()
    if ipython_shell is not None:
        ipython_shell.run_line_magic('matplotlib', 'inline')
    else:
        matplotlib.use('Agg')  # Headless backend for scripts/tests
except (ImportError, NameError):
    matplotlib.use('Agg')

def plot_histogram(series, bins=10):
    plt.figure(figsize=(6,4))
    plt.hist(series, bins=bins, edgecolor='black')
    plt.title(f"Histogram of {series.name}")
    plt.xlabel(series.name)
    plt.ylabel("Frequency")
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame):
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

def plot_feature_importance(importances, feature_names):
    plt.figure(figsize=(8,5))
    sns.barplot(x=importances, y=feature_names)
    plt.title("Feature Importance")
    plt.show()

def plot_boxplot(df: pd.DataFrame, column: str):
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()

def plot_pairplot(df: pd.DataFrame, hue=None):
    sns.pairplot(df, hue=hue)
    plt.suptitle("Pairplot of Features", y=1.02)
    plt.show()

def plot_interactive_histogram(df: pd.DataFrame, column: str):
    fig = px.histogram(df, x=column, nbins=30, title=f"Interactive Histogram of {column}")
    fig.show()
