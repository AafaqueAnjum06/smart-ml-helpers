# ml_helpers/interactive.py

import matplotlib.pyplot as plt
from ipywidgets import interact
import pandas as pd
import seaborn as sns

def interactive_plot(df: pd.DataFrame):
    """Interactive histogram for numeric columns."""
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if not num_cols:
        print("No numeric columns to display")
        return

    def plot(col):
        plt.figure(figsize=(6,4))
        plt.hist(df[col], bins=10, edgecolor='black')
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

    interact(plot, col=num_cols)

def interactive_scatter(df: pd.DataFrame):
    """Interactive scatter plot for any two numeric columns."""
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if len(num_cols) < 2:
        print("Not enough numeric columns")
        return

    def plot(x_col, y_col):
        plt.figure(figsize=(6,4))
        sns.scatterplot(data=df, x=x_col, y=y_col)
        plt.title(f"{y_col} vs {x_col}")
        plt.show()

    interact(plot, x_col=num_cols, y_col=num_cols)
