# code cleared
# ml_helpers/interactive.py

import matplotlib.pyplot as plt
from ipywidgets import interact
import pandas as pd

def interactive_plot(df: pd.DataFrame):
    """
    Interactive histogram for numeric columns.
    """
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
