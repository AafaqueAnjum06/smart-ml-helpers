# ml_helpers/feature_viz.py

import matplotlib.pyplot as plt

def plot_histogram(series, bins=10):
    """
    Plot a simple histogram for a pandas Series.
    """
    plt.figure(figsize=(6,4))
    plt.hist(series, bins=bins, edgecolor='black')
    plt.title(f"Histogram of {series.name}")
    plt.xlabel(series.name)
    plt.ylabel("Frequency")
    plt.show()
