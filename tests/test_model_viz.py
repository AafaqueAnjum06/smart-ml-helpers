# tests/test_model_viz.py
import pytest
import numpy as np
from ml_helpers import model_viz

@pytest.fixture
def y_data():
    # 2-class example
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    return y_true, y_pred

@pytest.fixture
def y_bin_data():
    # 2-class one-hot encoded
    y_true_bin = np.array([[1,0],[0,1],[0,1],[1,0]])
    y_pred_bin = np.array([[0.9,0.1],[0.2,0.8],[0.6,0.4],[0.7,0.3]])
    classes = [0, 1]
    return y_true_bin, y_pred_bin, classes

def test_plot_confusion(y_data):
    y_true, y_pred = y_data
    model_viz.plot_confusion(y_true, y_pred)

def test_plot_roc(y_bin_data):
    y_true_bin, y_pred_bin, classes = y_bin_data
    model_viz.plot_roc(y_true_bin, y_pred_bin, classes)

def test_plot_precision_recall(y_bin_data):
    y_true_bin, y_pred_bin, classes = y_bin_data
    model_viz.plot_precision_recall(y_true_bin, y_pred_bin, classes)
