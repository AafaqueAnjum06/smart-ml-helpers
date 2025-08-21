import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

# Detect if running in Jupyter notebook
try:
    from IPython import get_ipython
    ipython_shell = get_ipython()
    if ipython_shell is not None:
        ipython_shell.run_line_magic('matplotlib', 'inline')
    else:
        matplotlib.use('Agg')
except (ImportError, NameError):
    matplotlib.use('Agg')

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def plot_roc(y_true_bin, y_pred_bin, classes):
    import numpy as np
    plt.figure(figsize=(6,5))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:,i], y_pred_bin[:,i])
        plt.plot(fpr, tpr, label=f"Class {cls}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def plot_precision_recall(y_true_bin, y_pred_bin, classes):
    plt.figure(figsize=(6,5))
    for i, cls in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:,i], y_pred_bin[:,i])
        plt.plot(recall, precision, label=f"Class {cls}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()
