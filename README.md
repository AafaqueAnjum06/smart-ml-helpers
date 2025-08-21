# Smart ML Helpers 🚀

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Pytest-yellow)](https://docs.pytest.org/)

**Smart ML Helpers** is a Python library that makes **Machine Learning workflows smarter and faster**. It combines **data preprocessing, feature visualization, and model evaluation** with **AI-driven enhancements**, providing automatic suggestions for preprocessing and reducing manual effort.

---

## 🌟 Why Use Smart ML Helpers?

* Save time on repetitive ML preprocessing tasks
* Visualize features and models interactively
* Leverage AI suggestions for better data preparation
* Fully tested and modular, ready to integrate into any ML pipeline

---

## 🔥 Features

### 1. Data Preprocessing

* Clean datasets intelligently, handling missing values and outliers
* Encode categorical variables automatically
* Scale numeric features efficiently
* Split features and target columns
* Automated EDA summaries
* **AI-powered preprocessing suggestions** based on data types

### 2. Feature Visualization

* Histograms, boxplots, pairplots, and correlation matrices
* Feature importance visualization
* **Interactive Plotly histograms** for exploratory analysis

### 3. Model Evaluation

* Confusion matrices for classification models
* ROC curves for multiclass evaluation
* Precision-Recall curves for detailed analysis

### 4. AI Enhancements

* Suggests preprocessing steps automatically
* Detects columns needing scaling, encoding, or outlier checks
* Minimizes human errors in data preparation

---

## ⚡ Installation

```bash
git clone https://github.com/AafaqueAnjum06/smart-ml-helpers.git
cd smart-ml-helpers
pip install -r requirements.txt
```

---

## 🛠 Usage

### 1. Import Helpers

```python
from ml_helpers import (
    clean_data, encode_categorical, scale_features, split_features_target,
    automated_eda, suggest_ml_prep,
    plot_histogram, plot_correlation_matrix, plot_boxplot, plot_pairplot, plot_interactive_histogram,
    plot_confusion, plot_roc, plot_precision_recall,
    interactive_plot, interactive_scatter
)
```

### 2. Load Data

```python
import seaborn as sns
df = sns.load_dataset("iris")
```

### 3. Preprocess Data

```python
df_clean = clean_data(df)
df_encoded = encode_categorical(df_clean)
X, y = split_features_target(df_encoded, target_col='species')
X_scaled = scale_features(X)
```

### 4. Visualize Features

```python
plot_histogram(X_scaled['petal_length'])
plot_boxplot(X_scaled, column='sepal_length')
plot_interactive_histogram(X_scaled, column='petal_length')
interactive_plot(X_scaled)
interactive_scatter(X_scaled)
```

### 5. Evaluate Models

```python
plot_confusion(y_true, y_pred)
plot_roc(y_true_bin, y_pred_bin, classes)
plot_precision_recall(y_true_bin, y_pred_bin, classes)
```

---

## ✅ Testing

```bash
pytest -v
```

---

## 🤝 Contribution

1. Fork the repository
2. Implement improvements or new features
3. Add corresponding tests
4. Submit a pull request

---

## 📄 License

MIT License © 2025 Aafaque Anjum

---

## 🌐 Notes

* Optimized for Jupyter Notebooks and Python scripts
* AI-enhanced preprocessing tips make ML pipeline setup faster and easier.
* Fully modular design for seamless integration into ML pipelines
