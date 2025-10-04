Wine Quality Prediction
Project Overview

Predicts wine quality using the WineQT dataset based on chemical properties. 
Supports binary classification (good vs bad wine) and multi-class classification 
(quality scores 3–8) using machine learning models.

Features

Input: Alcohol, acidity, citric acid, sulphates, density, pH, etc.

Target: Wine quality score

Binary: Good (≥6) / Bad (<6)

Models Used

Random Forest

SGD Classifier

SVC

Evaluation Metrics

Accuracy | Precision | Recall | F1-Score

Cross-Validation | Confusion Matrix

Visualizations

Quality distribution

Correlation heatmap

Feature boxplots

Pairplots

Random Forest feature importance

Installation
pip install pandas numpy matplotlib seaborn scikit-learn

Usage

Place WineQT.csv in a known directory.

Update the file path in the script:

file_path = r"C:\path\to\WineQT.csv"


Run the script in IDLE or any Python IDE:

python wine_quality_analysis.py

Key Insights

Dataset: 11 features, quality scores 3–8

Random Forest performs best overall

Top influencing features: alcohol, volatile acidity, sulphates

Author

Harsh Raj
