"""
Diabetes Prediction - Model Training Script
Dataset: Pima Indians Diabetes Database
Algorithm: Gaussian Naive Bayes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# Remove Pregnancies for gender-neutral predictions
df = df.drop('Pregnancies', axis=1)

# Replace impossible zero-values with column medians (excluding Outcome)
zero_not_allowed = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_not_allowed:
    df[col] = df[col].replace(0, df[col].median())

# Split features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print("=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Specificity
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
print(f"Specificity: {specificity:.4f}")

# Print model parameters for frontend use
print("\n" + "=" * 50)
print("MODEL PARAMETERS (copy these to index.html)")
print("=" * 50)

features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
            'BMI', 'DiabetesPedigreeFunction', 'Age']

print("\nmeans: {")
print("    0: [", end="")
for i, feat in enumerate(features):
    print(f"{model.theta_[0][i]:.2f}", end="")
    if i < len(features) - 1:
        print(", ", end="")
print("],")
print("    1: [", end="")
for i, feat in enumerate(features):
    print(f"{model.theta_[1][i]:.2f}", end="")
    if i < len(features) - 1:
        print(", ", end="")
print("]")
print("},")

print("\nvariances: {")
print("    0: [", end="")
for i, feat in enumerate(features):
    print(f"{model.var_[0][i]:.2f}", end="")
    if i < len(features) - 1:
        print(", ", end="")
print("],")
print("    1: [", end="")
for i, feat in enumerate(features):
    print(f"{model.var_[1][i]:.2f}", end="")
    if i < len(features) - 1:
        print(", ", end="")
print("]")
print("},")

print(f"\nprior: {{ 0: {model.class_prior_[0]:.4f}, 1: {model.class_prior_[1]:.4f} }}")

print("\n" + "=" * 50)
print("CLASS DISTRIBUTION")
print("=" * 50)
print(f"Train: {len(y_train)} samples (Class 0: {sum(y_train==0)}, Class 1: {sum(y_train==1)})")
print(f"Test:  {len(y_test)} samples (Class 0: {sum(y_test==0)}, Class 1: {sum(y_test==1)})")