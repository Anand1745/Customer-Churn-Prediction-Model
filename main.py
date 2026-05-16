# ==========================================
# CUSTOMER CHURN PREDICTION MODEL
# ==========================================

# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# =========================
# LOAD DATASET
# =========================

print("Loading dataset...\n")

df = pd.read_csv("data/churn.csv")

print("First 5 Rows:\n")
print(df.head())

# =========================
# DATA INFORMATION
# =========================

print("\nDataset Information:\n")
print(df.info())

# =========================
# DATA CLEANING
# =========================

print("\nCleaning data...\n")

# Remove customerID column
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)

# Remove missing values
df.dropna(inplace=True)

# =========================
# ENCODING
# =========================

print("Encoding categorical columns...\n")

# Convert categorical columns using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

print("Encoded Dataset:\n")
print(df.head())

# =========================
# FEATURES & TARGET
# =========================

X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# =========================
# TRAIN TEST SPLIT
# =========================

print("\nSplitting dataset...\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# FEATURE SCALING
# =========================

print("Scaling features...\n")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODEL TRAINING
# =========================

print("Training Random Forest Model...\n")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# PREDICTION
# =========================

print("Making predictions...\n")

y_pred = model.predict(X_test)

# =========================
# MODEL EVALUATION
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("===================================")
print(f"Accuracy Score: {accuracy:.4f}")
print("===================================\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX
# =========================

print("Generating Confusion Matrix...\n")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("outputs/confusion_matrix.png")

plt.show()

# =========================
# ROC CURVE
# =========================

print("Generating ROC Curve...\n")

y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 4))

plt.plot(fpr, tpr, label=f"AUC = {roc_score:.4f}")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig("outputs/roc_curve.png")

plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================

print("Generating Feature Importance Graph...\n")

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance.head(10)
)

plt.title("Top 10 Important Features")

plt.tight_layout()

plt.savefig("outputs/feature_importance.png")

plt.show()

# =========================
# SAVE MODEL
# =========================

print("Saving model...\n")

joblib.dump(model, "models/churn_model.pkl")

print("===================================")
print("Model Saved Successfully!")
print("===================================\n")

# =========================
# SAMPLE PREDICTION
# =========================

print("Making Sample Prediction...\n")

sample_prediction = model.predict(X_test[:5])

print("Sample Predictions:")
print(sample_prediction)

print("\nProject Execution Completed Successfully!")