# ===================================
# GRADIENT BOOSTING CLASSIFIER
# ===================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ===================================
# LOAD DATASET
# ===================================

df = pd.read_csv("fertilizer_cleaned.csv")

# ===================================
# INPUTS AND OUTPUT
# ===================================

X = df.drop("Recommended_Fertilizer", axis=1)
y = df["Recommended_Fertilizer"]

# ===================================
# ENCODE CATEGORICAL VARIABLES
# ===================================

categorical_columns = X.select_dtypes(
    include=['object', 'string']
).columns

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Encode target variable
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# ===================================
# TRAIN-TEST SPLIT
# ===================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ===================================
# CREATE MODEL
# ===================================

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

# ===================================
# TRAIN MODEL
# ===================================

gb_model.fit(X_train, y_train)

# ===================================
# PREDICTIONS
# ===================================

y_pred = gb_model.predict(X_test)

# ===================================
# EVALUATION
# ===================================

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("GRADIENT BOOSTING CLASSIFIER")
print("===================================")

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n===================================")
print("GRADIENT BOOSTING MODEL COMPLETED")
print("===================================")
