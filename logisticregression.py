# ==========================================
# LOGISTIC REGRESSION MODEL
# ==========================================

import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("fertilizer_cleaned.csv")

# ==========================================
# SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Recommended_Fertilizer", axis=1)
y = df["Recommended_Fertilizer"]

# ==========================================
# ENCODE CATEGORICAL VARIABLES
# ==========================================

categorical_columns = X.select_dtypes(
    include=['object', 'string']
).columns

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Encode target variable
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# CREATE LOGISTIC REGRESSION MODEL
# ==========================================

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# ==========================================
# TRAIN MODEL
# ==========================================

lr_model.fit(X_train_scaled, y_train)

# ==========================================
# MAKE PREDICTIONS
# ==========================================

y_pred = lr_model.predict(X_test_scaled)

# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average='weighted'
)

recall = recall_score(
    y_test,
    y_pred,
    average='weighted'
)

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

# ==========================================
# PRINT RESULTS
# ==========================================

print("\n===== LOGISTIC REGRESSION RESULTS =====")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))
