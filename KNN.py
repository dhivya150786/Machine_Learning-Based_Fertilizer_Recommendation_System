# ===================================
# K-NEAREST NEIGHBORS (KNN)
# ===================================

import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
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
# TRAIN TEST SPLIT
# ===================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ===================================
# FEATURE SCALING
# ===================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===================================
# CREATE MODEL
# ===================================

knn_model = KNeighborsClassifier(
    n_neighbors=5
)

# ===================================
# TRAIN MODEL
# ===================================

knn_model.fit(X_train_scaled, y_train)

# ===================================
# PREDICTIONS
# ===================================

y_pred = knn_model.predict(X_test_scaled)

# ===================================
# EVALUATION
# ===================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average='weighted',
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average='weighted',
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted',
    zero_division=0
)

# ===================================
# RESULTS
# ===================================

print("\n===== KNN RESULTS =====")

print("Accuracy : {:.15f}".format(accuracy))
print("Precision: {:.15f}".format(precision))
print("Recall   : {:.15f}".format(recall))
print("F1 Score : {:.15f}".format(f1))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0,
        digits=10
    )
)

print("\n===== KNN MODEL COMPLETED =====")
