# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# ==========================================
# LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv("fertilizer_cleaned.csv")

# ==========================================
# SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Recommended_Fertilizer", axis=1)
y = df["Recommended_Fertilizer"]

# ==========================================
# IDENTIFY CATEGORICAL COLUMNS
# ==========================================

categorical_columns = X.select_dtypes(
    include=['object', 'string']
).columns

# ==========================================
# ENCODE CATEGORICAL FEATURES
# ==========================================

label_encoders = {}

for col in categorical_columns:

    le = LabelEncoder()

    X[col] = le.fit_transform(X[col])

    label_encoders[col] = le

# ==========================================
# ENCODE TARGET VARIABLE
# ==========================================

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(y)

label_encoders["Recommended_Fertilizer"] = target_encoder

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
# DISPLAY SHAPES
# ==========================================

print("\n===== Training Data Shape =====")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\n===== Testing Data Shape =====")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# ==========================================
# SAVE SPLIT DATASETS
# ==========================================

joblib.dump(X_train, "X_train.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(y_test, "y_test.pkl")

print("\nTrain-test split completed successfully!")

print("\nFiles saved:")
print("X_train.pkl")
print("X_test.pkl")
print("y_train.pkl")
print("y_test.pkl")
