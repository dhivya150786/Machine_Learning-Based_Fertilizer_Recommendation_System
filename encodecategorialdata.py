# ==========================================
# ENCODE CATEGORICAL VARIABLES
# ==========================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# ==========================================
# LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv("fertilizer_cleaned.csv")

# ==========================================
# SEPARATE INPUT FEATURES AND TARGET
# ==========================================

X = df.drop("Recommended_Fertilizer", axis=1)
y = df["Recommended_Fertilizer"]

# ==========================================
# IDENTIFY CATEGORICAL COLUMNS
# ==========================================

categorical_columns = X.select_dtypes(
    include=['object', 'string']
).columns

print("\n===== Categorical Columns =====")
print(categorical_columns)

# ==========================================
# LABEL ENCODE INPUT FEATURES
# ==========================================

label_encoders = {}

for col in categorical_columns:

    le = LabelEncoder()

    X[col] = le.fit_transform(X[col])

    label_encoders[col] = le

# ==========================================
# LABEL ENCODE TARGET VARIABLE
# ==========================================

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(y)

label_encoders["Recommended_Fertilizer"] = target_encoder

# ==========================================
# DISPLAY ENCODED DATASET
# ==========================================

print("\n===== Encoded Features (First 5 Rows) =====")
print(X.head())

print("\n===== Encoded Target (First 10 Values) =====")
print(y[:10])

# ==========================================
# DISPLAY MAPPINGS
# ==========================================

print("\n===== Encoding Mappings =====")

for col, encoder in label_encoders.items():

    print(f"\n{col}")

    mapping = dict(zip(
        encoder.classes_,
        encoder.transform(encoder.classes_)
    ))

    print(mapping)

# ==========================================
# SAVE ENCODERS
# ==========================================

joblib.dump(label_encoders, "label_encoders.pkl")

print("\nEncoders saved successfully!")
