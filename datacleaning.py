# ==========================================
# DATA CLEANING FOR FERTILIZER DATASET
# ==========================================

import pandas as pd
import numpy as np

# ==========================================
# LOAD DATASET
# ==========================================
df = pd.read_csv("fertilizer_recommendation.csv")

# ==========================================
# 1. CHECK MISSING VALUES
# ==========================================
print("\n===== Missing Values =====")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# ==========================================
# 2. HANDLE MISSING VALUES
# ==========================================

# Numerical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Categorical columns
categorical_columns = df.select_dtypes(include=['object', 'string']).columns

# Fill numerical columns with median
for col in numerical_columns:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after handling:")
print(df.isnull().sum())

# ==========================================
# 3. CHECK DUPLICATE ROWS
# ==========================================
print("\n===== Duplicate Rows =====")
print("Number of duplicates:", df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

print("\nShape after removing duplicates:")
print(df.shape)

# ==========================================
# 4. CHECK DATA TYPES
# ==========================================
print("\n===== Data Types =====")
print(df.dtypes)

# ==========================================
# 5. CHECK UNIQUE VALUES
# ==========================================
print("\n===== Number of Unique Values =====")

for col in df.columns:
    print(col, ":", df[col].nunique())

# ==========================================
# 6. STATISTICAL SUMMARY
# ==========================================
print("\n===== Statistical Summary =====")
print(df.describe())

# ==========================================
# 7. CHECK INVALID VALUES
# ==========================================
print("\n===== Invalid Value Checks =====")

# Soil pH should be between 0 and 14
if "Soil_pH" in df.columns:
    invalid_ph = df[(df["Soil_pH"] < 0) | (df["Soil_pH"] > 14)]
    print("Invalid Soil_pH values:", len(invalid_ph))

# Humidity should be between 0 and 100
if "Humidity" in df.columns:
    invalid_humidity = df[(df["Humidity"] < 0) | (df["Humidity"] > 100)]
    print("Invalid Humidity values:", len(invalid_humidity))

# Rainfall should not be negative
if "Rainfall" in df.columns:
    invalid_rainfall = df[df["Rainfall"] < 0]
    print("Negative Rainfall values:", len(invalid_rainfall))

# Temperature should not be negative
if "Temperature" in df.columns:
    invalid_temp = df[df["Temperature"] < 0]
    print("Negative Temperature values:", len(invalid_temp))

# ==========================================
# 8. OUTLIER DETECTION USING IQR
# ==========================================
print("\n===== Outlier Detection =====")

for col in numerical_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"{col}: {len(outliers)} outliers")

# ==========================================
# 9. CHECK FINAL SHAPE
# ==========================================
print("\n===== Final Dataset Shape =====")
print(df.shape)

# ==========================================
# 10. SAVE CLEANED DATASET
# ==========================================
df.to_csv("fertilizer_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully as:")
print("fertilizer_cleaned.csv")
