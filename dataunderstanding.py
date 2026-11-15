import pandas as pd

# ==============================
# LOAD DATASET
# ==============================
df = pd.read_csv("fertilizer_recommendation.csv")

# ==============================
# FIRST 5 ROWS
# ==============================
print("\n===== First 5 Rows =====")
print(df.head())

# ==============================
# LAST 5 ROWS
# ==============================
print("\n===== Last 5 Rows =====")
print(df.tail())

# ==============================
# DATASET SHAPE
# ==============================
print("\n===== Dataset Shape =====")
print("Rows and Columns:", df.shape)
print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

# ==============================
# COLUMN NAMES
# ==============================
print("\n===== Column Names =====")
print(df.columns.tolist())

# ==============================
# DATASET INFO
# ==============================
print("\n===== Dataset Information =====")
df.info()

# ==============================
# DATA TYPES
# ==============================
print("\n===== Data Types =====")
print(df.dtypes)

# ==============================
# STATISTICAL SUMMARY
# ==============================
print("\n===== Numerical Summary =====")
print(df.describe())

# ==============================
# SUMMARY INCLUDING CATEGORICAL COLUMNS
# ==============================
print("\n===== Complete Summary =====")
print(df.describe(include='all'))

# ==============================
# NUMERICAL COLUMNS
# ==============================
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

print("\n===== Numerical Columns =====")
print(numerical_columns.tolist())

# ==============================
# CATEGORICAL COLUMNS
# ==============================
categorical_columns = df.select_dtypes(include=['object', 'string']).columns

print("\n===== Categorical Columns =====")
print(categorical_columns.tolist())

# ==============================
# NUMBER OF FEATURES
# ==============================
print("\nNumber of Numerical Features:", len(numerical_columns))
print("Number of Categorical Features:", len(categorical_columns))

# ==============================
# TARGET VARIABLE
# ==============================
target_column = "Recommended_Fertilizer"

print("\n===== Target Variable =====")
print(target_column)

# ==============================
# UNIQUE CLASSES
# ==============================
print("\n===== Unique Fertilizer Classes =====")
print(df[target_column].unique())

# ==============================
# NUMBER OF CLASSES
# ==============================
print("\n===== Number of Classes =====")
print(df[target_column].nunique())

# ==============================
# CLASS DISTRIBUTION
# ==============================
print("\n===== Class Distribution =====")
print(df[target_column].value_counts())

# ==============================
# MEMORY USAGE
# ==============================
print("\n===== Memory Usage =====")
print(df.memory_usage(deep=True))

print("\nTotal Memory Used:",
      round(df.memory_usage(deep=True).sum() / 1024, 2), "KB")
