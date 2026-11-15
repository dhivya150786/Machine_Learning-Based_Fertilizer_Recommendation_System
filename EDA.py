# ==========================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv("fertilizer_cleaned.csv")

# Set style
sns.set_style("whitegrid")

# ==========================================
# 1. DATASET SHAPE
# ==========================================

print("\n===== Dataset Shape =====")
print(df.shape)

# ==========================================
# 2. NUMERICAL AND CATEGORICAL COLUMNS
# ==========================================

numerical_columns = df.select_dtypes(
    include=['int64', 'float64']
).columns

categorical_columns = df.select_dtypes(
    include=['object', 'string']
).columns

print("\nNumerical Columns:")
print(numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)

# ==========================================
# 3. CLASS DISTRIBUTION
# ==========================================

print("\n===== Fertilizer Distribution =====")
print(df['Recommended_Fertilizer'].value_counts())

plt.figure(figsize=(10,6))

sns.countplot(
    x='Recommended_Fertilizer',
    data=df
)

plt.xticks(rotation=45)
plt.title("Distribution of Fertilizer Classes")
plt.show()

# ==========================================
# 4. HISTOGRAMS OF NUMERICAL FEATURES
# ==========================================

for col in numerical_columns:

    plt.figure(figsize=(6,4))

    sns.histplot(
        df[col],
        bins=30,
        kde=True
    )

    plt.title(f"Distribution of {col}")
    plt.show()

# ==========================================
# 5. CORRELATION MATRIX
# ==========================================

plt.figure(figsize=(12,10))

corr_matrix = df[numerical_columns].corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Correlation Matrix")
plt.show()

# ==========================================
# 6. CATEGORICAL COLUMN DISTRIBUTION
# ==========================================

for col in categorical_columns:

    if col != 'Recommended_Fertilizer':

        plt.figure(figsize=(8,5))

        sns.countplot(
            x=col,
            data=df,
            order=df[col].value_counts().index
        )

        plt.xticks(rotation=45)
        plt.title(f"Distribution of {col}")

        plt.show()

# ==========================================
# 7. PAIRPLOT OF NUMERICAL FEATURES
# ==========================================

sns.pairplot(df[numerical_columns])
plt.show()

# ==========================================
# 8. SUMMARY STATISTICS
# ==========================================

print("\n===== Statistical Summary =====")
print(df.describe())

print("\nEDA Completed Successfully!")
