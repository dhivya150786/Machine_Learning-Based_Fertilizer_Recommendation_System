# ==========================================
# HYPERPARAMETER TUNING - LOGISTIC REGRESSION
# ==========================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("fertilizer_cleaned.csv")

# ==========================================
# INPUT FEATURES AND TARGET VARIABLE
# ==========================================

X = df.drop("Recommended_Fertilizer", axis=1)
y = df["Recommended_Fertilizer"]

# ==========================================
# ENCODE CATEGORICAL FEATURES
# ==========================================

categorical_columns = X.select_dtypes(
    include=['object', 'string']
).columns

label_encoders = {}

for col in categorical_columns:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])
    label_encoders[col] = encoder

# Encode target variable

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# ==========================================
# TRAIN TEST SPLIT
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
# HYPERPARAMETER GRID
# ==========================================

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [1000, 2000, 5000]
}

# ==========================================
# CREATE MODEL
# ==========================================

logistic_model = LogisticRegression(
    random_state=42
)

# ==========================================
# GRID SEARCH
# ==========================================

grid_search = GridSearchCV(
    estimator=logistic_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# ==========================================
# TRAIN MODEL
# ==========================================

print("\n===================================")
print("LOGISTIC REGRESSION TUNING")
print("===================================\n")

print("Training GridSearchCV...\n")

grid_search.fit(
    X_train_scaled,
    y_train
)

# ==========================================
# RESULTS
# ==========================================

print("\n===================================")
print("BEST PARAMETERS")
print("===================================\n")

print(grid_search.best_params_)

print("\n===================================")
print("BEST CROSS VALIDATION SCORE")
print("===================================\n")

print(grid_search.best_score_)

print("\n===================================")
print("BEST MODEL")
print("===================================\n")

print(grid_search.best_estimator_)

print("\n===================================")
print("LOGISTIC TUNING COMPLETED")
print("===================================")
