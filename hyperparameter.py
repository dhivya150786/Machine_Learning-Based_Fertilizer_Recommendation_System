# ==========================================
# GRADIENT BOOSTING HYPERPARAMETER TUNING
# ==========================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("fertilizer_cleaned.csv")

# ==========================================
# INPUTS AND OUTPUT
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
# PARAMETER GRID
# ==========================================

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}

# ==========================================
# MODEL
# ==========================================

gb_model = GradientBoostingClassifier(
    random_state=42
)

# ==========================================
# GRID SEARCH
# ==========================================

grid_search = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=1,      # avoids memory issues
    verbose=1
)

print("Running Hyperparameter Tuning...\n")

grid_search.fit(X_train, y_train)

# ==========================================
# RESULTS
# ==========================================

print("\n===================================")
print("BEST PARAMETERS")
print("===================================")
print(grid_search.best_params_)

print("\n===================================")
print("BEST CROSS VALIDATION ACCURACY")
print("===================================")
print(grid_search.best_score_)

print("\n===================================")
print("BEST MODEL")
print("===================================")
print(grid_search.best_estimator_)
