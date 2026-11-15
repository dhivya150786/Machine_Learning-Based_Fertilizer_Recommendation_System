# ==========================================
# CROSS VALIDATION FOR ALL MODELS
# ==========================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

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
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# MODELS
# ==========================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=5000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(),

    "SVM":
        SVC(
            kernel='rbf',
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            random_state=42,
            eval_metric='mlogloss'
        )
}

# ==========================================
# CROSS VALIDATION
# ==========================================

print("\n")
print("=" * 70)
print("5-FOLD CROSS VALIDATION RESULTS")
print("=" * 70)

results = []

for name, model in models.items():

    if name in [
        "Logistic Regression",
        "KNN",
        "SVM"
    ]:
        scores = cross_val_score(
            model,
            X_scaled,
            y,
            cv=5,
            scoring='accuracy'
        )
    else:
        scores = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring='accuracy'
        )

    mean_score = scores.mean()
    std_score = scores.std()

    results.append([
        name,
        mean_score,
        std_score
    ])

    print("\nModel :", name)

    print(
        "Fold Scores :",
        scores
    )

    print(
        "Average Accuracy :",
        mean_score
    )

    print(
        "Standard Deviation :",
        std_score
    )

# ==========================================
# RESULT TABLE
# ==========================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "CV Accuracy",
        "Std Deviation"
    ]
)

results_df = results_df.sort_values(
    by="CV Accuracy",
    ascending=False
)

print("\n")
print("=" * 70)
print("CROSS VALIDATION COMPARISON TABLE")
print("=" * 70)

print(results_df)

print("\n")
print("=" * 70)
print("BEST MODEL AFTER CROSS VALIDATION")
print("=" * 70)

best_model = results_df.iloc[0]

print(
    "Model :",
    best_model["Model"]
)

print(
    "Cross Validation Accuracy :",
    best_model["CV Accuracy"]
)

print(
    "Standard Deviation :",
    best_model["Std Deviation"]
)

print("\n")
print("=" * 70)
print("CROSS VALIDATION COMPLETED")
print("=" * 70)
