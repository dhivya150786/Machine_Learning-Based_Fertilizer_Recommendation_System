# ==========================================
# UNIVERSAL FERTILIZER PREDICTION SYSTEM
# ==========================================

import joblib
import pandas as pd

# ==========================================
# LOAD SAVED FILES
# ==========================================

model = joblib.load("fertilizer_model.pkl")
scaler = joblib.load("scaler.pkl")
target_encoder = joblib.load("target_encoder.pkl")
input_encoders = joblib.load("input_encoders.pkl")

# ==========================================
# GET EXPECTED FEATURES
# ==========================================

features = scaler.feature_names_in_

print("\n")
print("=" * 60)
print("ENTER INPUT VALUES")
print("=" * 60)

user_input = {}

# ==========================================
# COLLECT INPUTS
# ==========================================

for feature in features:

    # Categorical feature
    if feature in input_encoders:

        print(f"\nAvailable values for {feature}:")
        print(list(input_encoders[feature].classes_))

        value = input(f"Enter {feature}: ").strip()

        encoded_value = input_encoders[feature].transform(
            [value]
        )[0]

        user_input[feature] = encoded_value

    # Numerical feature
    else:
        value = float(
            input(f"Enter {feature}: ")
        )

        user_input[feature] = value

# ==========================================
# CREATE DATAFRAME
# ==========================================

input_data = pd.DataFrame(
    [user_input]
)

# ==========================================
# APPLY SCALING
# ==========================================

input_scaled = scaler.transform(
    input_data
)

# ==========================================
# PREDICT
# ==========================================

prediction = model.predict(
    input_scaled
)

fertilizer = target_encoder.inverse_transform(
    prediction
)[0]

# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n")
print("=" * 60)
print("FERTILIZER RECOMMENDATION RESULT")
print("=" * 60)

print(
    "Recommended Fertilizer :",
    fertilizer
)

print("=" * 60)
