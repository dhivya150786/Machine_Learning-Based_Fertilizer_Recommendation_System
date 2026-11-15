import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Fertilizer Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# ==========================================
# LOAD MODEL FILES
# ==========================================

model = joblib.load("fertilizer_model.pkl")
scaler = joblib.load("scaler.pkl")
target_encoder = joblib.load("target_encoder.pkl")
input_encoders = joblib.load("input_encoders.pkl")

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    margin-bottom: 30px;
}

.result-box {
    border: 2px solid #4CAF50;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    background-color: #f5fff5;
    width: 100%;
}

div.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: bold;
    background-color: #4CAF50;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.markdown(
    '<div class="title">🌱 Fertilizer Recommendation System</div>',
    unsafe_allow_html=True
)

# ==========================================
# FEATURES
# ==========================================

features = list(scaler.feature_names_in_)

left_features = features[:9]
right_features = features[9:]

user_input = {}

# ==========================================
# INPUT SECTION
# ==========================================

left_column, right_column = st.columns(2)

# ==========================================
# LEFT SIDE INPUTS
# ==========================================

with left_column:

    for feature in left_features:

        label_col, input_col = st.columns([1, 2])

        with label_col:
            st.markdown(f"**{feature}**")

        with input_col:

            if feature in input_encoders:

                options = ["Select"] + list(
                    input_encoders[feature].classes_
                )

                value = st.selectbox(
                    "",
                    options,
                    key=feature,
                    label_visibility="collapsed"
                )

                user_input[feature] = value

            else:

                value = st.text_input(
                    "",
                    key=feature,
                    placeholder="Enter value",
                    label_visibility="collapsed"
                )

                user_input[feature] = value

# ==========================================
# RIGHT SIDE INPUTS
# ==========================================

with right_column:

    for feature in right_features:

        label_col, input_col = st.columns([1, 2])

        with label_col:
            st.markdown(f"**{feature}**")

        with input_col:

            if feature in input_encoders:

                options = ["Select"] + list(
                    input_encoders[feature].classes_
                )

                value = st.selectbox(
                    "",
                    options,
                    key=feature,
                    label_visibility="collapsed"
                )

                user_input[feature] = value

            else:

                value = st.text_input(
                    "",
                    key=feature,
                    placeholder="Enter value",
                    label_visibility="collapsed"
                )

                user_input[feature] = value



    # ==========================================
# CENTER BUTTON
# ==========================================

st.write("")
st.write("")

left_space, center_button, right_space = st.columns([3,2,3])

with center_button:

    predict = st.button(
        "🔍 Predict Fertilizer",
        use_container_width=True
    )

# ==========================================
# OUTPUT PLACEHOLDER BELOW BUTTON
# ==========================================

output_placeholder = st.empty()

# ==========================================
# PREDICTION
# ==========================================

if predict:

    missing_fields = []

    for feature, value in user_input.items():

        if value == "" or value == "Select":
            missing_fields.append(feature)

    # --------------------------------------
    # No inputs entered
    # --------------------------------------

    if len(missing_fields) == len(features):

        output_placeholder.warning(
            "⚠ Please enter all input values before prediction."
        )

    # --------------------------------------
    # Some inputs missing
    # --------------------------------------

    elif len(missing_fields) > 0:

        output_placeholder.warning(
            "⚠ Missing Inputs:\n\n" +
            "\n".join(missing_fields)
        )

    # --------------------------------------
    # Prediction
    # --------------------------------------

    else:

        processed_input = {}

        for feature, value in user_input.items():

            if feature in input_encoders:

                processed_input[feature] = (
                    input_encoders[feature]
                    .transform([value])[0]
                )

            else:

                processed_input[feature] = float(value)

        input_df = pd.DataFrame(
            [processed_input]
        )

        input_scaled = scaler.transform(
            input_df
        )

        prediction = model.predict(
            input_scaled
        )

        fertilizer = target_encoder.inverse_transform(
            prediction
        )[0]

        output_placeholder.markdown(
            f"""
            <div style="
                width:450px;
                margin:auto;
                margin-top:20px;
                border:2px solid #4CAF50;
                border-radius:12px;
                padding:18px;
                text-align:center;
                background-color:#f5fff5;
                font-size:24px;
                font-weight:bold;
            ">
                🌱 Recommended Fertilizer
                <br><br>
                <span style="font-size:32px;">
                    {fertilizer}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
