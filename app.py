import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack

# ✅ Load trained models and preprocessing tools
hp_model = joblib.load("hp_model.pkl")
torque_model = joblib.load("torque_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")  # ✅ Updated TF-IDF vectorizer
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

# ✅ Streamlit App Title
st.title("🚗 Car Performance Predictor")
st.write("Enter your car's modifications and engine parameters to predict Horsepower (HP) and Torque.")

# ✅ User Input
rpm = st.number_input("Enter RPM:", min_value=500, max_value=9000, value=3000, step=100)
boost = st.number_input("Enter Boost Pressure (PSI):", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
torque = st.number_input("Enter Torque (Nm):", min_value=50.0, max_value=1000.0, value=300.0, step=10.0)
afr = st.number_input("Enter Air-Fuel Ratio (AFR):", value=14.7, step=0.1)  # ✅ Removed min_value constraint
modifications = st.text_area("Enter Modifications (e.g., 'Stage 2 Tune, Turbo Upgrade')")

# ✅ Button to make prediction
if st.button("Predict Performance"):
    # ✅ Transform input features
    input_data = pd.DataFrame({
        "Torque": [torque], "RPM": [rpm], "AFR": [afr], "Boost": [boost], "Specs": [modifications]
    })

    # ✅ Apply feature transformations
    input_data[["Torque", "RPM", "AFR", "Boost"]] = scaler.transform(input_data[["Torque", "RPM", "AFR", "Boost"]])
    specs_tfidf = vectorizer.transform(input_data["Specs"])  # ✅ Ensure 202 features

    # ✅ Combine numeric + text features
    input_features = hstack((input_data[["Torque", "RPM", "AFR", "Boost"]].values, specs_tfidf))
    input_features = imputer.transform(input_features)  # Apply imputation

    # ✅ Make predictions
    predicted_hp = hp_model.predict(input_features)[0]
    predicted_torque = torque_model.predict(input_features)[0]

    # ✅ Display Results
    st.subheader("Predicted Performance")
    st.success(f"🔹 Predicted Horsepower: **{predicted_hp:.2f} HP**")
    st.success(f"🔹 Predicted Torque: **{predicted_torque:.2f} Nm**")
