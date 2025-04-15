import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load("radi_recover_model.pkl")  # Update path if needed

st.set_page_config(page_title="RadiRecover", page_icon="🧬")
st.title("🧬 RadiRecover: Personalized Radiotherapy Recovery Assistant")
st.markdown("Now powered with real-time side effect prediction from a trained model.")

# --- Patient Profile ---
st.header("👤 Patient Profile")

name = st.text_input("Name")
age = st.slider("Age", 18, 90, 50)
gender = st.selectbox("Gender", ["Female", "Male"])
treatment_site = st.selectbox("Radiation Treatment Site", ["Breast", "Head & Neck", "Pelvis", "Lung", "Prostate"])
comorbidities = st.multiselect("Comorbidities", ["Diabetes", "Hypertension", "Asthma"])

# --- Prediction Logic ---
if st.button("🧠 Predict Side Effects"):

    # Encode inputs
    gender_num = 0 if gender == "Female" else 1
    diabetes = 1 if "Diabetes" in comorbidities else 0
    hypertension = 1 if "Hypertension" in comorbidities else 0
    asthma = 1 if "Asthma" in comorbidities else 0

    # One-hot encode treatment site
    site_list = ["Breast", "Head & Neck", "Lung", "Pelvis", "Prostate"]
    site_encoded = [1 if treatment_site == site else 0 for site in site_list]

    # Construct input array
    input_data = np.array([[age, gender_num, diabetes, hypertension, asthma] + site_encoded])
    input_df = pd.DataFrame(input_data, columns=[
        "age", "gender", "diabetes", "hypertension", "asthma",
        "site_Breast", "site_Head & Neck", "site_Lung", "site_Pelvis", "site_Prostate"
    ])

    # Predict
    prediction = model.predict_proba(input_df)
    labels = ["Fatigue", "Skin Irritation", "Nausea"]
    
    st.subheader("🔮 Predicted Side Effect Probabilities:")
    for i, label in enumerate(labels):
        prob = prediction[i][0][1] * 100  # Probability of class 1
        if prob > 75:
            st.error(f"🔴 {label}: {prob:.1f}%")
        elif prob > 40:
            st.warning(f"🟠 {label}: {prob:.1f}%")
        else:
            st.success(f"🟢 {label}: {prob:.1f}%")
