import streamlit as st
import numpy as np
import pandas as pd
import os
from utils import train_model
from patient_report import generate_pdf

st.set_page_config(page_title="RadiRecover", page_icon="🧬")
model = train_model()

# --- Header ---
st.title("🧬 RadiRecover: Personalized Radiotherapy Recovery Assistant")
st.markdown("Your AI-powered side effect tracker for post-radiotherapy care.")

# --- Patient Profile ---
st.header("👤 Patient Profile")
name = st.text_input("Name")
age = st.slider("Age", 18, 90)
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
treatment_site = st.selectbox("Radiation Treatment Site", ["Breast", "Head & Neck", "Pelvis", "Lung", "Prostate"])
comorbidities = st.multiselect("Comorbidities", ["Diabetes", "Hypertension", "Asthma", "None"])

# --- Prediction ---
if st.button("🧠 Generate Predicted Side Effects"):
    st.subheader("🔮 Predicted Side Effects")
    gender_val = 0 if gender == "Female" else 1
    diabetes = int("Diabetes" in comorbidities)
    hypertension = int("Hypertension" in comorbidities)
    asthma = int("Asthma" in comorbidities)

    site_options = ["Breast", "Head & Neck", "Lung", "Pelvis", "Prostate"]
    site_encoded = [1 if treatment_site == site else 0 for site in site_options]

    input_data = np.array([[age, gender_val, diabetes, hypertension, asthma] + site_encoded])
    input_df = pd.DataFrame(input_data, columns=[
        "age", "gender", "diabetes", "hypertension", "asthma",
        "site_Breast", "site_Head & Neck", "site_Lung", "site_Pelvis", "site_Prostate"
    ])

    preds = model.predict_proba(input_df)
    labels = ["Fatigue", "Skin Irritation", "Nausea"]
    pred_probs = {label: preds[i][0][1] * 100 for i, label in enumerate(labels)}

    for label, prob in pred_probs.items():
        if prob > 75:
            st.error(f"🔴 {label}: {prob:.1f}%")
        elif prob > 40:
            st.warning(f"🟠 {label}: {prob:.1f}%")
        else:
            st.success(f"🟢 {label}: {prob:.1f}%")

    if st.button("📥 Download Report as PDF"):
        path = generate_pdf(name, age, gender, treatment_site, comorbidities, pred_probs)
        with open(path, "rb") as f:
            st.download_button("⬇️ Download Report", f, file_name="RadiRecover_Report.pdf")

# --- Daily Check-In ---
st.header("📅 Daily Symptom Tracker")
symptom_today = st.radio("How are you feeling today?", ["Great", "Tired", "Very Tired", "In Pain", "Nauseous"])
skin_status = st.radio("Any skin issues?", ["None", "Redness", "Peeling", "Blistering"])
mood = st.slider("Mood Level (0 = low, 10 = high)", 0, 10, 5)

def log_checkin(name, symptom, skin, mood):
    log_exists = os.path.exists("symptom_logs.csv")
    with open("symptom_logs.csv", "a") as f:
        if not log_exists:
            f.write("Name,Symptom,Skin,Mood\n")
        f.write(f"{name},{symptom},{skin},{mood}\n")

if st.button("📤 Submit Today's Check-In"):
    log_checkin(name, symptom_today, skin_status, mood)
    st.success("✔️ Your check-in has been submitted.")
    if symptom_today in ["Very Tired", "In Pain"]:
        st.warning("⚠️ Consider contacting your care team.")

# --- View History ---
if st.button("📊 View My Past Check-Ins"):
    if os.path.exists("symptom_logs.csv"):
        logs = pd.read_csv("symptom_logs.csv")
        logs = logs[logs["Name"] == name]
        st.dataframe(logs)
    else:
        st.info("No past check-ins yet.")

# --- Self-Care Tips ---
st.header("💡 Self-Care Tip of the Day")
if treatment_site == "Breast":
    st.info("🛁 Keep your skin moisturized, avoid tight clothing, and stay hydrated.")
elif treatment_site == "Head & Neck":
    st.info("🍵 Use a soft toothbrush, stay hydrated, and rinse mouth with baking soda water.")
else:
    st.info("🧘 Gentle movement, rest when needed, and track your hydration daily.")
