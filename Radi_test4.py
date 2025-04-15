import streamlit as st
import joblib
import numpy as np

# --- Load model ---
@st.cache_resource
def load_model():
    return joblib.load("radi_recover_model.pkl")

model = load_model()

# --- Streamlit Config ---
st.set_page_config(page_title="RadiRecover", page_icon="🧬")

# --- Header ---
st.title("🧬 RadiRecover: Personalized Radiotherapy Recovery Assistant")
st.markdown("Your AI-powered side effect tracker for post-radiotherapy care.")

# --- Section 1: Patient Profile ---
st.header("👤 Patient Profile")

name = st.text_input("Name")
age = st.slider("Age", 18, 90)
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
treatment_site = st.selectbox("Radiation Treatment Site", ["Breast", "Head & Neck", "Pelvis", "Lung", "Prostate"])
comorbidities = st.multiselect("Comorbidities", ["Diabetes", "Hypertension", "Asthma", "None"])

# --- Preprocessing input ---
def preprocess_inputs():
    gender_map = {"Female": 0, "Male": 1, "Other": 2}
    site_map = {"Breast": 0, "Head & Neck": 1, "Pelvis": 2, "Lung": 3, "Prostate": 4}

    gender_val = gender_map.get(gender, 2)
    site_val = site_map.get(treatment_site, 0)

    # One-hot or binary comorbidities
    diabetes = 1 if "Diabetes" in comorbidities else 0
    hypertension = 1 if "Hypertension" in comorbidities else 0
    asthma = 1 if "Asthma" in comorbidities else 0

    return np.array([[age, gender_val, site_val, diabetes, hypertension, asthma]])

# --- Prediction Section ---
if st.button("🧠 Predict Side Effect Risk"):
    st.subheader("🔮 Predicted Risk for Fatigue")

    features = preprocess_inputs()
    prediction = model.predict_proba(features)[0][1]  # Prob. of class 1 (fatigue)

    percentage = int(prediction * 100)

    if percentage >= 75:
        st.error(f"⚠️ High Risk of Fatigue ({percentage}%)")
    elif percentage >= 40:
        st.warning(f"⚠️ Medium Risk of Fatigue ({percentage}%)")
    else:
        st.success(f"✅ Low Risk of Fatigue ({percentage}%)")

# --- Daily Symptom Tracker ---
st.header("📅 Daily Symptom Tracker")

symptom_today = st.radio("How are you feeling today?", ["Great", "Tired", "Very Tired", "In Pain", "Nauseous"])
skin_status = st.radio("Any skin issues?", ["None", "Redness", "Peeling", "Blistering"])
mood = st.slider("Mood Level (0 = low, 10 = high)", 0, 10, 5)

if st.button("📤 Submit Today's Check-In"):
    st.success("✔️ Your check-in has been submitted. Thank you!")
    if symptom_today in ["Very Tired", "In Pain"]:
        st.warning("⚠️ You're reporting serious symptoms. Please consider contacting your care team.")

# --- Self-Care Tips ---
st.header("💡 Self-Care Tip of the Day")

if treatment_site == "Breast":
    st.info("🛁 Keep your skin moisturized, avoid tight clothing, and stay hydrated.")
elif treatment_site == "Head & Neck":
    st.info("🍵 Use a soft toothbrush, stay hydrated, and rinse mouth with baking soda water.")
else:
    st.info("🧘 Gentle movement, rest when needed, and track your hydration daily.")
