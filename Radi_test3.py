import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

st.set_page_config(page_title="RadiRecover", page_icon="🧬")

# --- Train Model on First Run ---
@st.cache_resource
def get_model():
    np.random.seed(42)
    n = 500
    age = np.random.randint(25, 85, n)
    gender = np.random.choice([0, 1], n)  # 0 = Female, 1 = Male
    site = np.random.choice(['Breast', 'Head & Neck', 'Pelvis', 'Lung', 'Prostate'], n)
    diabetes = np.random.choice([0, 1], n, p=[0.8, 0.2])
    hypertension = np.random.choice([0, 1], n, p=[0.7, 0.3])
    asthma = np.random.choice([0, 1], n, p=[0.9, 0.1])

    site_encoded = pd.get_dummies(site, prefix='site')
    X = pd.DataFrame({'age': age, 'gender': gender, 'diabetes': diabetes,
                      'hypertension': hypertension, 'asthma': asthma})
    X = pd.concat([X, site_encoded], axis=1)

    def simulate_y(row):
        return pd.Series([
            int(row['age'] > 60 or row.get('site_Pelvis', 0)),
            int(row.get('site_Breast', 0) or row.get('site_Head & Neck', 0)),
            int(row.get('site_Pelvis', 0) or row['asthma'])
        ])

    y = X.apply(simulate_y, axis=1)
    y.columns = ['Fatigue', 'Skin Irritation', 'Nausea']

    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100))
    model.fit(X, y)

    return model

model = get_model()

# --- Header ---
st.title("🧬 RadiRecover: Personalized Radiotherapy Recovery Assistant")
st.markdown("Your AI-powered side effect tracker for post-radiotherapy care.")

# --- Section 1: Patient Onboarding ---
st.header("👤 Patient Profile")

name = st.text_input("Name")
age = st.slider("Age", 18, 90)
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
treatment_site = st.selectbox("Radiation Treatment Site", ["Breast", "Head & Neck", "Pelvis", "Lung", "Prostate"])
comorbidities = st.multiselect("Comorbidities", ["Diabetes", "Hypertension", "Asthma", "None"])

# --- Section 2: Side Effect Prediction ---
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

    for i, label in enumerate(labels):
        prob = preds[i][0][1] * 100
        if prob > 75:
            st.error(f"🔴 {label}: {prob:.1f}%")
        elif prob > 40:
            st.warning(f"🟠 {label}: {prob:.1f}%")
        else:
            st.success(f"🟢 {label}: {prob:.1f}%")

# --- Section 3: Daily Check-In ---
st.header("📅 Daily Symptom Tracker")

symptom_today = st.radio("How are you feeling today?", ["Great", "Tired", "Very Tired", "In Pain", "Nauseous"])
skin_status = st.radio("Any skin issues?", ["None", "Redness", "Peeling", "Blistering"])
mood = st.slider("Mood Level (0 = low, 10 = high)", 0, 10, 5)

if st.button("📤 Submit Today's Check-In"):
    st.success("✔️ Your check-in has been submitted. Thank you!")
    if symptom_today in ["Very Tired", "In Pain"]:
        st.warning("⚠️ Alert: You're reporting serious symptoms. Please consider contacting your care team.")

# --- Section 4: Self-Care Tips ---
st.header("💡 Self-Care Tip of the Day")

if treatment_site == "Breast":
    st.info("🛁 Keep your skin moisturized, avoid tight clothing, and stay hydrated.")
elif treatment_site == "Head & Neck":
    st.info("🍵 Use a soft toothbrush, stay hydrated, and rinse mouth with baking soda water.")
elif treatment_site == "Pelvis":
    st.info("🧘 Gentle movement, rest when needed, and track your hydration daily.")
else:
    st.info("🛌 Prioritize rest and maintain a light, nutritious diet.")
