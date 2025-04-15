import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Simulate Patient Data ---
def simulate_data(n=500):
    np.random.seed(42)
    data = pd.DataFrame({
        "age": np.random.randint(18, 90, n),
        "gender": np.random.choice(["Female", "Male", "Other"], n),
        "site": np.random.choice(["Breast", "Head & Neck", "Pelvis", "Lung", "Prostate"], n),
        "diabetes": np.random.choice([0, 1], n),
        "hypertension": np.random.choice([0, 1], n),
        "asthma": np.random.choice([0, 1], n),
    })

    # Simulate fatigue risk
    data["fatigue"] = (
        (data["age"] > 65).astype(int) +
        data["diabetes"] +
        (data["site"].isin(["Pelvis", "Breast"]).astype(int)) +
        (data["gender"] == "Female").astype(int)
    ) >= 2
    data["fatigue"] = data["fatigue"].astype(int)
    return data

# --- Train Model ---
@st.cache_resource
def train_model():
    df = simulate_data()
    df["gender"] = df["gender"].map({"Female": 0, "Male": 1, "Other": 2})
    df["site"] = df["site"].map({
        "Breast": 0, "Head & Neck": 1, "Pelvis": 2, "Lung": 3, "Prostate": 4
    })

    X = df.drop("fatigue", axis=1)
    y = df["fatigue"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# --- App UI ---
st.set_page_config(page_title="RadiRecover", page_icon="🧬")
st.title("🧬 RadiRecover: Personalized Radiotherapy Recovery Assistant")
st.markdown("Your AI-powered side effect tracker for post-radiotherapy care.")

st.header("👤 Patient Profile")

name = st.text_input("Name")
age = st.slider("Age", 18, 90)
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
treatment_site = st.selectbox("Radiation Treatment Site", ["Breast", "Head & Neck", "Pelvis", "Lung", "Prostate"])
comorbidities = st.multiselect("Comorbidities", ["Diabetes", "Hypertension", "Asthma", "None"])

# --- Predict Fatigue ---
def predict_risk():
    gender_map = {"Female": 0, "Male": 1, "Other": 2}
    site_map = {"Breast": 0, "Head & Neck": 1, "Pelvis": 2, "Lung": 3, "Prostate": 4}

    gender_val = gender_map[gender]
    site_val = site_map[treatment_site]

    diabetes = 1 if "Diabetes" in comorbidities else 0
    hypertension = 1 if "Hypertension" in comorbidities else 0
    asthma = 1 if "Asthma" in comorbidities else 0

    X = np.array([[age, gender_val, site_val, diabetes, hypertension, asthma]])
    prob = model.predict_proba(X)[0][1]
    return int(prob * 100)

if st.button("🧠 Predict Side Effect Risk"):
    risk = predict_risk()
    st.subheader("🔮 Predicted Risk for Fatigue")
    if risk >= 75:
        st.error(f"⚠️ High Risk of Fatigue ({risk}%)")
    elif risk >= 40:
        st.warning(f"⚠️ Medium Risk of Fatigue ({risk}%)")
    else:
        st.success(f"✅ Low Risk of Fatigue ({risk}%)")

# --- Daily Check-In ---
st.header("📅 Daily Symptom Tracker")
symptom_today = st.radio("How are you feeling today?", ["Great", "Tired", "Very Tired", "In Pain", "Nauseous"])
skin_status = st.radio("Any skin issues?", ["None", "Redness", "Peeling", "Blistering"])
mood = st.slider("Mood Level (0 = low, 10 = high)", 0, 10, 5)

if st.button("📤 Submit Today's Check-In"):
    st.success("✔️ Your check-in has been submitted.")
    if symptom_today in ["Very Tired", "In Pain"]:
        st.warning("⚠️ Please consider contacting your care team.")

# --- Self-Care Tip ---
st.header("💡 Self-Care Tip of the Day")
if treatment_site == "Breast":
    st.info("🛁 Keep your skin moisturized, avoid tight clothing, and stay hydrated.")
elif treatment_site == "Head & Neck":
    st.info("🍵 Use a soft toothbrush, stay hydrated, and rinse mouth with baking soda water.")
else:
    st.info("🧘 Gentle movement, rest when needed, and track your hydration daily.")
