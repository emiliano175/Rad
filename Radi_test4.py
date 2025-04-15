import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from fpdf import FPDF
from datetime import datetime

# --- Set Page Configuration ---
st.set_page_config(page_title="RadiRecover", page_icon="🧬")

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

# --- PDF Generation ---
def generate_pdf(name, age, gender, treatment_site, comorbidities, fatigue_risk, symptoms, skin, mood):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="RadiRecover Report", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age}    Gender: {gender}", ln=True)
    pdf.cell(200, 10, txt=f"Treatment Site: {treatment_site}", ln=True)
    pdf.cell(200, 10, txt=f"Comorbidities: {', '.join(comorbidities) if comorbidities else 'None'}", ln=True)
    pdf.cell(200, 10, txt=f"Predicted Fatigue Risk: {fatigue_risk}%", ln=True)
    pdf.cell(200, 10, txt=f"Daily Check-In: {symptoms}, Skin: {skin}, Mood: {mood}/10", ln=True)
    pdf.cell(200, 10, txt=f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)

    return pdf.output(dest='S').encode('latin1')

# --- UI Components ---
st.title("🧬 RadiRecover: Personalized Radiotherapy Recovery Assistant")
st.markdown("Your AI-powered side effect tracker for post-radiotherapy care.")

# Patient Profile Section
st.header("👤 Patient Profile")
name = st.text_input("Name")
age = st.slider("Age", 18, 90)
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
treatment_site = st.selectbox("Radiation Treatment Site", ["Breast", "Head & Neck", "Pelvis", "Lung", "Prostate"])
comorbidities = st.multiselect("Comorbidities", ["Diabetes", "Hypertension", "Asthma", "None"])

# Simulate fatigue prediction based on the selected inputs
def predict_risk():
    # Dummy logic for fatigue risk prediction
    risk = {
        "Breast": 80,
        "Head & Neck": 70,
        "Pelvis": 85,
        "Lung": 60,
        "Prostate": 50
    }
    return risk.get(treatment_site, 50)

if st.button("🧠 Generate Predicted Side Effects"):
    st.subheader("🔮 Predicted Side Effects")

    if treatment_site == "Breast":
        st.markdown("- **Fatigue** – High Risk (85%)")
        st.markdown("- **Skin Irritation** – Medium Risk (60%)")
        st.markdown("- **Nausea** – Low Risk (25%)")
    elif treatment_site == "Head & Neck":
        st.markdown("- **Mouth Sores** – High Risk (75%)")
        st.markdown("- **Dry Mouth** – Medium Risk (50%)")
    elif treatment_site == "Pelvis":
        st.markdown("- **Diarrhea** – Medium Risk (60%)")
        st.markdown("- **Fatigue** – High Risk (80%)")
    else:
        st.markdown("No prediction rules for this site yet.")

# Daily Symptom Tracker Section
st.header("📅 Daily Symptom Tracker")
symptom_today = st.radio("How are you feeling today?", ["Great", "Tired", "Very Tired", "In Pain", "Nauseous"])
skin_status = st.radio("Any skin issues?", ["None", "Redness", "Peeling", "Blistering"])
mood = st.slider("Mood Level (0 = low, 10 = high)", 0, 10, 5)

if st.button("📤 Submit Today's Check-In"):
    st.success("✔️ Your check-in has been submitted. Thank you!")
    if symptom_today in ["Very Tired", "In Pain"]:
        st.warning("⚠️ Alert: You're reporting serious symptoms. Please consider contacting your care team.")

# Generate PDF Report Section
if st.button("📤 Generate PDF Report"):
    fatigue_risk = predict_risk()
    pdf_bytes = generate_pdf(
        name=name,
        age=age,
        gender=gender,
        treatment_site=treatment_site,
        comorbidities=comorbidities,
        fatigue_risk=fatigue_risk,
        symptoms=symptom_today,
        skin=skin_status,
        mood=mood,
    )

    st.download_button(
        label="📄 Download Report",
        data=pdf_bytes,
        file_name=f"{name}_radi_recovery_report.pdf",
        mime="application/pdf"
    )
