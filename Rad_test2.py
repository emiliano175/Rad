import streamlit as st

st.set_page_config(page_title="RadiRecover", page_icon="🧬")

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

# --- Section 2: Side Effect Prediction (Basic Logic) ---
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
else:
    st.info("🧘 Gentle movement, rest when needed, and track your hydration daily.")

