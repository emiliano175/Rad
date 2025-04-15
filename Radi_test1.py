# streamlit run app.py
import streamlit as st

st.title("RadiRecover – Personalized Side Effect Tracker")

st.subheader("📋 Patient Profile")
name = st.text_input("Patient Name")
treatment_site = st.selectbox("Radiation Site", ["Breast", "Head & Neck", "Pelvis", "Lung"])
age = st.slider("Age", 18, 90)
comorbidities = st.multiselect("Comorbidities", ["Diabetes", "Hypertension", "None"])

st.subheader("🧠 Predicted Side Effects")
# (Use dummy rules or random for now)
if treatment_site == "Breast":
    st.markdown("- Likely fatigue (85%)")
    st.markdown("- Mild skin burn (60%)")
else:
    st.markdown("⚠️ Prediction coming soon...")

st.subheader("📅 Daily Check-In")
st.radio("How do you feel today?", ["Great", "Tired", "Very Tired", "Sick"])
st.button("Submit")

