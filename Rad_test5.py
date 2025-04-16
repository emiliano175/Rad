import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
@st.cache_resource
def load_model():
    model = joblib.load("fatigue_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    return model, encoders

model, encoders = load_model()

# Set config
st.set_page_config(page_title="RadiRecover", page_icon="🧬")

# --- UI ---
st.title("🧬 RadiRecover: Personalized Radiotherapy Recovery Assistant")
st.markdown("Predict fatigue risk after radiotherapy based on patient profile.")

# --- Input ---
st.header("👤 Patient Profile")
name = st.text_input("Name")
age = st.slider("Age", 30, 85, 50)
gender = st.selectbox("Gender", ["Female", "Male"])
treatment_site = st.selectbox("Treatment Site", ["Breast", "Lung"])
tumor_size = st.slider("Tumor Size (cm)", 1.0, 7.0, 3.5)
radiation_dose = st.selectbox("Radiation Dose (Gy)", [40, 50, 60, 70])
comorbidities = st.selectbox("Comorbidities", ["None", "Diabetes", "Hypertension", "Asthma"])
chemotherapy = st.radio("Undergoing Chemotherapy?", ["Yes", "No"])

if st.button("🧠 Predict Fatigue Risk"):
    # Prepare input for model
    input_dict = {
        "age": age,
        "gender": encoders["gender"].transform([gender])[0],
        "treatment_site": encoders["treatment_site"].transform([treatment_site])[0],
        "tumor_size_cm": tumor_size,
        "radiation_dose_Gy": radiation_dose,
        "comorbidities": encoders["comorbidities"].transform([comorbidities])[0],
        "chemotherapy": encoders["chemotherapy"].transform([chemotherapy])[0],
    }
    input_df = pd.DataFrame([input_dict])

    # Make prediction
    pred_encoded = model.predict(input_df)[0]
    fatigue_risk = encoders["fatigue_risk"].inverse_transform([pred_encoded])[0]

    st.subheader("🔮 Predicted Fatigue Risk")
    if fatigue_risk == "High":
        st.error(f"⚠️ {name} is at **High Risk** of fatigue.")
    elif fatigue_risk == "Medium":
        st.warning(f"⚠️ {name} is at **Medium Risk** of fatigue.")
    else:
        st.success(f"✅ {name} is at **Low Risk** of fatigue.")

# Optional: show the model confidence/proba
    proba = model.predict_proba(input_df)[0]
    st.markdown("📊 Probability Breakdown:")
    for i, label in enumerate(encoders["fatigue_risk"].classes_):
        st.markdown(f"- **{label}**: {round(proba[i]*100)}%")
