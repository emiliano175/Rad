import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="RadiRecover", page_icon="🧬")
st.title("🧬 RadiRecover: Personalized Radiotherapy Recovery Assistant")

# --- 1. Generate synthetic data ---
@st.cache_resource
def train_model():
    np.random.seed(42)
    n_patients = 500

    ages = np.random.randint(25, 85, n_patients)
    genders = np.random.choice([0, 1], size=n_patients)  # 0 = Female, 1 = Male
    treatment_sites = np.random.choice(['Breast', 'Head & Neck', 'Pelvis', 'Lung', 'Prostate'], n_patients)
    diabetes = np.random.choice([0, 1], size=n_patients, p=[0.8, 0.2])
    hypertension = np.random.choice([0, 1], size=n_patients, p=[0.7, 0.3])
    asthma = np.random.choice([0, 1], size=n_patients, p=[0.9, 0.1])

    site_encoded = pd.get_dummies(treatment_sites, prefix='site')

    X = pd.DataFrame({
        'age': ages,
        'gender': genders,
        'diabetes': diabetes,
        'hypertension': hypertension,
        'asthma': asthma
    })
    X = pd.concat([X, site_encoded], axis=1)

    def simulate_outcomes(row):
        fatigue = int(row['age'] > 60 or row.get('site_Pelvis', 0) == 1 or row['hypertension'] == 1)
        skin_irritation = int(row.get('site_Breast', 0) == 1 or row.get('site_Head & Neck', 0) == 1)
        nausea = int(row.get('site_Pelvis', 0) == 1 or row['asthma'] == 1)
        return pd.Series([fatigue, skin_irritation, nausea])

    y = X.apply(simulate_outcomes, axis=1)
    y.columns = ['fatigue', 'skin_irritation', 'nausea']

    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X, y)

    return model

model = train_model()

# --- 2. User input form ---
st.header("👤 Enter Patient Information")

name = st.text_input("Name")
age = st.slider("Age", 18, 90, 50)
gender = st.selectbox("Gender", ["Female", "Male"])
treatment_site = st.selectbox("Radiation Treatment Site", ["Breast", "Head & Neck", "Pelvis", "Lung", "Prostate"])
comorbidities = st.multiselect("Comorbidities", ["Diabetes", "Hypertension", "Asthma"])

# --- 3. Predict button ---
if st.button("🧠 Predict Side Effects"):

    gender_num = 0 if gender == "Female" else 1
    diabetes = 1 if "Diabetes" in comorbidities else 0
    hypertension = 1 if "Hypertension" in comorbidities else 0
    asthma = 1 if "Asthma" in comorbidities else 0

    site_list = ["Breast", "Head & Neck", "Lung", "Pelvis", "Prostate"]
    site_encoded = [1 if treatment_site == site else 0 for site in site_list]

    input_data = np.array([[age, gender_num, diabetes, hypertension, asthma] + site_encoded])
    input_df = pd.DataFrame(input_data, columns=[
        "age", "gender", "diabetes", "hypertension", "asthma",
        "site_Breast", "site_Head & Neck", "site_Lung", "site_Pelvis", "site_Prostate"
    ])

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
