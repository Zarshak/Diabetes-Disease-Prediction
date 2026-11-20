import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Disease Prediction")
st.write("Enter the health measurements below to see the output of the trained machine-learning model.")
ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "diabetes_model.pkl"
FEATURES_PATH = ROOT / "models" / "features.json"

if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    st.error("The model has not been trained yet. First run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)
features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))

st.info("Academic demonstration only. This tool is not a medical test and must not be used to diagnose or make treatment decisions.")

with st.form("prediction_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose level", min_value=1.0, max_value=300.0, value=120.0)
    blood_pressure = st.number_input("Diastolic blood pressure", min_value=1.0, max_value=200.0, value=70.0)
    skin_thickness = st.number_input("Skin thickness", min_value=0.0, max_value=100.0, value=20.0)
    insulin = st.number_input("Insulin level", min_value=0.0, max_value=900.0, value=80.0)
    bmi = st.number_input("BMI", min_value=1.0, max_value=90.0, value=28.0)
    pedigree = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                         insulin, bmi, pedigree, age]], columns=features)
    prediction = int(model.predict(row)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(row)[0][1])
    if prediction == 1:
        st.warning("Model output: Higher likelihood of diabetes (class 1).")
    else:
        st.success("Model output: Lower likelihood of diabetes (class 0).")
    if probability is not None:
        st.write(f"Model-estimated probability for class 1: {probability:.1%}")
    st.caption("This is a machine-learning estimate based on a limited training dataset, not a clinical diagnosis.")
