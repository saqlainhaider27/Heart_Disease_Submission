"""
Heart Disease Prediction System - Streamlit App
Name: Saqlain Haider | Roll No: SP24-BSE-109 | Section: FA24-BSE-A
Course: AIC354 Machine Learning Fundamentals - Assignment 2

Loads heart_svc_trained_model.pkl (produced in Step 6.4 of the notebook) and
re-creates the same LabelEncoders (fit on the same fixed category lists).
"""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "heart_svc_trained_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    return {
        "Sex": LabelEncoder().fit(["Male", "Female"]),
        "ChestPain": LabelEncoder().fit(["Typical", "Atypical", "NonAnginal", "Asymptomatic"]),
        "ExerciseAngina": LabelEncoder().fit(["Yes", "No"]),
        "MajorVessels": LabelEncoder().fit(["Zero", "One", "Two", "Three"]),
    }


model = load_model()
encoders = build_label_encoders()

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    Enter the patient's details and the trained **Support Vector Classifier (SVC)**
    will predict whether heart disease is present.
    Built from the UCI Heart Disease (Cleveland) dataset using **4 input attributes**.

    *Built by Saqlain Haider (SP24-BSE-109, FA24-BSE-A) for AIC354 Assignment 2.*
    """
)
st.divider()

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", ["Male", "Female"])
    chestpain = st.selectbox("Chest Pain Type", ["Typical", "Atypical", "NonAnginal", "Asymptomatic"], index=3)
with col2:
    angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    vessels = st.selectbox("Major Vessels Coloured (0-3)", ["Zero", "One", "Two", "Three"])

if st.button("Predict", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "Sex": [sex], "ChestPain": [chestpain],
        "ExerciseAngina": [angina], "MajorVessels": [vessels],
    })
    encoded = pd.DataFrame({c: encoders[c].transform(user_input[c]) for c in user_input.columns})
    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE DETECTED")
    else:
        st.success("### ✅ NO HEART DISEASE")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption("For learning purposes only. Based on 4 attributes of the UCI Cleveland data; not a medical diagnosis.")
