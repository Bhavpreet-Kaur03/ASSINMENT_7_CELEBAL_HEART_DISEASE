#!/usr/bin/env python
# coding: utf-8

# In[5]:


# app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("heart_disease_final.pkl")

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.markdown("<h1 style='text-align:center; color:#e74c3c;'>❤️ Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Input patient details to check for heart disease risk.</h4>", unsafe_allow_html=True)

with st.form("predict_form"):
    st.subheader("Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 100, 50)
        gender = st.selectbox("Gender", ["Male", "Female"])
        cholesterol = st.slider("Cholesterol", 100, 400, 200)
        blood_pressure = st.slider("Blood Pressure", 80, 200, 120)
        heart_rate = st.slider("Heart Rate", 60, 200, 100)
        smoking = st.selectbox("Smoking", ["Yes", "No"])
        alcohol = st.selectbox("Alcohol Intake", ["None", "Low", "Moderate", "High"])

    with col2:
        exercise_hours = st.slider("Exercise Hours", 0, 20, 3)
        family_history = st.selectbox("Family History", ["Yes", "No"])
        diabetes = st.selectbox("Diabetes", ["Yes", "No"])
        obesity = st.selectbox("Obesity", ["Yes", "No"])
        stress = st.slider("Stress Level (1-10)", 1, 10, 5)
        blood_sugar = st.slider("Blood Sugar", 70, 250, 100)
        angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        chest_pain = st.selectbox("Chest Pain Type", ["Typical", "Atypical", "Non-anginal", "Asymptomatic"])

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Cholesterol": [cholesterol],
        "Blood Pressure": [blood_pressure],
        "Heart Rate": [heart_rate],
        "Smoking": [smoking],
        "Alcohol Intake": [alcohol],
        "Exercise Hours": [exercise_hours],
        "Family History": [family_history],
        "Diabetes": [diabetes],
        "Obesity": [obesity],
        "Stress Level": [stress],
        "Blood Sugar": [blood_sugar],
        "Exercise Induced Angina": [angina],
        "Chest Pain Type": [chest_pain],
    })

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.subheader("🩺 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({proba[1]*100:.2f}% confidence)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({proba[0]*100:.2f}% confidence)")

    st.subheader("📊 Confidence Visualization")
    fig, ax = plt.subplots()
    ax.barh(["No Disease", "Heart Disease"], proba, color=["green", "red"])
    ax.set_xlim(0, 1)
    for i, v in enumerate(proba):
        ax.text(v + 0.01, i, f"{v*100:.2f}%", va="center")
    st.pyplot(fig)

st.markdown("""
---
<div style="text-align:center; color:gray">
    Developed by <strong>Bhavpreet Kaur</strong> 💻
</div>
""", unsafe_allow_html=True)


# In[ ]:




