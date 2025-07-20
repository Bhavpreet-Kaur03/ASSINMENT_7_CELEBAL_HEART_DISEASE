#!/usr/bin/env python
# coding: utf-8

# In[1]:


# app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("heart_disease_pipeline.pkl")

# Set page config
st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")

# Title and subtitle
st.markdown("<h1 style='text-align: center; color: #c0392b;'>💓 Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #7f8c8d;'>Enter patient health details below to assess heart disease risk.</h4>", unsafe_allow_html=True)

# Sidebar form for user input
with st.form("prediction_form"):
    st.subheader("🧍‍♂️ Patient Health Profile")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 100, 50)
        gender = st.selectbox("Gender", ["Male", "Female"])
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 400, 200)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 80, 200, 120)
        heart_rate = st.slider("Heart Rate (bpm)", 60, 200, 100)
        smoking = st.selectbox("Smoking", ["Yes", "No"])
        alcohol = st.selectbox("Alcohol Intake", ["None", "Low", "Moderate", "High"])

    with col2:
        exercise_hours = st.slider("Exercise Hours per Week", 0, 20, 3)
        family_history = st.selectbox("Family History of Heart Disease", ["Yes", "No"])
        diabetes = st.selectbox("Diabetes", ["Yes", "No"])
        obesity = st.selectbox("Obesity", ["Yes", "No"])
        stress = st.slider("Stress Level (1 = Low, 10 = High)", 1, 10, 5)
        blood_sugar = st.slider("Blood Sugar (mg/dL)", 70, 250, 100)
        angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        chest_pain = st.selectbox("Chest Pain Type", ["Typical", "Atypical", "Non-anginal", "Asymptomatic"])

    submitted = st.form_submit_button("🔍 Predict Risk")

# Predict when form is submitted
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
        "Chest Pain Type": [chest_pain]
    })

    # Make prediction
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    # Display result
    st.markdown("---")
    st.subheader("🩺 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({probabilities[1]*100:.2f}% confidence)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({probabilities[0]*100:.2f}% confidence)")

    # Visualize confidence
    st.subheader("📊 Prediction Confidence")

    fig, ax = plt.subplots()
    labels = ["No Heart Disease", "Heart Disease"]
    colors = ['#27ae60', '#c0392b']
    ax.barh(labels, probabilities, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence Score")
    for i, v in enumerate(probabilities):
        ax.text(v + 0.02, i, f"{v*100:.2f}%", va='center')
    st.pyplot(fig)

    st.markdown("---")

# Footer
st.markdown("""
<hr style="border: 1px solid #eee;" />
<div style="text-align: center; color: gray;">
    Developed by <strong>Bhavpreet Kaur</strong> 💻
</div>
""", unsafe_allow_html=True)


# In[ ]:




