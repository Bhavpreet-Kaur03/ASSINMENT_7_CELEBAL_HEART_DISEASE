#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("heart_disease_final_.pkl")

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.markdown("<h1 style='text-align: center; color: #c0392b;'>💓 Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Enter your health profile to assess risk of heart disease</h4>", unsafe_allow_html=True)

# Input Form
with st.form("form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 20, 100, 50)
        gender = st.selectbox("Gender", ["Male", "Female"])
        cholesterol = st.slider("Cholesterol", 100, 400, 200)
        bp = st.slider("Blood Pressure", 80, 200, 120)
        hr = st.slider("Heart Rate", 60, 200, 100)
        exercise = st.slider("Exercise Hours/Week", 0, 20, 3)
        stress = st.slider("Stress Level (1-10)", 1, 10, 5)
        sugar = st.slider("Blood Sugar", 70, 250, 100)

    with col2:
        smoking = st.selectbox("Smoking", ["Yes", "No"])
        alcohol = st.selectbox("Alcohol Intake", ["None", "Low", "Moderate", "High"])
        family = st.selectbox("Family History", ["Yes", "No"])
        diabetes = st.selectbox("Diabetes", ["Yes", "No"])
        obesity = st.selectbox("Obesity", ["Yes", "No"])
        angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        pain = st.selectbox("Chest Pain Type", ["Typical", "Atypical", "Non-anginal", "Asymptomatic"])

    submit = st.form_submit_button("🔍 Predict")

# Make prediction
if submit:
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Cholesterol": [cholesterol],
        "Blood Pressure": [bp],
        "Heart Rate": [hr],
        "Exercise Hours": [exercise],
        "Stress Level": [stress],
        "Blood Sugar": [sugar],
        "Smoking": [smoking],
        "Alcohol Intake": [alcohol],
        "Family History": [family],
        "Diabetes": [diabetes],
        "Obesity": [obesity],
        "Exercise Induced Angina": [angina],
        "Chest Pain Type": [pain]
    })

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    st.subheader("🩺 Prediction Result")
    if pred == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({prob[1]*100:.2f}% confidence)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({prob[0]*100:.2f}% confidence)")

    # Confidence bar
    st.subheader("📊 Prediction Confidence")
    fig, ax = plt.subplots()
    ax.barh(["No Disease", "Disease"], prob, color=["green", "red"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    for i, v in enumerate(prob):
        ax.text(v + 0.02, i, f"{v*100:.2f}%", va='center')
    st.pyplot(fig)

# Footer
st.markdown("<hr><div style='text-align:center;'>Developed by <strong>Bhavpreet Kaur</strong> 💻</div>", unsafe_allow_html=True)


# In[ ]:




