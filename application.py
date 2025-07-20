#!/usr/bin/env python
# coding: utf-8

# In[11]:


import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("heart_disease_model_v1.pkl")

# Page config
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: red;'>💓 Heart Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Fill out the details to assess the risk.</h4>", unsafe_allow_html=True)

# Form
with st.form("patient_form"):
    st.subheader("Patient Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 20, 100, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        cholesterol = st.slider("Cholesterol", 100, 400, 200)
        blood_pressure = st.slider("Blood Pressure", 80, 200, 120)
        heart_rate = st.slider("Heart Rate", 60, 200, 100)
        smoking = st.selectbox("Smoking", ["Yes", "No"])
        alcohol = st.selectbox("Alcohol Intake", ["None", "Low", "Moderate", "High"])
    with col2:
        exercise_hours = st.slider("Exercise Hours/Week", 0, 20, 3)
        family_history = st.selectbox("Family History", ["Yes", "No"])
        diabetes = st.selectbox("Diabetes", ["Yes", "No"])
        obesity = st.selectbox("Obesity", ["Yes", "No"])
        stress = st.slider("Stress Level", 1, 10, 5)
        blood_sugar = st.slider("Blood Sugar", 70, 250, 100)
        angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        chest_pain = st.selectbox("Chest Pain Type", ["Typical", "Atypical", "Non-anginal", "Asymptomatic"])

    submit = st.form_submit_button("🔍 Predict Risk")

# Prediction
if submit:
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Cholesterol": cholesterol,
        "Blood Pressure": blood_pressure,
        "Heart Rate": heart_rate,
        "Smoking": smoking,
        "Alcohol Intake": alcohol,
        "Exercise Hours": exercise_hours,
        "Family History": family_history,
        "Diabetes": diabetes,
        "Obesity": obesity,
        "Stress Level": stress,
        "Blood Sugar": blood_sugar,
        "Exercise Induced Angina": angina,
        "Chest Pain Type": chest_pain
    }])

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    st.subheader("🩺 Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk ({prob[1]*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk ({prob[0]*100:.2f}%)")

    st.subheader("📊 Prediction Confidence")
    fig, ax = plt.subplots()
    ax.bar(["No Disease", "Heart Disease"], prob, color=["green", "red"])
    st.pyplot(fig)


# In[ ]:




