#!/usr/bin/env python
# coding: utf-8

# In[13]:


import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("heart_disease_model_v2.pkl")

st.title("❤️ Heart Disease Prediction App")

# User input form
with st.form("user_input_form"):
    age = st.number_input("Age", 18, 100)
    cholesterol = st.number_input("Cholesterol", 100, 400)
    blood_pressure = st.number_input("Blood Pressure", 80, 200)
    heart_rate = st.number_input("Heart Rate", 50, 200)
    exercise_hours = st.number_input("Exercise Hours per Week", 0, 20)
    stress_level = st.slider("Stress Level", 1, 10)
    blood_sugar = st.number_input("Blood Sugar", 50, 200)

    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    alcohol = st.selectbox("Alcohol Intake", ["Yes", "No"])
    family_history = st.selectbox("Family History of Disease", ["Yes", "No"])
    diabetes = st.selectbox("Diabetes", ["Yes", "No"])
    obesity = st.selectbox("Obesity", ["Yes", "No"])
    angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    chest_pain = st.selectbox("Chest Pain Type", ["Type A", "Type B", "Type C", "Type D"])

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([{
        "Age": age,
        "Cholesterol": cholesterol,
        "Blood Pressure": blood_pressure,
        "Heart Rate": heart_rate,
        "Exercise Hours": exercise_hours,
        "Stress Level": stress_level,
        "Blood Sugar": blood_sugar,
        "Gender": gender,
        "Smoking": smoking,
        "Alcohol Intake": alcohol,
        "Family History": family_history,
        "Diabetes": diabetes,
        "Obesity": obesity,
        "Exercise Induced Angina": angina,
        "Chest Pain Type": chest_pain
    }])

    prediction = model.predict(input_df)[0]
    result = "🛑 High Risk of Heart Disease" if prediction == 1 else "✅ Low Risk of Heart Disease"
    st.subheader("Prediction Result:")
    st.success(result)


# In[ ]:




