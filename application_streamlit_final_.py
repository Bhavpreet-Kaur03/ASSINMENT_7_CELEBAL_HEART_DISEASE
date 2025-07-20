#!/usr/bin/env python
# coding: utf-8

# In[16]:


import streamlit as st
import pandas as pd
import joblib

# Load the correct model
model = joblib.load("clean_heart_model.pkl")

st.title("❤️ Heart Disease Prediction")

# User input form
with st.form("heart_form"):
    age = st.number_input("Age", 18, 100)
    cholesterol = st.number_input("Cholesterol", 100, 400)
    bp = st.number_input("Blood Pressure", 80, 200)
    heart_rate = st.number_input("Heart Rate", 50, 200)
    exercise_hours = st.number_input("Exercise Hours/Week", 0, 20)
    stress_level = st.slider("Stress Level (1–10)", 1, 10)
    blood_sugar = st.number_input("Blood Sugar", 50, 200)

    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    alcohol = st.selectbox("Alcohol Intake", ["Yes", "No"])
    family = st.selectbox("Family History", ["Yes", "No"])
    diabetes = st.selectbox("Diabetes", ["Yes", "No"])
    obesity = st.selectbox("Obesity", ["Yes", "No"])
    angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    chest_pain = st.selectbox("Chest Pain Type", ["Type A", "Type B", "Type C", "Type D"])

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([{
        "Age": age,
        "Cholesterol": cholesterol,
        "Blood Pressure": bp,
        "Heart Rate": heart_rate,
        "Exercise Hours": exercise_hours,
        "Stress Level": stress_level,
        "Blood Sugar": blood_sugar,
        "Gender": gender,
        "Smoking": smoking,
        "Alcohol Intake": alcohol,
        "Family History": family,
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





# In[ ]:




