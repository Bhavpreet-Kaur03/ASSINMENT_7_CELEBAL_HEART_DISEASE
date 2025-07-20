#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import sklearn
import json

# Display version info
st.sidebar.write(f"scikit-learn: {sklearn.__version__}")

# Set Streamlit page config
st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")

# App title
st.markdown("<h1 style='text-align: center; color: #c0392b;'>💓 Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #7f8c8d;'>Enter patient health details below to assess heart disease risk.</h4>", unsafe_allow_html=True)

# Load model and info
try:
    model = joblib.load("./heart_disease_pipeline.pkl")  # Relative path
    st.sidebar.success("✅ Model loaded")
    
    # Try to load model info
    try:
        with open("./model_info.json", "r") as f:  # Relative path
            model_info = json.load(f)
        st.sidebar.write(f"Model trained with: {model_info['sklearn_version']}")
    except:
        st.sidebar.warning("⚠️ Model info not found")
        
except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")
    st.stop()

# Input form
with st.form("prediction_form"):
    st.subheader("🧍‍♂️ Patient Health Profile")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 100, 50)
        gender = st.selectbox("Gender", ["Male", "Female"])
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 400, 200)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 80, 200, 120)
        heart_rate = st.slider("Heart Rate (bpm)", 60, 200, 100)
        smoking = st.selectbox("Smoking", ["Never", "Former", "Current"])
        alcohol = st.selectbox("Alcohol Intake", ["None", "Moderate", "Heavy"])

    with col2:
        exercise_hours = st.slider("Exercise Hours per Week", 0, 20, 3)
        family_history = st.selectbox("Family History of Heart Disease", ["Yes", "No"])
        diabetes = st.selectbox("Diabetes", ["Yes", "No"])
        obesity = st.selectbox("Obesity", ["Yes", "No"])
        stress = st.slider("Stress Level (1 = Low, 10 = High)", 1, 10, 5)
        blood_sugar = st.slider("Blood Sugar (mg/dL)", 70, 250, 100)
        angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])

    submitted = st.form_submit_button("🔍 Predict Risk")

if submitted:
    # Create input dataframe
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

    try:
        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        # Show result
        st.markdown("---")
        st.subheader("🩺 Prediction Result")
        
        if prediction == 1:
            st.error(f"⚠️ **High Risk of Heart Disease**")
            st.error(f"Confidence: {probabilities[1]*100:.1f}%")
        else:
            st.success(f"✅ **Low Risk of Heart Disease**")
            st.success(f"Confidence: {probabilities[0]*100:.1f}%")

        # Confidence visualization
        st.subheader("📊 Prediction Confidence")
        fig, ax = plt.subplots(figsize=(8, 3))
        labels = ["Low Risk", "High Risk"]
        colors = ['#27ae60', '#c0392b']
        bars = ax.barh(labels, probabilities, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Confidence Score")
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            ax.text(prob + 0.02, i, f"{prob*100:.1f}%", 
                   va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.info("💡 **Note:** This is a prediction tool for educational purposes. Always consult healthcare professionals for medical advice.")

    except Exception as e:
        st.error("❌ **Prediction Failed**")
        st.write(f"**Error:** {str(e)}")
        st.write("**Troubleshooting:**")
        st.write("- Model compatibility issue")
        st.write("- Check if model was trained with same scikit-learn version")
        
        # Debug info
        with st.expander("🔧 Debug Information"):
            st.write("**Input data:**", input_df)
            st.write("**Data types:**", input_df.dtypes.to_dict())

# Footer
st.markdown("""
<hr style="border: 1px solid #eee;" />
<div style="text-align: center; color: gray;">
    💻 Developed by <strong>Bhavpreet Kaur</strong>
</div>
""", unsafe_allow_html=True)


# In[ ]:




