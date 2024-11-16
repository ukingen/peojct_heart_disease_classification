import streamlit as st
import numpy as np
import joblib

# Load the model from the file
model = joblib.load('Heart Disease Classification Model.pkl')


# Title of the app
st.title("Heart Disease Prediction")

# Description
st.write("""
### Predict the likelihood of heart disease based on medical parameters.
Fill out the details below to get started.
""")

# Input fields for the features
age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
sex = st.selectbox("Sex", options=["Male", "Female"])
cp = st.selectbox("Chest Pain Type (CP)", options=[0, 1, 2, 3], format_func=lambda x: f"Type {x}")
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=200, value=120, step=1)
chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200, step=1)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
restecg = st.selectbox("Resting ECG Results (restecg)", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150, step=1)
exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

# Converting categorical inputs to numerical values
sex = 1 if sex == "Male" else 0

# Prepare input features for prediction
input_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Make prediction 
if st.button("Predict"):
    prediction = model.predict(input_features)
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"
    st.subheader(f"Prediction: {result}")