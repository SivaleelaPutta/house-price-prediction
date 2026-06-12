import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("house_price_model.pkl")

# Title
st.title("House Price Prediction App")
st.write("Enter house details below to predict price")

# Input fields
area = st.number_input("Area (sq ft)", min_value=300, max_value=10000, value=1000)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=1)
floors = st.number_input("Floors", min_value=1, max_value=5, value=1)
age = st.number_input("Age of House (years)", min_value=0, max_value=100, value=5)
location_score = st.slider("Location Score (1 - 10)", 1, 10, 5)

# Prediction button
if st.button("Predict Price"):
    features = np.array([[area, bedrooms, bathrooms, floors, age, location_score]])
    prediction = model.predict(features)

    st.success(f"Estimated House Price: ₹ {int(prediction[0])}")