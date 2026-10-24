import streamlit as st
import joblib
import numpy as np

# Model load karein
model = joblib.load('house_model.pkl')

st.title("House Price Prediction App")
st.write("Machine Learning Assignment (AIC354)")

# User inputs (apnay dataset ke features ke mutabiq)
area = st.number_input("Enter feature value:", min_value=0.0)

if st.button("Predict Price"):
    # Features array (model ki zaroorat ke mutabiq shape dein)
    features = np.array([[area]]) 
    prediction = model.predict(features)
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")