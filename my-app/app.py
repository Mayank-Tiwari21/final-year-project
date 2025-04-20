import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("🔍 Credit Card Fraud Detection App")

st.markdown("""
This app uses a pre-trained machine learning model to predict whether a credit card transaction is **fraudulent** or **genuine**.
""")

# Example feature names (you can change based on your dataset)
feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'Amount']

# Sidebar input for features
st.sidebar.header("Input Transaction Details")

input_data = {}
for feature in feature_names:
    input_data[feature] = st.sidebar.number_input(f"Enter {feature}", value=0.0)

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Show user input
st.subheader("Entered Transaction Data:")
st.write(input_df)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Confidence: {prediction_proba:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction. (Confidence: {prediction_proba:.2f})")
