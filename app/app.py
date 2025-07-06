import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title
st.title("🧾 Customer Churn Prediction Demo")

# Load model, scaler, and columns
model_path = 'final_lr_model.pkl'
scaler_path = 'final_scaler.pkl'
columns_path = 'feature_columns.pkl'

lr = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_columns = joblib.load(columns_path)

# Sidebar for input
st.sidebar.header("Input Customer Details")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fibre optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
MonthlyCharges = st.sidebar.number_input("Monthly Charges", 0.0, 150.0, 70.0)
TotalCharges = st.sidebar.number_input("Total Charges", 0.0, 9000.0, 2500.0)

# Predict button
if st.sidebar.button("Predict"):
    # Create dataframe
    new_customer = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
    })

    # Map binary columns
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        new_customer[col] = new_customer[col].map({'Yes': 1, 'No': 0})

    # One-hot encode
    new_customer_encoded = pd.get_dummies(new_customer, drop_first=True)

    # Align columns
    new_customer_aligned = new_customer_encoded.reindex(columns=feature_columns, fill_value=0)

    # Scale numeric columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    new_customer_aligned[num_cols] = scaler.transform(new_customer_aligned[num_cols])

    # Predict
    pred = lr.predict(new_customer_aligned)[0]
    prob = lr.predict_proba(new_customer_aligned)[0][1]

    # Output
    st.subheader("Prediction Result")
    st.write(f"**Predicted Churn:** {'Yes' if pred == 1 else 'No'}")
    st.write(f"**Churn Probability:** {prob:.2f}")
