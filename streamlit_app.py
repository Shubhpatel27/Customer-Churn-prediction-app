import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction")

st.markdown("Upload customer data (single row) or manually enter below:")

# Manual form
with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100)
    monthly = st.number_input("Monthly Charges", min_value=0.0)
    total = st.number_input("Total Charges", min_value=0.0)
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["Yes", "No"])
    fiber = st.selectbox("Internet Service - Fiber optic", [1, 0])
    no_internet = st.selectbox("Internet Service - No", [0, 1])
    credit = st.selectbox("PaymentMethod: Credit Card (auto)", [1, 0])
    electronic = st.selectbox("PaymentMethod: Electronic Check", [1, 0])
    mailed = st.selectbox("PaymentMethod: Mailed Check", [1, 0])
    tenure_group_mid = st.selectbox("TenureGroup_Mid", [0, 1])
    charge_ratio = st.number_input("Charge Ratio", min_value=0.0)
    senior_fiber = st.selectbox("Senior_Fiber", [0, 1])
    high_risk = st.selectbox("HighRisk", [0, 1])

    submit = st.form_submit_button("Predict")

if submit:
    # Preprocess values to match API input
    feature_list = [
        gender, senior, partner, dependents, tenure, phone, paperless,
        monthly, total, multiple, fiber, no_internet, credit, electronic,
        mailed, tenure_group_mid, charge_ratio, senior_fiber, high_risk
    ]

    # Convert categorical to numeric if needed
    mapping = {"Male": 0, "Female": 1, "Yes": 1, "No": 0}
    features = [mapping.get(v, v) for v in feature_list]

    # Send to FastAPI
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"features": features}
        )
        if response.status_code == 200:
            pred = response.json()["churn_probability"]
            st.success(f"📉 Churn Probability: **{pred:.2%}**")
        else:
            st.error("Prediction failed. Check server or input format.")
    except Exception as e:
        st.error(f"Server error: {e}")
