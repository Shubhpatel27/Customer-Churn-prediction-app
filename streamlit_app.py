import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction")
st.markdown("Choose input method:")

# Dropdown to toggle mode
mode = st.radio("Select Input Method", ["Manual Input", "Upload CSV"])

# Common FastAPI URL
API_URL = "http://127.0.0.1:8000/predict"

# ----------------------------
# 🔹 Manual Form Input
# ----------------------------
if mode == "Manual Input":
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
        feature_list = [
            gender, senior, partner, dependents, tenure, phone, paperless,
            monthly, total, multiple, fiber, no_internet, credit, electronic,
            mailed, tenure_group_mid, charge_ratio, senior_fiber, high_risk
        ]

        mapping = {"Male": 0, "Female": 1, "Yes": 1, "No": 0}
        features = [mapping.get(v, v) for v in feature_list]

        try:
            response = requests.post(API_URL, json={"features": features})
            if response.status_code == 200:
                pred = response.json()["churn_probability"]
                st.success(f"📉 Churn Probability: **{pred:.2%}**")
            else:
                st.error("Prediction failed. Check server or input format.")
        except Exception as e:
            st.error(f"Server error: {e}")

# ----------------------------
# 🔹 CSV Upload
# ----------------------------
else:
    st.markdown("Upload a **preprocessed CSV** file (with same 19 features):")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview", df.head())

        if st.button("Predict for All Rows"):
            try:
                results = []
                for i, row in df.iterrows():
                    row_data = row.tolist()
                    response = requests.post(API_URL, json={"features": row_data})
                    if response.status_code == 200:
                        prob = response.json()["churn_probability"]
                    else:
                        prob = None
                    results.append(prob)

                df["Churn Probability"] = results
                st.success("✅ Predictions complete!")
                st.write(df)

                # Allow CSV download
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error during prediction: {e}")
