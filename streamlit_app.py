import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# APP + MODEL
# -----------------------------
st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction")
st.markdown("Upload customer data below or manually enter values to predict churn.")

@st.cache_resource
def load_model():
    # Make sure churn_model.pkl is in the same folder as this file
    return joblib.load("churn_model.pkl")

model = load_model()

# The exact feature order your model expects
FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check', 'TenureGroup_Mid', 'ChargeRatio',
    'Senior_Fiber', 'HighRisk'
]

# -----------------------------
# RAW -> PROCESSED
# -----------------------------
def preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # robust numeric casting
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')

        # binary maps
        yes_no = {'Yes': 1, 'No': 0}
        male_female = {'Male': 0, 'Female': 1}  # <- match training
        df['gender'] = df['gender'].map(male_female)
        df['Partner'] = df['Partner'].map(yes_no)
        df['Dependents'] = df['Dependents'].map(yes_no)
        df['PhoneService'] = df['PhoneService'].map(yes_no)
        df['PaperlessBilling'] = df['PaperlessBilling'].map(yes_no)
        df['MultipleLines_Yes'] = df['MultipleLines'].map(yes_no)

        # Internet service
        df['InternetService_Fiber optic'] = df['InternetService'].apply(lambda x: 1 if x == 'Fiber optic' else 0)
        df['InternetService_No'] = df['InternetService'].apply(lambda x: 1 if x == 'No' else 0)

        # Payment method 1-hot (subset used in training)
        df['PaymentMethod_Credit card (automatic)'] = (df['PaymentMethod'] == 'Credit card (automatic)').astype(int)
        df['PaymentMethod_Electronic check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
        df['PaymentMethod_Mailed check'] = (df['PaymentMethod'] == 'Mailed check').astype(int)

        # Engineered features (replicate training logic)
        df['TenureGroup_Mid'] = df['tenure'].apply(lambda x: 1 if pd.notnull(x) and 12 <= x <= 36 else 0)
        df['ChargeRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
        df['Senior_Fiber'] = df['SeniorCitizen'] * df['InternetService_Fiber optic']
        df['HighRisk'] = (df['MonthlyCharges'] > 80).astype(int)

        # final selection & order
        out = df[FEATURES].copy()
        return out
    except Exception as e:
        raise ValueError(f"Preprocessing error: {e}")

# -----------------------------
# BULK PREDICTION (CSV)
# -----------------------------
st.header("📤 Upload Data for Bulk Prediction")

input_type = st.radio("Choose input type:", ["Raw CSV", "Preprocessed CSV"])
csv_file = st.file_uploader("Upload CSV file", type=["csv"])

if csv_file:
    df_uploaded = pd.read_csv(csv_file)
    st.write("📄 Uploaded Data Preview:", df_uploaded.head())

    if input_type == "Raw CSV":
        try:
            df_processed = preprocess_raw(df_uploaded.copy())
            st.success("✅ Raw data successfully preprocessed.")
        except Exception as e:
            st.error(f"Error in preprocessing: {e}")
            st.stop()
    else:
        # Ensure preprocessed file has the exact columns in order
        missing = [c for c in FEATURES if c not in df_uploaded.columns]
        if missing:
            st.error(f"Your preprocessed CSV is missing required columns: {missing}")
            st.stop()
        df_processed = df_uploaded[FEATURES].copy()

    # Clean NaNs (predict_proba can't handle NaN)
    if df_processed.isnull().any().any():
        st.warning("⚠️ Some values were missing — filling NaNs with 0.")
        df_processed = df_processed.fillna(0)

    try:
        # Predict probabilities for all rows
        probs = model.predict_proba(df_processed.values)[:, 1]
        df_result = df_uploaded.copy()
        df_result["Churn Probability"] = probs
        st.write("🔮 Predictions:", df_result)

        # Download button
        st.download_button(
            "📥 Download Predictions",
            df_result.to_csv(index=False).encode("utf-8"),
            "churn_predictions.csv",
            "text/csv"
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()

# -----------------------------
# MANUAL SINGLE PREDICTION
# -----------------------------
st.subheader("📝 Or Enter Customer Info Manually")

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
    mapping = {"Male": 0, "Female": 1, "Yes": 1, "No": 0}
    features = [
        mapping[gender], senior, mapping[partner], mapping[dependents], float(tenure),
        mapping[phone], mapping[paperless], float(monthly), float(total), mapping[multiple],
        int(fiber), int(no_internet), int(credit), int(electronic), int(mailed),
        int(tenure_group_mid), float(charge_ratio), int(senior_fiber), int(high_risk)
    ]

    try:
        X = np.array(features, dtype=float).reshape(1, -1)
        proba = float(model.predict_proba(X)[0, 1])
        st.success(f"📉 Churn Probability: **{proba:.2%}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
