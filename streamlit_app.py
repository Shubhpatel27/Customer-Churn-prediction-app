import streamlit as st
import requests
import pandas as pd

# Load model once at startup
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()

# ---- APP SETUP ----
st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction")
st.markdown("Upload customer data below or manually enter values to predict churn.")

# ---- RAW TO PROCESSED TRANSFORMATION ----
def preprocess_raw(df):
    try:
        # Type casting to avoid string-int concat issues
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0)

        df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
        df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
        df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
        df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
        df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
        df['MultipleLines_Yes'] = df['MultipleLines'].map({'Yes': 1, 'No': 0})
        df['InternetService_Fiber optic'] = df['InternetService'].map(lambda x: 1 if x == 'Fiber optic' else 0)
        df['InternetService_No'] = df['InternetService'].map(lambda x: 1 if x == 'No' else 0)

        df['PaymentMethod_Credit card (automatic)'] = df['PaymentMethod'].apply(lambda x: 1 if x == 'Credit card (automatic)' else 0)
        df['PaymentMethod_Electronic check'] = df['PaymentMethod'].apply(lambda x: 1 if x == 'Electronic check' else 0)
        df['PaymentMethod_Mailed check'] = df['PaymentMethod'].apply(lambda x: 1 if x == 'Mailed check' else 0)

        df['TenureGroup_Mid'] = df['tenure'].apply(lambda x: 1 if 12 <= x <= 36 else 0)
        df['ChargeRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
        df['Senior_Fiber'] = df['SeniorCitizen'] * df['InternetService_Fiber optic']
        df['HighRisk'] = (df['MonthlyCharges'] > 80).astype(int)

        features = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
            'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No',
            'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
            'PaymentMethod_Mailed check', 'TenureGroup_Mid', 'ChargeRatio',
            'Senior_Fiber', 'HighRisk'
        ]
        return df[features]
    except Exception as e:
        raise ValueError(f"Preprocessing error: {e}")

# ---- FILE UPLOAD HANDLING ----
st.header("📤 Upload Data for Bulk Prediction")

input_type = st.radio("Choose input type:", ["Raw CSV", "Preprocessed CSV"])
csv_file = st.file_uploader("Upload CSV file", type=["csv"])

if csv_file:
    df = pd.read_csv(csv_file)
    st.write("📄 Uploaded Data Preview:", df.head())

    if input_type == "Raw CSV":
        try:
            df_processed = preprocess_raw(df.copy())
            if df_processed.isnull().any().any():
                st.warning("⚠️ Some values were missing — filling NaNs with 0.")
            df_processed = df_processed.fillna(0)

            st.success("✅ Raw data successfully preprocessed.")
        except Exception as e:
            st.error(f"Error in preprocessing: {e}")
            st.stop()
    else:
        df_processed = df.copy()

    try:
        predictions = []
        for _, row in df_processed.iterrows():
            row_clean = row.fillna(0).tolist()  # Ensure no NaNs in row
            res = requests.post("http://127.0.0.1:8000/predict", json={"features": row_clean})

            if res.status_code == 200:
                predictions.append(res.json()["churn_probability"])
            else:
                predictions.append(None)

        df["Churn Probability"] = predictions
        st.write("🔮 Predictions:", df)
        st.download_button("📥 Download Predictions", df.to_csv(index=False), "churn_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Failed to connect to prediction API: {e}")

st.divider()

# ---- MANUAL INPUT SECTION ----
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
        mapping[gender], senior, mapping[partner], mapping[dependents], tenure,
        mapping[phone], mapping[paperless], monthly, total, mapping[multiple],
        fiber, no_internet, credit, electronic, mailed, tenure_group_mid,
        charge_ratio, senior_fiber, high_risk
    ]

    try:
        #res = requests.post("http://127.0.0.1:8000/predict", json={"features": features})
        #if res.status_code == 200:
           # pred = res.json()["churn_probability"]
            #st.success(f"📉 Churn Probability: **{pred:.2%}**")
        if preds ==: model.predict(df)
        else:
            st.error("Prediction failed.")
    except Exception as e:
        st.error(f"Error contacting API: {e}")




