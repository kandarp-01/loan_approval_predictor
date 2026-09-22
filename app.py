import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model Files
# -----------------------------
model = joblib.load("DT_grid_loan.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Approval Prediction System")
st.markdown(
    "Predict whether a loan application is **Approved** or **Rejected** using a Machine Learning model."
)

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Applicant Details")

    person_age = st.slider("Age", 18, 70, 30)
    is_male = st.selectbox("Gender", ["Male", "Female"])
    person_income = st.number_input(
        "Annual Income ($)", min_value=1000, max_value=500000, value=50000
    )
    person_emp_exp = st.slider("Employment Experience (Years)", 0, 40, 5)
    credit_score = st.slider("Credit Score", 300, 850, 700)
    cb_person_cred_hist_length = st.slider(
        "Credit History Length (Years)", 1, 30, 8
    )

with col2:
    st.subheader("💰 Loan Details")

    loan_amnt = st.number_input(
        "Loan Amount ($)", min_value=500, max_value=100000, value=10000
    )
    loan_int_rate = st.slider(
        "Interest Rate (%)", 5.0, 25.0, 10.5, step=0.1
    )
    loan_percent_income = st.slider(
        "Loan as % of Income", 0.0, 1.0, 0.20, step=0.01
    )
    previous_default = st.selectbox(
        "Previous Loan Default", ["No", "Yes"]
    )

    home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )

    loan_intent = st.selectbox(
        "Loan Purpose",
        [
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "PERSONAL",
            "DEBTCONSOLIDATION",
            "HOMEIMPROVEMENT"
        ]
    )

    education = st.selectbox(
        "Education",
        [
            "High School",
            "Associate",
            "Bachelor",
            "Master",
            "Doctorate"
        ]
    )

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Loan Status", use_container_width=True):

    # Create dictionary with numerical features
    raw_input = {
        "person_age": person_age,
        "is_male": 1 if is_male == "Male" else 0,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": 1 if previous_default == "Yes" else 0
    }

    # One-Hot Encoding (Education)
    education_columns = [
        "person_education_Associate",
        "person_education_Bachelor",
        "person_education_Doctorate",
        "person_education_High School",
        "person_education_Master"
    ]

    for col in education_columns:
        raw_input[col] = 0

    raw_input[f"person_education_{education}"] = 1

    # One-Hot Encoding (Home Ownership)
    home_columns = [
        "person_home_ownership_MORTGAGE",
        "person_home_ownership_OTHER",
        "person_home_ownership_OWN",
        "person_home_ownership_RENT"
    ]

    for col in home_columns:
        raw_input[col] = 0

    raw_input[f"person_home_ownership_{home_ownership}"] = 1

    # One-Hot Encoding (Loan Intent)
    intent_columns = [
        "loan_intent_DEBTCONSOLIDATION",
        "loan_intent_EDUCATION",
        "loan_intent_HOMEIMPROVEMENT",
        "loan_intent_MEDICAL",
        "loan_intent_PERSONAL",
        "loan_intent_VENTURE"
    ]

    for col in intent_columns:
        raw_input[col] = 0

    raw_input[f"loan_intent_{loan_intent}"] = 1

    # DataFrame
    input_df = pd.DataFrame([raw_input])

    # Fill missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale numerical columns
    numeric_columns = [
        "person_age",
        "person_income",
        "person_emp_exp",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "credit_score"
    ]

    input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])

    # Prediction
    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0]

    confidence = round(max(probability) * 100, 2)

    st.divider()

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
        st.metric("Approval Confidence", f"{confidence}%")
    else:
        st.error("❌ Loan Rejected")
        st.metric("Rejection Confidence", f"{confidence}%")

    st.progress(float(max(probability)))

    st.divider()

    st.subheader("📋 Applicant Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Age",
            "Gender",
            "Income",
            "Employment Experience",
            "Credit Score",
            "Loan Amount",
            "Interest Rate",
            "Loan % Income",
            "Home Ownership",
            "Loan Purpose",
            "Education",
            "Previous Default"
        ],
        "Value": [
            person_age,
            is_male,
            person_income,
            person_emp_exp,
            credit_score,
            loan_amnt,
            f"{loan_int_rate}%",
            loan_percent_income,
            home_ownership,
            loan_intent,
            education,
            previous_default
        ]
    })

    st.dataframe(summary, use_container_width=True)