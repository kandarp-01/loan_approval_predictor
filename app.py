import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD MODEL FILES
# ==========================================================
model = joblib.load("DT_loan.pkl")
expected_columns = joblib.load("columns.pkl")

# ==========================================================
# CUSTOM CSS
# ==========================================================
st.markdown("""
<style>

.main-title{
    font-size:40px;
    font-weight:700;
    color:#1D4ED8;
}

.subtitle{
    color:#64748B;
    font-size:17px;
}

.metric-card{
    background:#F8FAFC;
    border:1px solid #E2E8F0;
    border-radius:15px;
    padding:15px;
    text-align:center;
}

.small-title{
    color:#1E3A8A;
    font-weight:600;
    margin-bottom:8px;
}

hr{
    margin-top:15px;
    margin-bottom:15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<p class="main-title">🏦 Loan Approval Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Predict whether a customer loan application will be <b>Approved</b> or <b>Rejected</b> using a Decision Tree Machine Learning model.</p>',
    unsafe_allow_html=True
)

st.divider()

# ==========================================================
# SIDEBAR INFO
# ==========================================================

st.sidebar.header("ℹ️ Model Information")

st.sidebar.success("Model : Decision Tree Classifier")

st.sidebar.write("""
This model predicts loan approval based on:

- Applicant Profile
- Credit History
- Income
- Employment Experience
- Loan Details
- Engineered Financial Features
""")

st.sidebar.divider()

# ==========================================================
# INPUT SECTION
# ==========================================================

left, right = st.columns(2)

# -----------------------------
# Applicant Details
# -----------------------------
with left:

    st.subheader("👤 Applicant Information")

    person_age = st.slider("Age",18,70,30)

    person_gender = st.radio(
        "Gender",
        ["Male","Female"],
        horizontal=True
    )

    person_income = st.number_input(
        "Annual Income ($)",
        min_value=1000,
        value=50000,
        step=1000
    )

    person_emp_exp = st.slider(
        "Employment Experience (Years)",
        0,40,5
    )

    credit_score = st.slider(
        "Credit Score",
        300,850,700
    )

    cb_person_cred_hist_length = st.slider(
        "Credit History Length",
        1,30,8
    )

# -----------------------------
# Loan Details
# -----------------------------
with right:

    st.subheader("💰 Loan Information")

    loan_amnt = st.number_input(
        "Loan Amount ($)",
        min_value=500,
        value=10000,
        step=500
    )

    loan_int_rate = st.slider(
        "Interest Rate (%)",
        5.0,25.0,10.5,0.1
    )

    loan_percent_income = st.slider(
        "Loan / Income Ratio",
        0.00,1.00,0.20,0.01
    )

    previous_default = st.radio(
        "Previous Loan Default",
        ["No","Yes"],
        horizontal=True
    )

    home_ownership = st.selectbox(
        "Home Ownership",
        ["MORTGAGE","OWN","RENT","OTHER"]
    )

    loan_intent = st.selectbox(
        "Loan Purpose",
        [
            "DEBTCONSOLIDATION",
            "EDUCATION",
            "HOMEIMPROVEMENT",
            "MEDICAL",
            "PERSONAL",
            "VENTURE"
        ]
    )

    education = st.selectbox(
        "Education",
        [
            "Associate",
            "Bachelor",
            "Doctorate",
            "High School",
            "Master"
        ]
    )

st.divider()

# ==========================================================
# LIVE RISK DASHBOARD
# ==========================================================

st.subheader("📊 Live Risk Dashboard")

c1,c2,c3,c4 = st.columns(4)

c1.metric("💳 Credit Score", credit_score)

c2.metric("💸 Loan / Income", f"{loan_percent_income:.2f}")

c3.metric("💼 Experience", f"{person_emp_exp} yrs")

income_left = person_income-loan_amnt

c4.metric("💵 Income Left", f"${income_left:,}")

# Risk Indicator

risk_score = 0

if credit_score < 550:
    risk_score += 1

if loan_percent_income > 0.50:
    risk_score += 1

if previous_default == "Yes":
    risk_score += 2

if person_emp_exp < 2:
    risk_score += 1

if risk_score >= 3:
    st.error("🔴 High Risk Applicant")
elif risk_score == 2:
    st.warning("🟠 Medium Risk Applicant")
else:
    st.success("🟢 Low Risk Applicant")

st.divider()

# ==========================================================
# PREDICTION
# ==========================================================

if st.button("🚀 Predict Loan Status", use_container_width=True):

    # -------------------------
    # Raw Numerical Features
    # -------------------------
    raw_input = {
        "person_age": person_age,
        "person_gender": 1 if person_gender=="Male" else 0,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": 1 if previous_default=="Yes" else 0
    }

    # ======================================================
    # FEATURE ENGINEERING
    # ======================================================

    raw_input["loan_income_gap"] = person_income-loan_amnt

    raw_input["credit_age_ratio"] = round(
        cb_person_cred_hist_length/person_age,
        3
    )

    raw_input["experience_age_ratio"] = round(
        person_emp_exp/person_age,
        3
    )

    raw_input["high_risk_flag"] = int(
        loan_percent_income>0.50 or previous_default=="Yes"
    )

    # ======================================================
    # ONE HOT ENCODING
    # ======================================================

    # Home Ownership
    for col in [
        "person_home_ownership_OTHER",
        "person_home_ownership_OWN",
        "person_home_ownership_RENT"
    ]:
        raw_input[col]=0

    if home_ownership!="MORTGAGE":
        raw_input[f"person_home_ownership_{home_ownership}"]=1

    # Education
    for col in [
        "person_education_Bachelor",
        "person_education_Doctorate",
        "person_education_High School",
        "person_education_Master"
    ]:
        raw_input[col]=0

    if education!="Associate":
        raw_input[f"person_education_{education}"]=1

    # Loan Intent
    for col in [
        "loan_intent_EDUCATION",
        "loan_intent_HOMEIMPROVEMENT",
        "loan_intent_MEDICAL",
        "loan_intent_PERSONAL",
        "loan_intent_VENTURE"
    ]:
        raw_input[col]=0

    if loan_intent!="DEBTCONSOLIDATION":
        raw_input[f"loan_intent_{loan_intent}"]=1

    # ======================================================
    # CREATE DATAFRAME
    # ======================================================

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0

    input_df=input_df[expected_columns]

    # ======================================================
    # PREDICTION
    # ======================================================

    prediction=model.predict(input_df)[0]
    probability=model.predict_proba(input_df)[0]

    confidence=round(max(probability)*100,2)

    st.divider()
    st.subheader("🎯 Prediction Result")

    col1,col2=st.columns([2,1])

    with col1:

        if prediction==1:
            st.success("## ✅ Loan Approved")
        else:
            st.error("## ❌ Loan Rejected")

        st.progress(max(probability))

    with col2:
        st.metric("Model Confidence",f"{confidence}%")

    # ======================================================
    # Probability Visualization
    # ======================================================

    st.subheader("📈 Approval Probability")

    prob_df=pd.DataFrame({
        "Status":["Rejected","Approved"],
        "Probability":probability
    })

    st.bar_chart(prob_df.set_index("Status"))

    st.divider()

    # ======================================================
    # Risk Explanation
    # ======================================================

    st.subheader("🧠 Why did the model think this?")

    explanation=[]

    if loan_percent_income>0.50:
        explanation.append("⚠️ Loan amount is more than **50%** of annual income.")

    if previous_default=="Yes":
        explanation.append("⚠️ Applicant has a **previous loan default history**.")

    if credit_score<550:
        explanation.append("⚠️ Credit score is considered **low risk profile**.")

    if person_emp_exp<2:
        explanation.append("⚠️ Employment experience is relatively low.")

    if income_left<10000:
        explanation.append("⚠️ Disposable income after loan is quite low.")

    if len(explanation)==0:
        st.success(
            "No major financial risk indicators detected based on the entered information."
        )
    else:
        for item in explanation:
            st.warning(item)

    st.divider()

    # ======================================================
    # Applicant Summary
    # ======================================================

    st.subheader("📋 Applicant Summary")

    summary=pd.DataFrame({
        "Feature":[
            "Age",
            "Gender",
            "Annual Income",
            "Employment Experience",
            "Credit Score",
            "Credit History Length",
            "Loan Amount",
            "Interest Rate",
            "Loan / Income Ratio",
            "Loan Income Gap",
            "Home Ownership",
            "Loan Purpose",
            "Education",
            "Previous Default"
        ],
        "Value":[
            person_age,
            person_gender,
            f"${person_income:,}",
            f"{person_emp_exp} Years",
            credit_score,
            f"{cb_person_cred_hist_length} Years",
            f"${loan_amnt:,}",
            f"{loan_int_rate}%",
            loan_percent_income,
            f"${income_left:,}",
            home_ownership,
            loan_intent,
            education,
            previous_default
        ]
    })

    st.dataframe(summary, use_container_width=True)

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "🏦 Loan Approval Prediction • Decision Tree Classifier • Built with Streamlit"
)