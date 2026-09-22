# 🏦 Loan Approval Prediction System

<div align="center">

### End-to-End Machine Learning Classification Project using Scikit-Learn & Streamlit

**Predict whether a loan application will be Approved or Rejected using customer financial and personal information.**

![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge\&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge\&logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge\&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</div>

---

## 📌 Project Overview

Loan approval is one of the most common real-world **binary classification** problems in the banking and finance industry. Banks evaluate an applicant's income, employment history, credit score, education, loan amount, and other financial information before deciding whether to approve a loan.

This project builds a **complete Machine Learning pipeline** from data preprocessing to deployment. Multiple supervised learning models were trained, evaluated, and tuned using **RandomizedSearchCV**, and the best-performing model was deployed with **Streamlit**.

### 🎯 Objective

Predict whether a customer's loan application should be:

* ✅ **Approved**
* ❌ **Rejected**

---

# ✨ Features

* 📊 Complete Exploratory Data Analysis (EDA)
* 🧹 Data Cleaning & Preprocessing
* 🔤 One-Hot Encoding for categorical features
* 📈 Feature Scaling using `StandardScaler`
* 🤖 Multiple Classification Models
* 🎯 Hyperparameter Tuning with **RandomizedSearchCV**
* ⚖️ Imbalanced Dataset Handling (SMOTE, Oversampling & Undersampling)
* 🏆 Model Comparison & Selection
* 🌐 Interactive Streamlit Web Application

---

# 🧠 Machine Learning Workflow

```text
Dataset
   │
   ▼
EDA & Data Cleaning
   │
   ▼
Encoding + Feature Scaling
   │
   ▼
Train/Test Split
   │
   ▼
Train Multiple ML Models
(Logistic Regression, SVC, Decision Tree, Random Forest)
   │
   ▼
RandomizedSearchCV
   │
   ▼
Model Evaluation
   │
   ▼
Best Model Saved (.pkl)
   │
   ▼
Streamlit Deployment
```

---

# 📂 Dataset Information

### Dataset Source

Dataset Name: Loan Approval Classification Dataset

Source: Kaggle

Dataset Link: https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data

### Dataset Summary

| Property            | Value                 |
| ------------------- | --------------------- |
| **Rows**            | 45,000                |
| **Features**        | 14                    |
| **Target Variable** | `loan_status`         |
| **Problem Type**    | Binary Classification |

### Target Variable

| Value | Meaning       |
| ----- | ------------- |
| **1** | Loan Approved |
| **0** | Loan Rejected |

---

# 📋 Features Used

### Applicant Information

| Feature          | Description           |
| ---------------- | --------------------- |
| `person_age`     | Applicant age         |
| `is_male`        | Gender (Male/Female)  |
| `person_income`  | Annual income         |
| `person_emp_exp` | Employment experience |

### Credit Information

| Feature                          | Description            |
| -------------------------------- | ---------------------- |
| `credit_score`                   | Applicant credit score |
| `cb_person_cred_hist_length`     | Credit history length  |
| `previous_loan_defaults_on_file` | Previous loan default  |

### Loan Information

| Feature               | Description                         |
| --------------------- | ----------------------------------- |
| `loan_amnt`           | Requested loan amount               |
| `loan_int_rate`       | Loan interest rate                  |
| `loan_percent_income` | Loan amount as percentage of income |

### Encoded Categorical Features

**Education**

* Associate
* Bachelor
* Doctorate
* High School
* Master

**Home Ownership**

* RENT
* OWN
* MORTGAGE
* OTHER

**Loan Purpose**

* Education
* Medical
* Personal
* Venture
* Debt Consolidation
* Home Improvement

---

# 🛠️ Tech Stack

| Category                  | Technology         |
| ------------------------- | ------------------ |
| **Programming Language**  | Python             |
| **Data Analysis**         | Pandas, NumPy      |
| **Visualization**         | Matplotlib, Plotly |
| **Machine Learning**      | Scikit-Learn       |
| **Hyperparameter Tuning** | RandomizedSearchCV |
| **Model Serialization**   | Joblib             |
| **Web Application**       | Streamlit          |

---

# 📁 Project Structure

```bash
Loan Predictor/
│
├── app.py                 # Streamlit Application
├── DT_grid_loan.pkl        # Trained Decision Tree Model
├── scaler.pkl              # StandardScaler Object
├── columns.pkl             # Expected Feature Columns
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```

---

# 📊 Exploratory Data Analysis (EDA)

The dataset was analyzed before model training.

### ✔ Data Understanding

* Dataset shape and structure.
* Data types.
* Missing value analysis.
* Duplicate value checking.
* Unique value analysis.

### ✔ Numerical Feature Analysis

Performed analysis for:

* Income distribution.
* Age distribution.
* Credit score distribution.
* Loan amount distribution.
* Interest rate distribution.
* Loan percentage of income.

### ✔ Categorical Feature Analysis

Analyzed:

* Education distribution.
* Home ownership distribution.
* Loan intent distribution.
* Previous loan default distribution.
* Gender distribution.

### ✔ Target Variable Analysis

* Class distribution.
* Imbalanced dataset identification.
* Loan approval vs rejection ratio.

---

# 🧹 Data Preprocessing

The preprocessing pipeline included several important ML steps.

## Missing Values

* Verified dataset for missing values.
* Dataset contained **no missing values** after preprocessing.

## Feature Encoding

| Feature Type          | Technique        |
| --------------------- | ---------------- |
| Gender                | Binary Encoding  |
| Previous Loan Default | Binary Encoding  |
| Education             | One-Hot Encoding |
| Home Ownership        | One-Hot Encoding |
| Loan Purpose          | One-Hot Encoding |

## Feature Scaling

`StandardScaler` was applied to numerical features.

Scaled columns:

* person_age
* person_income
* person_emp_exp
* loan_amnt
* loan_int_rate
* loan_percent_income
* cb_person_cred_hist_length
* credit_score

---

# 🤖 Machine Learning Models Implemented

The following supervised learning models were trained and evaluated.

| Model                           | Purpose                       |
| ------------------------------- | ----------------------------- |
| Logistic Regression             | Baseline Classification Model |
| Support Vector Classifier (SVC) | Non-linear Classification     |
| Decision Tree Classifier        | Tree-Based Classification     |
| Random Forest Classifier        | Ensemble Classification       |

---

# ⚙️ Hyperparameter Tuning

Hyperparameter tuning was performed using **RandomizedSearchCV**.

### Configuration

* **Cross Validation:** 5-Fold
* **Random State:** 42
* **Search Strategy:** Random Hyperparameter Search

### Why RandomizedSearchCV?

* Faster than GridSearchCV.
* Efficient for larger search spaces.
* Finds near-optimal hyperparameters with fewer iterations.
* Reduces computational cost.

---

# 📈 Model Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

## Final Model Performance

| Model                     | Accuracy | Precision | Recall   | F1 Score |
| ------------------------- | -------- | --------- | -------- | -------- |
| Logistic Regression       | **89%**  | 0.80      | 0.69     | 0.74     |
| Support Vector Classifier | **91%**  | **0.84**  | 0.71     | **0.77** |
| Decision Tree             | **90%**  | 0.83      | **0.72** | 0.77     |

### 🏆 Selected Final Model

**Support Vector Classifier (RandomizedSearchCV)**

Reasons:

* Highest Accuracy.
* Highest Precision.
* Highest F1 Score.
* Best overall balance between loan approval and business risk.

---

# 📉 Confusion Matrix Interpretation

The confusion matrix was used to understand prediction quality from a banking perspective.

| Prediction     | Business Meaning                                  |
| -------------- | ------------------------------------------------- |
| True Positive  | Correctly approved an eligible customer.          |
| True Negative  | Correctly rejected a risky customer.              |
| False Positive | Approved a risky customer (financial risk).       |
| False Negative | Rejected an eligible customer (lost opportunity). |

### Business Insight

The selected model minimized **False Positives**, helping reduce the number of risky loan approvals while maintaining strong overall performance.

---

# ⚖️ Imbalanced Dataset Handling

The dataset contained approximately:

| Class         | Percentage |
| ------------- | ---------- |
| Loan Rejected | **78%**    |
| Loan Approved | **22%**    |

To improve prediction for the minority class, three imbalance handling techniques were explored.

## Techniques Evaluated

| Technique            | Precision | Recall   | F1 Score |
| -------------------- | --------- | -------- | -------- |
| Original Model       | **0.84**  | 0.71     | **0.77** |
| SMOTE                | 0.62      | 0.82     | 0.71     |
| Random Oversampling  | 0.65      | **0.83** | 0.73     |
| Random Undersampling | 0.80      | 0.69     | 0.74     |

### Conclusion

* **SMOTE** significantly improved recall but also increased False Positives.
* **Oversampling** produced similar behavior with slightly better precision than SMOTE.
* **Undersampling** removed useful majority-class information and provided minimal improvement.

The original tuned model was selected because it achieved the best trade-off between precision, recall, F1-score, and business risk.

---

# 🌟 Feature Importance

Feature importance analysis was performed using the Decision Tree model.

Top influential features included:

* Credit Score
* Annual Income
* Loan Amount
* Loan Interest Rate
* Loan Percent Income
* Previous Loan Defaults
* Employment Experience

These features contributed the most toward predicting loan approval.

---

# 🖥️ Streamlit Application

The trained model was deployed using **Streamlit** to provide a simple user interface.

## Application Features

* Applicant Information Form.
* Loan Information Form.
* Education Selection.
* Home Ownership Selection.
* Loan Purpose Selection.
* Previous Loan Default Selection.
* Prediction Confidence.
* Applicant Summary Table.

### User Workflow

1. Enter applicant details.
2. Click **Predict Loan Status**.
3. View prediction result.
4. View confidence score.
5. Review submitted applicant information.

---

# 📦 Model Serialization

The trained model and preprocessing objects were saved using **Joblib**.

Saved files:

```python
DT_grid_loan.pkl
scaler.pkl
columns.pkl
```

These files are loaded directly by the Streamlit application during prediction.

---

# ▶️ Run the Project Locally

## Step 1 — Clone Repository

```bash
git clone <your-github-repository-url>
cd Loan-Predictor
```

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3 — Run Streamlit

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

# 📌 Requirements

```text
streamlit
pandas
numpy
scikit-learn
joblib
```

---

# 💻 Streamlit UI Highlights

The application includes:

* Modern banking-style interface.
* Two-column input layout.
* Numerical and categorical input handling.
* Confidence percentage using prediction probability.
* Applicant summary after prediction.

---

# 📚 Concepts Covered

### Data Analysis

* Exploratory Data Analysis (EDA)
* Correlation Analysis
* Outlier Detection
* Feature Distribution Analysis

### Data Preprocessing

* Binary Encoding
* One-Hot Encoding
* StandardScaler
* Feature Engineering

### Supervised Learning

* Logistic Regression
* Support Vector Machine
* Decision Tree
* Random Forest

### Model Optimization

* RandomizedSearchCV
* Cross Validation
* Hyperparameter Tuning

### Model Evaluation

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

### Deployment

* Joblib Model Serialization
* Streamlit Web Application

---

# 🎓 Key Learning Outcomes

This project helped me learn:

* Building an end-to-end supervised machine learning pipeline.
* Performing EDA before model training.
* Handling categorical and numerical features correctly.
* Feature scaling using StandardScaler.
* Comparing multiple classification algorithms.
* Hyperparameter tuning with RandomizedSearchCV.
* Understanding precision-recall trade-offs in imbalanced datasets.
* Experimenting with SMOTE, Oversampling, and Undersampling.
* Deploying a trained ML model using Streamlit.

---

# 🚀 Future Improvements

Planned enhancements for this project include:

* ROC Curve & ROC-AUC analysis.
* Precision-Recall Curve.
* Threshold tuning.
* SHAP Explainability.
* XGBoost & LightGBM comparison.
* Docker containerization.
* Cloud deployment.

---

# 👨‍💻 Author

## Kandarp Dubey

Machine Learning | MERN Stack | AI & GenAI Enthusiast

Currently building production-style Machine Learning and AI projects while following a structured ML Engineer roadmap.

---

<div align="center">

### ⭐ If you found this project helpful, consider giving it a Star.

**Built with ❤️ using Python, Scikit-Learn, and Streamlit**

</div>
