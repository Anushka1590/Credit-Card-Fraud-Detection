# Credit Card Fraud Detection

An end-to-end machine learning project that detects fraudulent credit card transactions using the Kaggle dataset. The project includes data preprocessing, 
model training, evaluation, explainability with SHAP, and a Streamlit dashboard for real-time prediction.
---

## Project Overview

Credit card fraud is rare but costly, accounting for less than 0.17% of transactions in this dataset.
The goal of this project is to build a robust model that can accurately detect fraud despite the severe class imbalance.

**Models experimented:**

- Logistic Regression (`class_weight='balanced'`)
- Random Forest Classifier
- XGBoost Classifier (Selected as final model)

 **XGBoost** was chosen based on:
- Highest ROC–AUC  
- Best tradeoff between precision & recall  
- Strong recall for fraud class 
- Well-calibrated probability outputs

---

## Key Highlights

### Class Imbalance Handling
- `class_weight='balanced'` for Logistic Regression  
- `scale_pos_weight` for XGBoost (computed manually)

### Model Training & Comparison  
All three models were trained and evaluated:

| Model | Strengths | Weaknesses |
|-------|-----------|-------------|
| Logistic Regression | Simple, fast | low fraud recall |
| Random Forest | Good recall | Slower, lower AUC |
| XGBoost (FINAL) | High AUC, reliable fraud detection | Slower training |

---

## Performance (XGBoost)

- **Accuracy:** ~99.9%  
- **Precision (Fraud):** ~0.75  
- **Recall (Fraud):** ~0.88  
- **ROC–AUC:** ~0.99  

---

## Explainability

SHAP plots were used to understand:

- Feature importance  
- Transaction-level predictions
- Model reasoning for fraud classification

---

## Streamlit Dashboard

supports:

**1. Predict on unlabelled CSV** - returns fraud predictions and probabilities.
**2. Evaluate labelled CSV** - generates confusion matrix, classification report, and fraud summary.
**3. Single transaction prediction** -manual input for real-time prediction.

---

## Running Locally

### 1. Clone Repository

git clone https://github.com/Anushka1590/Credit-Card-Fraud-Detection.git

cd Credit-Card-Fraud-Detection

### 2. Install Requirements

pip install -r requirements.txt

### 3. Run Streamlit App

streamlit run app.py

---

## Dataset

Kaggle Credit Card Fraud Dataset (284,807 transactions, 492 frauds)
- PCA-transformed features: V1…V28  
- Highly imbalanced (492 frauds)  

---

## Learnings

- Handling imbalanced datasets effectively
- Evaluating models using precision, recall, and AUC instead of accuracy
- Deploying ML models with Streamlit
- Using SHAP for model interpretability

---

## Future Work

- Explore deep learning approaches (Autoencoders)
- Implement anomaly detection (Isolation Forest)
- Deploy dashboard to cloud
- Develop a real-time fraud detection API 

---
