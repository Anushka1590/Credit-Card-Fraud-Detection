import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve,
                             precision_recall_curve, roc_auc_score, precision_recall_fscore_support)

# --------------------
# Configuration
# --------------------
FEATURES = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# --------------------
# Helpers to load model and scaler
# --------------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        return None

@st.cache_resource
def load_scaler(path=SCALER_PATH):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading scaler from {path}: {e}")
        return None

model = load_model()
scaler = load_scaler()

# --------------------
# Small utility functions
# --------------------

def predict_single(feature_list, threshold=0.5):
    """feature_list should be in the same order as FEATURES"""
    arr = np.array(feature_list).reshape(1, -1)
    scaled = scaler.transform(arr)
    prob = model.predict_proba(scaled)[0][1]
    pred = int(prob >= threshold)
    return pred, prob


def predict_batch(df, threshold=0.5):
    X = df[FEATURES].values
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:,1]
    preds = (probs >= threshold).astype(int)
    df_out = df.copy()
    df_out["fraud_prob"] = probs
    df_out["predicted_class"] = preds
    return df_out


def plot_confusion(y_true, y_pred, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return ax


def plot_roc(y_true, y_prob, ax=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}")
    ax.plot([0,1],[0,1], linestyle='--', color='grey')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return ax


def plot_pr(y_true, y_prob, ax=None):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(True)
    return ax

# --------------------
# Sidebar navigation
# --------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home / Overview", "Single Prediction", "Batch Prediction / Upload", "Model Metrics (upload test set)"])

# --------------------
# Page: Home / Overview
# --------------------
if page == "Home / Overview":
    st.title("Credit Card Fraud Detection — Demo")
    st.markdown("""
    This demo uses your saved model (`model.pkl`) and scaler (`scaler.pkl`).\
    \n**Important:** Ensure `model.pkl` and `scaler.pkl` exist in the same directory as this app.\
    """)

    col1, col2 = st.columns([2,3])

    with col1:
        st.subheader("Model & Data")
        st.write(f"**Model loaded:** {type(model).__name__ if model else 'None'}")
        st.write(f"**Scaler loaded:** {type(scaler).__name__ if scaler else 'None'}")
        st.write("\n")
        st.info("If you want the app to show model metrics, upload a test set on 'Model Metrics' page (CSV with same features).")

    with col2:
        st.subheader("Quick Demo")
        st.write("Use the 'Single Prediction' page to test a single transaction, or 'Batch Prediction' to upload a CSV and get predictions.")

    st.markdown("---")
    st.subheader("About this project")
    st.markdown(
        """
        - Dataset: Credit Card Fraud (PCA features V1..V28, Time, Amount)\
        - Models tried: Logistic Regression (baseline), Random Forest, XGBoost (final)\
        - Recommended final model: XGBoost (highest ROC-AUC / best recall)\
        - This app performs scaling using the shipped scaler and inference using the shipped model.
        """
    )

# --------------------
# Page: Single Prediction
# --------------------
elif page == "Single Prediction":
    st.header("Single Transaction Prediction")

    st.markdown("Enter the feature values exactly in the same order as used during training.")

    with st.form(key='single_form'):
        cols = st.columns(4)
        inputs = []
        # Generate input boxes for all features
        for i, feat in enumerate(FEATURES):
            col = cols[i % 4]
            if feat == 'Time' or feat == 'Amount':
                val = col.number_input(feat, value=0.0, format="%f")
            else:
                val = col.number_input(feat, value=0.0, format="%f")
            inputs.append(val)

        threshold = st.slider("Prediction threshold", 0.0, 1.0, 0.5, 0.01)
        submit = st.form_submit_button("Predict")

    if submit:
        if model is None or scaler is None:
            st.error("Model or scaler not loaded. Place model.pkl and scaler.pkl in app folder.")
        else:
            pred, prob = predict_single(inputs, threshold)
            st.write(f"**Fraud probability**: {prob:.4f}")
            if pred == 1:
                st.error("⚠ Fraud Detected")
            else:
                st.success("✓ Not Fraudulent")

# --------------------
# Page: Batch Prediction / Upload
# --------------------
elif page == "Batch Prediction / Upload":
    st.header("Batch Prediction — Upload CSV")
    st.markdown("Upload a CSV file with the same features (Time, V1..V28, Amount). The app will return predictions and a downloadable CSV.")

    uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        # Check required columns
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing columns in uploaded CSV: {missing}")
            st.stop()

        threshold = st.slider("Prediction threshold", 0.0, 1.0, 0.5, 0.01)
        df_out = predict_batch(df, threshold)

        st.write("### Predictions (head)")
        st.dataframe(df_out.head())

        st.write("### Summary")
        total = len(df_out)
        predicted_frauds = df_out['predicted_class'].sum()
        st.write(f"Total rows: {total}")
        st.write(f"Predicted frauds: {predicted_frauds}")

        # Download link
        csv = df_out.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions CSV", data=csv, file_name='predictions.csv', mime='text/csv')

# --------------------
# Page: Model Metrics (upload test set)
# --------------------
elif page == "Model Metrics (upload test set)":
    st.header("Model Metrics — Upload Test Set (CSV)")
    st.markdown("Upload a labelled test CSV containing columns: Time, V1..V28, Amount, Class (0 or 1). The app will compute metrics and show ROC/PR curves.")

    test_file = st.file_uploader("Upload labelled test CSV", type=["csv"]) 

    if test_file is not None:
        try:
            df_test = pd.read_csv(test_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        # Verify columns
        missing = [c for c in FEATURES + ['Class'] if c not in df_test.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        X_test_local = df_test[FEATURES].values
        y_test_local = df_test['Class'].values
        X_test_scaled_local = scaler.transform(X_test_local)

        probs = model.predict_proba(X_test_scaled_local)[:,1]
        preds = (probs >= 0.5).astype(int)

        st.subheader("Classification Report")
        st.text(classification_report(y_test_local, preds))

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        plot_confusion(y_test_local, preds, ax)
        st.pyplot(fig)

        st.subheader("ROC Curve")
        fig2, ax2 = plt.subplots()
        plot_roc(y_test_local, probs, ax2)
        st.pyplot(fig2)

        st.subheader("Precision-Recall Curve")
        fig3, ax3 = plt.subplots()
        plot_pr(y_test_local, probs, ax3)
        st.pyplot(fig3)

        st.write("### Metrics summary")
        st.write(f"ROC-AUC: {roc_auc_score(y_test_local, probs):.4f}")
        prec, rec, f1, _ = precision_recall_fscore_support(y_test_local, preds, average='binary', zero_division=0)
        st.write(f"Precision: {prec:.4f}")
        st.write(f"Recall: {rec:.4f}")
        st.write(f"F1-score: {f1:.4f}")

# --------------------
# Footer
# --------------------
st.markdown("---")
st.caption("Built for demonstration. Make sure the features match exactly what the model was trained on.")
