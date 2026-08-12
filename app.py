"""
Streamlit app: Obesity Level Classification Demo
Assignment 2 - Machine Learning (M.Tech AIML/DSE, BITS Pilani WILP)

Features:
  a. CSV upload of test data
  b. Model selection dropdown
  c. Display of evaluation metrics
  d. Confusion matrix + classification report
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import classification_report

st.set_page_config(page_title="Obesity Level Classifier", layout="wide")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}

TARGET_COL = "NObeyesdad"


@st.cache_resource
def load_preprocessor():
    return joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))


@st.cache_resource
def load_model(model_filename):
    return joblib.load(os.path.join(MODEL_DIR, model_filename))


def transform_features(df, preproc):
    df = df.copy()
    for col in preproc["categorical_cols"]:
        le = preproc["encoders"][col]
        df[col] = le.transform(df[col].astype(str))
    scaled_numeric = preproc["scaler"].transform(df[preproc["numeric_cols"]])
    df[preproc["numeric_cols"]] = scaled_numeric
    return df[preproc["feature_cols"]].values


def main():
    st.title("🍎 Obesity Level Classification")
    st.caption(
        "Assignment 2 · Machine Learning · M.Tech (AIML/DSE) · BITS Pilani WILP"
    )
    st.markdown(
        "Predicts obesity level (7 classes) from eating habits and physical "
        "condition data, using the UCI **Estimation of Obesity Levels** dataset. "
        "Upload the provided `test_data.csv`, pick a model, and view its "
        "evaluation on that data."
    )

    preproc = load_preprocessor()

    # --- Sidebar controls -------------------------------------------------
    st.sidebar.header("Controls")
    model_name = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))
    uploaded_file = st.sidebar.file_uploader(
        "Upload test data (CSV)", type=["csv"],
        help="Upload the test_data.csv provided in this repository (or any CSV "
             "with the same 16 feature columns + the NObeyesdad label column).",
    )

    with st.sidebar.expander("Expected columns"):
        st.write("**Categorical:**", ", ".join(preproc["categorical_cols"]))
        st.write("**Numeric:**", ", ".join(preproc["numeric_cols"]))
        st.write("**Target:**", TARGET_COL)

    if uploaded_file is None:
        st.info("⬅️ Upload `test_data.csv` from the sidebar to get started.")
        st.stop()

    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded CSV: {e}")
        st.stop()

    missing_cols = [c for c in preproc["feature_cols"] if c not in data.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
        st.stop()

    has_labels = TARGET_COL in data.columns

    st.subheader("Preview of uploaded data")
    st.dataframe(data.head(10), use_container_width=True)
    st.caption(f"{data.shape[0]} rows × {data.shape[1]} columns")

    # --- Load model and predict -------------------------------------------
    model = load_model(MODEL_FILES[model_name])
    X = transform_features(data, preproc)
    y_pred_enc = model.predict(X)
    y_pred_labels = preproc["target_encoder"].inverse_transform(y_pred_enc)

    result_df = data.copy()
    result_df["Predicted_" + TARGET_COL] = y_pred_labels

    st.subheader(f"Predictions — {model_name}")
    st.dataframe(
        result_df[
            (["NObeyesdad"] if has_labels else [])
            + ["Predicted_" + TARGET_COL]
        ].head(20),
        use_container_width=True,
    )

    # --- Evaluation metrics (only possible if true labels are present) ----
    if has_labels:
        y_true_enc = preproc["target_encoder"].transform(data[TARGET_COL])

        acc = accuracy_score(y_true_enc, y_pred_enc)
        prec = precision_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)
        rec = recall_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)
        f1 = f1_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(y_true_enc, y_pred_enc)
        try:
            y_proba = model.predict_proba(X)
            auc = roc_auc_score(y_true_enc, y_proba, multi_class="ovr", average="weighted")
        except Exception:
            auc = np.nan

        st.subheader("Evaluation metrics")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{acc:.4f}")
        m2.metric("AUC", f"{auc:.4f}" if not np.isnan(auc) else "N/A")
        m3.metric("Precision", f"{prec:.4f}")
        m4.metric("Recall", f"{rec:.4f}")
        m5.metric("F1 Score", f"{f1:.4f}")
        m6.metric("MCC", f"{mcc:.4f}")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Confusion matrix")
            labels = preproc["target_encoder"].classes_
            cm = confusion_matrix(y_true_enc, y_pred_enc, labels=range(len(labels)))
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            st.pyplot(fig)

        with col_right:
            st.subheader("Classification report")
            report = classification_report(
                y_true_enc, y_pred_enc,
                target_names=preproc["target_encoder"].classes_,
                output_dict=True, zero_division=0,
            )
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df, use_container_width=True)
    else:
        st.warning(
            "Uploaded CSV has no `NObeyesdad` column, so evaluation metrics and "
            "the confusion matrix can't be computed — only predictions are shown."
        )

    # --- Model comparison table --------------------------------------------
    st.divider()
    st.subheader("All models — comparison on the held-out test set")
    comparison_path = os.path.join(MODEL_DIR, "metrics_comparison.csv")
    if os.path.exists(comparison_path):
        st.dataframe(pd.read_csv(comparison_path), use_container_width=True)


if __name__ == "__main__":
    main()
