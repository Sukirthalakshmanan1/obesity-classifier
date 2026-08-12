"""
train_models.py
----------------
Trains 5 classification models on the UCI "Estimation of Obesity Levels
Based On Eating Habits and Physical Condition" dataset, evaluates them,
and saves:
    - Trained model files          -> model/*.pkl
    - Fitted preprocessing objects -> model/preprocessor.pkl
    - Held-out test split (raw, human-readable) -> ../test_data.csv
    - Metrics comparison table     -> model/metrics_comparison.csv

Run this once from the project root:
    python model/train_models.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_DATA_PATH = os.path.join(HERE, "..", "data", "ObesityDataSet_raw_and_data_sinthetic.csv")
TEST_DATA_OUT = os.path.join(ROOT, "test_data.csv")

TARGET_COL = "NObeyesdad"
CATEGORICAL_COLS = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]
NUMERIC_COLS = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS  # 16 features total


def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def build_preprocessor(df):
    """Fit label encoders for categorical columns + target, and a scaler for numeric columns."""
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(df[col])
        encoders[col] = le

    target_encoder = LabelEncoder()
    target_encoder.fit(df[TARGET_COL])

    scaler = StandardScaler()
    scaler.fit(df[NUMERIC_COLS])

    return {
        "encoders": encoders,
        "target_encoder": target_encoder,
        "scaler": scaler,
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "feature_cols": FEATURE_COLS,
    }


def transform(df, preproc):
    """Apply fitted encoders/scaler to a dataframe of raw features. Returns X (np.array)."""
    df = df.copy()
    for col in preproc["categorical_cols"]:
        le = preproc["encoders"][col]
        df[col] = le.transform(df[col])

    scaled_numeric = preproc["scaler"].transform(df[preproc["numeric_cols"]])
    df[preproc["numeric_cols"]] = scaled_numeric

    return df[preproc["feature_cols"]].values


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE
        ),
    }


def evaluate(model, X_test, y_test, n_classes):
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        try:
            metrics["AUC"] = roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="weighted"
            )
        except ValueError:
            metrics["AUC"] = np.nan
    else:
        metrics["AUC"] = np.nan
    return metrics


def main():
    print("Loading data...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")

    print("Fitting preprocessor...")
    preproc = build_preprocessor(df)

    # Train / test split (stratified, 80/20) done on the RAW dataframe first,
    # so the raw test rows can be exported as test_data.csv for the Streamlit app.
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL],
    )

    X_train = transform(train_df, preproc)
    X_test = transform(test_df, preproc)
    y_train = preproc["target_encoder"].transform(train_df[TARGET_COL])
    y_test = preproc["target_encoder"].transform(test_df[TARGET_COL])
    n_classes = len(preproc["target_encoder"].classes_)

    models = get_models()
    results = []

    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test, n_classes)
        metrics["Model"] = name
        results.append(metrics)

        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, os.path.join(HERE, f"{fname}.pkl"))
        print(f"  -> saved model/{fname}.pkl | " + ", ".join(
            f"{k}={v:.4f}" for k, v in metrics.items() if k != "Model"
        ))

    # Save preprocessor
    joblib.dump(preproc, os.path.join(HERE, "preprocessor.pkl"))
    print("Saved model/preprocessor.pkl")

    # Save metrics comparison table
    results_df = pd.DataFrame(results)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    results_df = results_df.round(4)
    results_df.to_csv(os.path.join(HERE, "metrics_comparison.csv"), index=False)
    print("\n=== Metrics Comparison ===")
    print(results_df.to_string(index=False))

    # Save markdown version for easy pasting into README.md
    with open(os.path.join(HERE, "metrics_comparison.md"), "w") as f:
        f.write(results_df.to_markdown(index=False))
    print("Saved model/metrics_comparison.csv and model/metrics_comparison.md")

    # Export the raw (untransformed) test split as test_data.csv for the Streamlit app.
    # Includes the true label column so the app can compute evaluation metrics.
    test_df.to_csv(TEST_DATA_OUT, index=False)
    print(f"Saved held-out test split ({test_df.shape[0]} rows) -> {TEST_DATA_OUT}")

    # Save class list for reference
    with open(os.path.join(HERE, "classes.json"), "w") as f:
        json.dump(list(preproc["target_encoder"].classes_), f, indent=2)


if __name__ == "__main__":
    main()
