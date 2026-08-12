# Obesity Level Classification — ML Assignment 2

**Course:** Machine Learning · M.Tech (AIML/DSE) · BITS Pilani WILP
**Live app:** `<PASTE YOUR STREAMLIT COMMUNITY CLOUD LINK HERE>`
**GitHub repo:** `<PASTE YOUR GITHUB REPO LINK HERE>`

---

## a. Problem Statement

Obesity is a growing public-health concern linked to eating habits, physical
activity, and lifestyle factors. This project builds and compares multiple
classification models to **predict a person's obesity level (7 categories,
from Insufficient Weight to Obesity Type III)** from their demographic data,
eating habits, and physical condition. The trained models are wrapped in an
interactive Streamlit web app so predictions and evaluation metrics can be
explored without writing any code.

## b. Dataset Description

- **Source:** UCI Machine Learning Repository — *Estimation of Obesity
  Levels Based on Eating Habits and Physical Condition* (data for
  individuals from Mexico, Peru, and Colombia).
- **Instances:** 2,111
- **Features:** 16 (8 numeric, 8 categorical) — e.g. `Age`, `Height`,
  `Weight`, `FCVC` (vegetable consumption frequency), `NCP` (number of main
  meals), `CH2O` (daily water intake), `FAF` (physical activity frequency),
  `TUE` (time using technology devices), `Gender`,
  `family_history_with_overweight`, `FAVC` (frequent high-caloric food),
  `CAEC` (eating between meals), `SMOKE`, `SCC` (calorie monitoring),
  `CALC` (alcohol consumption), `MTRANS` (transportation mode).
- **Target:** `NObeyesdad` — 7 classes: `Insufficient_Weight`,
  `Normal_Weight`, `Overweight_Level_I`, `Overweight_Level_II`,
  `Obesity_Type_I`, `Obesity_Type_II`, `Obesity_Type_III`.
- **Class balance:** Fairly balanced — each class has between 272 and 351
  records (~13–17% of the dataset each). No missing values.
- **Split used:** 80% train / 20% test, stratified by target class,
  `random_state=42` (423 rows held out as the test set — this is the
  `test_data.csv` shipped in this repo and used by the Streamlit app).

## c. GitHub Repository Link

`<PASTE YOUR GITHUB REPO LINK HERE>`

Repository contains: complete source code, `requirements.txt`, this
`README.md`, `test_data.csv`, and all trained model files under `model/`.

## d. Models Used

All 5 models were trained on the **same preprocessed dataset** (label
encoding for categorical features, standard scaling for numeric features)
and evaluated on the same 20% held-out test split.

### Comparison Table

| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|----------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression        | 0.8723   | 0.9873 | 0.8719    | 0.8723 | 0.8702 | 0.8515 |
| Decision Tree               | 0.9125   | 0.9491 | 0.9154    | 0.9125 | 0.9134 | 0.8980 |
| kNN                         | 0.8274   | 0.9670 | 0.8313    | 0.8274 | 0.8165 | 0.8018 |
| Naive Bayes                 | 0.5981   | 0.9000 | 0.6465    | 0.5981 | 0.5732 | 0.5435 |
| Random Forest (Ensemble)    | 0.9598   | 0.9971 | 0.9634    | 0.9598 | 0.9604 | 0.9535 |

*(AUC = weighted one-vs-rest ROC-AUC; Precision/Recall/F1 = weighted
averages across the 7 classes; MCC computed for the multi-class case.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline (87% accuracy) with the second-highest AUC (0.987), showing the classes are largely linearly separable once features are scaled. It slightly under-performs on the boundary classes (e.g. distinguishing adjacent overweight/obesity levels), where the true decision boundary is not linear. |
| Decision Tree | Performs well (91% accuracy) since it can capture the non-linear, threshold-like relationships between BMI-driving features (Height, Weight) and the target. However, its AUC (0.949) is noticeably lower than the ensemble and linear models, and it is the most prone to overfitting individual quirks of the training split among the top performers. |
| kNN | Middling performance (83% accuracy). Distance-based classification is sensitive to the mix of scaled numeric and label-encoded categorical features here — label encoding imposes an artificial ordinal distance on nominal categories (e.g. `MTRANS`), which distorts neighbor distances and hurts accuracy relative to the tree/ensemble models. |
| Naive Bayes | Clearly the weakest model (60% accuracy, MCC 0.54) despite a respectable AUC (0.90). Gaussian Naive Bayes assumes feature independence, but obesity-related features are highly correlated (e.g. Weight and Height jointly determine BMI, which directly drives the label) — violating this assumption costs it significant accuracy even though it still ranks classes reasonably by probability (hence the AUC/accuracy gap). |
| Random Forest (Ensemble) | Best model across every metric (96% accuracy, MCC 0.95, AUC 0.997). Averaging many decision trees reduces the overfitting/variance problem seen in the single Decision Tree while still capturing non-linear feature interactions, making it the clear winner for this dataset. |
| **Overall Winner for this dataset** | **Random Forest (Ensemble)** — highest accuracy, precision, recall, F1, MCC, and AUC of all 5 models. |

---

## Repository Structure

```
project-folder/
│-- app.py                     # Streamlit application
│-- requirements.txt
│-- README.md
│-- test_data.csv              # held-out test split (423 rows) used by the app
│-- data/
│   └── ObesityDataSet_raw_and_data_sinthetic.csv   # full raw dataset
│-- model/
│   ├── train_models.py        # trains all 5 models + saves artifacts
│   ├── preprocessor.pkl       # fitted encoders + scaler
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest_ensemble.pkl
│   ├── metrics_comparison.csv
│   └── classes.json
```

## How to Reproduce

```bash
pip install -r requirements.txt
python model/train_models.py     # retrains all models, regenerates test_data.csv & metrics
streamlit run app.py             # launches the app locally
```

## Streamlit App Features

- **Dataset upload (CSV):** Upload `test_data.csv` (or any CSV with the same
  16 feature columns, optionally including the `NObeyesdad` label column).
- **Model selection dropdown:** Choose any of the 5 trained models.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1, MCC computed
  live on the uploaded data (when true labels are present).
- **Confusion matrix & classification report:** Per-class breakdown of
  predictions vs. actual labels.
- **Model comparison table:** Side-by-side view of all 5 models' test-set
  performance.

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub.
3. Click **New App** → select this repository → branch `main` → file `app.py`.
4. Click **Deploy**.

## BITS Virtual Lab Screenshot

A screenshot of this assignment's execution on the BITS Virtual Lab is
included in the submitted PDF as required.
