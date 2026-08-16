# Credit Card Fraud Detection — Cost-Sensitive ML

A machine learning project that goes beyond standard fraud classification by weighing model choice against **real financial cost**, not just accuracy metrics.

## Problem

Credit card fraud is rare — only 0.17% of transactions in this dataset (492 out of 284,807) are fraudulent. A model that predicts "normal" every time would still be 99.8% "accurate" while catching zero fraud. The real challenge is building a model that performs well on metrics that actually matter for a heavily imbalanced problem, then choosing the one that saves the most money in practice.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (mlg-ulb, via Kaggle) — anonymized European cardholder transactions from September 2013. Features `V1`–`V28` are PCA-transformed for privacy; `Time` and `Amount` are the only unmodified columns.

> The dataset (`creditcard.csv`, ~144MB) is not included in this repo due to GitHub's file size limits. Download it directly from Kaggle and place it in the project root to run the notebook.

## Approach

1. **EDA** — confirmed the 0.17% class imbalance and examined transaction amount patterns
2. **Split** — stratified 80/20 train-test split, preserving the fraud ratio in both sets
3. **Balance** — SMOTE applied to training data only, never to the test set
4. **Scale** — StandardScaler applied to features (fit on train, transformed on test)
5. **Train & compare** — 8 classification models: Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN, Naive Bayes, ANN (MLPClassifier), SVM
6. **Tune** — hyperparameter search (RandomizedSearchCV) on all 8 models, with SMOTE applied *inside* each cross-validation fold via an imbalanced-learn pipeline, to avoid data leakage from synthetic oversampling
7. **Evaluate on cost, not just F1** — converted false negatives and false positives into real ₹ cost, since missing a large fraud case is far more expensive than a false alarm
8. **Explain** — SHAP values to show which features drive each individual prediction
9. **Deploy** — a Streamlit app for live, interactive testing

## Results

| Model | Precision | Recall | F1 |
|---|---|---|---|
| **XGBoost** | 0.81 | 0.87 | **0.84** |
| Random Forest | 0.82 | 0.83 | 0.82 |
| ANN | 0.73 | 0.80 | 0.77 |
| KNN | 0.47 | 0.84 | 0.66 |
| Decision Tree | 0.42 | 0.80 | 0.57 |
| Naive Bayes | 0.06 | 0.88 | 0.54 |
| SVM | 0.25 | 0.89 | 0.45 |
| Logistic Regression | 0.13 | 0.90 | 0.25 |

### The cost-sensitive verdict

XGBoost and Random Forest were close on F1 (0.84 vs 0.82). But scoring each model's actual missed-fraud amounts told a different story:

| Model | Estimated total cost |
|---|---|
| Random Forest | ₹4,262.93 |
| **XGBoost** | **₹2,027.93** |

XGBoost's missed fraud cases were consistently the cheaper ones — Random Forest let a ₹1,809.68 fraud case through undetected. **XGBoost reduces estimated financial loss by ~48% compared to Random Forest**, a distinction standard classification metrics alone would have missed.

## Explainability

SHAP (TreeExplainer) was used to identify which features drive XGBoost's predictions, both globally (which features matter most across all transactions) and per-transaction (why *this specific* transaction was flagged).

## Demo app

A Streamlit app lets you test the model on real transactions from the dataset (or enter values manually) and see the prediction alongside a SHAP explanation.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack

Python · pandas · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · SHAP · Streamlit · matplotlib

## Project structure

```
├── Fraud_Detection.ipynb   # Full analysis: EDA, modeling, tuning, cost analysis, SHAP
├── app.py                  # Streamlit demo app
├── fraud_xgb_model.pkl     # Trained final XGBoost model
├── fraud_scaler.pkl        # Fitted StandardScaler
└── README.md
```

## Author

Hebah — [GitHub](https://github.com/workwithhebah-bit) · [LinkedIn](https://linkedin.com/in/hebah-/) · [Kaggle](https://kaggle.com/hebah2003)
