# 🛡️ Fraud Detection API with Explainable AI

An end-to-end fraud detection system: a PyTorch neural network trained on highly imbalanced bank-account application data, served through a FastAPI endpoint, with a React dashboard that shows **why** each transaction was flagged using Captum Integrated Gradients.

<img width="1058" height="884" alt="Screenshot 2026-11-08 at 8 47 53 PM" src="https://github.com/user-attachments/assets/d5a84475-8f76-4c38-89d9-b64e1b409d28" />

## The problem
Only **1.15%** of applications in the dataset are fraud. A model can reach 99% accuracy by never predicting fraud, which is useless. This project optimises for **catching fraud (recall)** instead, and explains every flag so an analyst can review the decision.

## How it works
1. **Data preparation (Polars):** scanned the dataset lazily, found 20,594 hidden missing values coded as `-1` in `prev_address_months_count`, and added an `is_missing_prev_address` flag instead of losing that signal.
2. **Preprocessing (scikit-learn):** median imputation and scaling for numeric features, one-hot encoding for categorical features, combined in a `ColumnTransformer` fitted on training data only to avoid leakage.
3. **Baseline:** Random Forest with balanced class weights.
4. **Model (PyTorch):** a 3-layer network (64 → 32 → 1) with ReLU and dropout, trained with `BCEWithLogitsLoss` and a positive-class weight of 👉 **XX.X** (the ratio of legitimate to fraud cases), so a missed fraud costs far more than a false alarm.
5. **Explainability (Captum):** Integrated Gradients attributes each flagged prediction to its input features. The API returns the **top 3 reasons for that specific transaction**.
6. **Serving (FastAPI):** a `/predict_fraud` endpoint scores a transaction, and `/sample_transaction` serves real transactions from the held-out test set.
7. **Dashboard (React + Tailwind):** load a real fraud or legitimate case from the test set, see the model's decision next to the true label, and read the reasons behind each flag.

## Results (held-out test set: 2,000 transactions, 21 fraud)

| Metric (fraud class) | Value |
|---|---|
| Recall | **62%** (13 of 21 fraud cases caught) |
| Precision | 4% (333 false alarms) |
| Decision threshold | 0.5 |

**The trade-off:** the model is tuned to miss as little fraud as possible, at the cost of many false alarms that an analyst would review. In a real system, the threshold would be chosen with the business, based on the cost of a missed fraud compared with the cost of a manual review.

<img width="860" height="797" alt="Screenshot 2026-11-08 at 8 34 07 PM" src="https://github.com/user-attachments/assets/a7750f40-bf6f-4f92-bd0a-fcfa5fce569d" />

## Tech stack
**ML:** Python · Polars · pandas · scikit-learn · PyTorch · Captum
**Backend:** FastAPI · Pydantic · Uvicorn
**Frontend:** React · Tailwind CSS · Vite

## Project structure
```
Fraud-Detection/
├── fraud.ipynb              # data preparation, training, evaluation, explainability
├── app.py                   # FastAPI service
├── requirements.txt         # pinned Python dependencies
├── demo_transactions.csv    # held-out test set used by the dashboard
└── fraud-dashboard/         # React + Vite frontend
```

## Run it locally
1. Download `Base.csv` from the 👉 [Bank Account Fraud dataset (NeurIPS 2022) on Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) and place it in the project root.
2. Create the environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run `fraud.ipynb` to create `fraud_preprocessor.joblib`, `fraud_model_weights.pth` and `demo_transactions.csv`.
4. Start the API:
   ```bash
   uvicorn app:app --reload
   ```
5. Start the dashboard in a second terminal:
   ```bash
   cd fraud-dashboard
   npm install
   npm run dev
   ```
6. Open http://localhost:5173 and click **Load real fraud case**.

## Limitations and next steps
- Trained on a 10,000-row sample for speed; training on the full dataset should improve precision.
- Tune the decision threshold on a separate validation set rather than using a fixed 0.5.
- Compare against gradient boosting models (XGBoost, LightGBM), which are strong on tabular data.
- Add automated tests for the API.<img width="1058" height="884" alt="Screenshot 2026-11-08 at 8 47 53 PM" src="https://github.com/user-attachments/assets/174bb1ae-dd53-4199-a679-3e2e8749b551" />
