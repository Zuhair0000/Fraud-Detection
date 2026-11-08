from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional
import pandas as pd
import torch
import torch.nn as nn
import joblib
from captum.attr import IntegratedGradients

# Decision threshold on the sigmoid probability (matches the notebook's evaluation).
FRAUD_THRESHOLD = 0.5
# Number of top contributing features returned in the audit trail.
TOP_N_REASONS = 3

class FraudNet(nn.Module):
    def __init__(self, input_shape):
        super(FraudNet, self).__init__()
        self.layer1 = nn.Linear(input_shape, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
        
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)
        
        self.layer3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.layer1(x)))
        x = self.dropout2(self.relu2(self.layer2(x)))
        x = self.layer3(x) 
        return x

app = FastAPI(title="Deep Learning Transaction Monitor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TransactionRequest(BaseModel):
    income: float
    name_email_similarity: float
    prev_address_months_count: int
    device_os: str
    payment_type: str
    # If set, predict on this full held-out row from demo_transactions.csv
    # instead of patching the 5 fields above onto the template row.
    transaction_id: Optional[int] = None

# The fields the dashboard displays and edits.
DASHBOARD_FIELDS = ["income", "name_email_similarity", "prev_address_months_count", "device_os", "payment_type"]
LABEL_NAMES = {1: "fraud", 0: "legit"}

try:
    preprocessor = joblib.load('fraud_preprocessor.joblib')
    # Computed once at startup: readable names aligned with the model's input columns.
    feature_names = [name.split("__")[-1] for name in preprocessor.get_feature_names_out()]
    model = FraudNet(input_shape=53)
    model.load_state_dict(torch.load('fraud_model_weights.pth'))
    model.eval() 
    ig = IntegratedGradients(model)
    

    df_chunk = pd.read_csv('Base.csv', nrows=1000)
    
    template_df = df_chunk[df_chunk['fraud_bool'] == 0].head(1).copy().reset_index(drop=True)
    
    if 'fraud_bool' in template_df.columns:
        template_df = template_df.drop(columns=['fraud_bool'])
        
    print("✅ Model, Preprocessor, and Safe Template loaded successfully!")
except Exception as e:
    print(f"❌ Error loading artifacts: {e}")

# Held-out test rows exported by fraud.ipynb. Loaded separately so the API still
# serves the template-based flow if the file hasn't been generated yet.
demo_df = None
try:
    demo_df = pd.read_csv('demo_transactions.csv').set_index('transaction_id', drop=False)
    print(f"✅ Loaded {len(demo_df)} demo transactions ({int(demo_df['fraud_bool'].sum())} fraud).")
except Exception as e:
    print(f"⚠️ demo_transactions.csv not loaded ({e}). Re-run fraud.ipynb to generate it.")


def require_demo_df():
    if demo_df is None:
        raise HTTPException(status_code=503, detail="demo_transactions.csv is not loaded. Re-run fraud.ipynb to generate it.")
    return demo_df


@app.get("/sample_transaction")
async def sample_transaction(label: Literal["fraud", "legit"]):
    df = require_demo_df()
    pool = df[df['fraud_bool'] == (1 if label == "fraud" else 0)]
    if pool.empty:
        raise HTTPException(status_code=404, detail=f"No {label} transactions in demo set.")
    row = pool.sample(n=1).iloc[0]

    fields = {name: row[name].item() if hasattr(row[name], "item") else row[name] for name in DASHBOARD_FIELDS}
    # The notebook stored missing addresses as 0 + a flag; show the raw -1 sentinel the dashboard uses.
    if int(row['is_missing_prev_address']) == 1:
        fields['prev_address_months_count'] = -1

    return {
        "transaction_id": int(row['transaction_id']),
        **fields,
        "true_label": LABEL_NAMES[int(row['fraud_bool'])],
    }


@app.post("/predict_fraud")
async def predict_fraud(transaction: TransactionRequest):
    true_label = None
    if transaction.transaction_id is not None:
        df = require_demo_df()
        if transaction.transaction_id not in df.index:
            raise HTTPException(status_code=404, detail=f"Unknown transaction_id {transaction.transaction_id}")
        real_row = df.loc[[transaction.transaction_id]].reset_index(drop=True)
        true_label = LABEL_NAMES[int(real_row['fraud_bool'].iloc[0])]

    try:
        if true_label is not None:
            # Full real row; the dashboard fields in the request are ignored.
            incoming_data = real_row.drop(columns=['transaction_id', 'fraud_bool'])
        else:
            incoming_data = template_df.copy()

            react_data = transaction.model_dump(exclude={'transaction_id'})
            for column_name, user_value in react_data.items():
                if column_name in incoming_data.columns:
                    incoming_data[column_name] = user_value

            if incoming_data['income'].iloc[0] > 1.0:
                converted_income = round(incoming_data['income'].iloc[0] / 100000, 1)
                incoming_data['income'] = max(0.1, min(converted_income, 0.9))

        is_missing = (incoming_data['prev_address_months_count'] == -1).astype(int)
        if 'is_missing_prev_address' in incoming_data.columns:
            # Demo rows were already cleaned by the notebook (-1 -> 0 + flag); keep their flag.
            is_missing = is_missing | incoming_data['is_missing_prev_address'].astype(int)
        incoming_data['is_missing_prev_address'] = is_missing
        incoming_data.loc[incoming_data['prev_address_months_count'] == -1, 'prev_address_months_count'] = 0

        processed_data = preprocessor.transform(incoming_data)
        tensor_data = torch.tensor(processed_data, dtype=torch.float32)
        
        with torch.no_grad():
            output_logit = model(tensor_data)
            probability = torch.sigmoid(output_logit).item()
            
            is_fraud = bool(probability > FRAUD_THRESHOLD)
            
        explanation = []
        if is_fraud:
            with torch.enable_grad():
                tensor_data.requires_grad_()
                attributions = ig.attribute(tensor_data)

            scores = attributions.squeeze(0).detach().cpu().numpy()
            positive = [(name, float(score)) for name, score in zip(feature_names, scores) if score > 0]
            positive.sort(key=lambda pair: pair[1], reverse=True)

            explanation = [
                f"{name} (Score: {score:.4f})"
                for name, score in positive[:TOP_N_REASONS]
            ]

        response = {
            "status": "success",
            "fraud_detected": is_fraud,
            "confidence": round(probability, 4),
            "audit_trail": explanation if is_fraud else "Transaction Approved. No audit required."
        }
        if true_label is not None:
            response["transaction_id"] = transaction.transaction_id
            response["true_label"] = true_label
        return response
        
    except Exception as e:
        print(f"🚨 API CRASH: {str(e)}") 
        raise HTTPException(status_code=500, detail=str(e))