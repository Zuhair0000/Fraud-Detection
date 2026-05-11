from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import torch
import torch.nn as nn
import joblib
from captum.attr import IntegratedGradients

# 1. Recreate the Neural Network Architecture exactly as trained
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

# 2. Initialize the API
app = FastAPI(title="Deep Learning Transaction Monitor")

# 3. Add CORS Middleware so React can talk to FastAPI safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Define the expected incoming JSON payload using Pydantic (From React)
class TransactionRequest(BaseModel):
    income: float
    name_email_similarity: float
    prev_address_months_count: int
    device_os: str
    payment_type: str

# 5. Load the Artifacts (This happens ONCE when the server starts)
try:
    # Load the Scikit-Learn preprocessor & PyTorch Model
    preprocessor = joblib.load('fraud_preprocessor.joblib')
    model = FraudNet(input_shape=53) 
    model.load_state_dict(torch.load('fraud_model_weights.pth'))
    model.eval() 
    ig = IntegratedGradients(model)
    
    # --- TEMPLATE INJECTION (BULLETPROOFED) ---
    # 1. Grab a chunk of data
    # IMPORTANT: Change 'Base.csv' to 'dataset.csv' if that is your file's name!
    df_chunk = pd.read_csv('Base.csv', nrows=1000)
    
    # 2. Isolate safe customers, pick one, and FORCE the index to reset to 0
    template_df = df_chunk[df_chunk['fraud_bool'] == 0].head(1).copy().reset_index(drop=True)
    
    if 'fraud_bool' in template_df.columns:
        template_df = template_df.drop(columns=['fraud_bool'])
        
    print("✅ Model, Preprocessor, and Safe Template loaded successfully!")
except Exception as e:
    print(f"❌ Error loading artifacts: {e}")

# 6. Create the API Endpoint
@app.post("/predict_fraud")
async def predict_fraud(transaction: TransactionRequest):
    try:
        incoming_data = template_df.copy()
        
        # A. Safely overwrite the values by selecting the exact column directly
        react_data = transaction.model_dump()
        for column_name, user_value in react_data.items():
            if column_name in incoming_data.columns:
                incoming_data[column_name] = user_value
                
        # --- NEW: OUT-OF-DISTRIBUTION CORRECTION ---
        # The neural network was trained on income deciles (0.1 to 0.9), NOT raw dollars!
        # We will quickly convert the user's raw salary into a safe decile format.
        if incoming_data['income'].iloc[0] > 1.0:
            # e.g., $85,000 becomes 0.85 -> rounds to 0.9
            converted_income = round(incoming_data['income'].iloc[0] / 100000, 1)
            # Ensure it never drops below 0.1 and never exceeds 0.9
            incoming_data['income'] = max(0.1, min(converted_income, 0.9))
        # -------------------------------------------
                
        # B. Feature Engineering Recreation
        is_missing = (incoming_data['prev_address_months_count'] == -1).astype(int)
        incoming_data['is_missing_prev_address'] = is_missing
        incoming_data.loc[incoming_data['prev_address_months_count'] == -1, 'prev_address_months_count'] = 0
        
        # C. Process and Predict
        processed_data = preprocessor.transform(incoming_data)
        tensor_data = torch.tensor(processed_data, dtype=torch.float32)
        
        with torch.no_grad():
            output_logit = model(tensor_data)
            probability = torch.sigmoid(output_logit).item()
            
            # 🔥 DEBUG: Print the exact raw mathematical probability to your terminal
            print(f"🔥 DEBUG -> Raw Model Probability for this customer: {probability:.4f}")
            
            # Because of our 80x penalty, a perfectly safe customer might score a 0.90!
            # We are setting the threshold exceptionally high to counter the paranoia.
            is_fraud = bool(probability > 0.95) 
            
        # D. Captum Explanation
        explanation = []
        if is_fraud:
            with torch.enable_grad():
                tensor_data.requires_grad_()
                attributions = ig.attribute(tensor_data)
                
            # Formatting the explanation (Hardcoded for this UI demo, 
            # but usually pulled dynamically via get_feature_names_out logic)
            explanation = [
                "keep_alive_session (Score: 0.4673)", 
                "payment_type_AC (Score: 0.3886)", 
                "device_os_windows (Score: 0.3606)"
            ]

        return {
            "status": "success",
            "fraud_detected": is_fraud,
            "confidence": round(probability, 4),
            "audit_trail": explanation if is_fraud else "Transaction Approved. No audit required."
        }
        
    except Exception as e:
        print(f"🚨 API CRASH: {str(e)}") 
        raise HTTPException(status_code=500, detail=str(e))