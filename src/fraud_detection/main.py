from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from xgboost import XGBClassifier

# Initialize FastAPI
app = FastAPI(
    title="Fraud Detection API",
    description="An API for detecting fraudulent financial transactions using XGBoost",
    version="1.0.0"
)

# Load trained model artifact
PROJECT_ROOT = Path(__file__).resolve().parents[2]
models_dir = PROJECT_ROOT / "models"

# Load XGBoost model
model = XGBClassifier()
model.load_model(models_dir / "xgboost_model.json")

# Load preprocessing objects
artifact = joblib.load(models_dir / "preprocessing.joblib")

dv = artifact["dv"]
threshold = artifact["threshold"]

# Define transaction input
class Transaction(BaseModel):
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    step: int = Field(gt=0)
    amount: float = Field(ge=0)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "model": "XGBoost",
        "threshold": threshold
    }


# Fraud prediction endpoint
@app.post("/predict")
def predict(transaction: Transaction):

    # Convert transaction into dictionary
    transaction_dict = transaction.model_dump()

    # Transform input using fitted vectorizer
    X = dv.transform([transaction_dict])

    # Predict fraud probability
    fraud_probability = float(model.predict_proba(X)[0, 1])

    # Apply selected threshold
    prediction = (
        "Fraud"
        if fraud_probability >= threshold
        else "Legitimate"
    )

    return {
        "fraud_probability": fraud_probability,
        "threshold": threshold,
        "prediction": prediction
    }
