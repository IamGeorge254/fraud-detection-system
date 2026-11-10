
from pathlib import Path
import joblib

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

# Initialize FastAPI
app = FastAPI(
    title="Fraud Detection API",
    description="An API for detecting fraudulent financial transactions",
    version="1.0.0"
)

# Load trained model artifact
model_path = Path("models/fraud_detection_model.joblib")
artifact = joblib.load(model_path)

model = artifact["model"]
dv = artifact["dv"]
threshold = artifact["threshold"]


# Define transaction input
class Transaction(BaseModel):
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    step: int
    amount: float


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running"
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