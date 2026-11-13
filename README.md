# Fraud Detection System

An end-to-end machine learning project that detects potentially fraudulent financial transactions using **XGBoost**. The model is served through a **FastAPI REST API**, containerized with Docker, and deployed publicly on Render.

## Live Demo

- **Live API:** https://fraud-detection-system-g2co.onrender.com
- **API Documentation (Swagger UI):** https://fraud-detection-system-g2co.onrender.com/docs
- **GitHub Repository:** https://github.com/IamGeorge254/fraud-detection-system

> **Note:** The application is hosted on Render's free tier. The service may spin down after inactivity, so the first request can take 50 seconds or more to respond.

## Project Overview

Financial fraud is a major challenge for digital payment platforms. This project applies machine learning to identify potentially fraudulent transactions based on transaction characteristics.

The system uses an XGBoost classification model to estimate the probability that a transaction is fraudulent. A classification threshold of **0.8** is applied to determine the final prediction.

The trained model and preprocessing artifacts are saved and loaded by the API, allowing users to submit transaction details and receive predictions.

## Features

- Fraud detection using an XGBoost classification model.
- Transaction preprocessing using a saved DictVectorizer.
- Probability-based fraud predictions.
- Configurable classification threshold (saved threshold: 0.8).
- REST API built with FastAPI.
- Interactive API documentation using Swagger UI.
- Docker containerization.
- Cloud deployment using Render.

## Technologies Used

| Category | Tools |
|---|---|
| Programming | Python |
| Data analysis | Pandas, NumPy |
| Machine learning | XGBoost, Scikit-learn |
| API development | FastAPI, Pydantic |
| Model persistence | Joblib |
| Environment management | uv |
| Containerization | Docker |
| Deployment | Render |
| Version control | Git, GitHub |

## Model Input Features

The API accepts the following transaction details:

| Feature | Description |
|---|---|
| `type` | Transaction type |
| `step` | Time step of the transaction |
| `amount` | Transaction amount |

Supported transaction types:

- CASH_IN
- CASH_OUT
- DEBIT
- PAYMENT
- TRANSFER

## Testing the Live API

Open the interactive documentation:

https://fraud-detection-system-g2co.onrender.com/docs

1. Expand `POST /predict`.
2. Click **Try it out**.
3. Submit the following sample transaction:

```json
{
  "type": "CASH_OUT",
  "step": 1,
  "amount": 181
}
```

4. Click **Execute** to receive the prediction.

### Example API Response

```json
{
  "fraud_probability": 0.95,
  "threshold": 0.8,
  "prediction": "Fraud"
}
```

*The probability above is illustrative. Actual results depend on the trained model's prediction.*

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/IamGeorge254/fraud-detection-system.git
cd fraud-detection-system
```

### 2. Install dependencies

Ensure Python and uv are installed, then run:

```bash
uv sync
```

### 3. Start the API

```bash
uv run uvicorn main:app --app-dir src/fraud_detection --host 0.0.0.0 --port 8000
```

### 4. Open the documentation

Visit:

http://127.0.0.1:8000/docs

You can now test the prediction endpoint locally.

## Running with Docker

Build the Docker image:

```bash
docker build -t fraud-detection .
```

Start the container:

```bash
docker run -d -p 8000:10000 --name fraud-api fraud-detection
```

Open:

http://127.0.0.1:8000/docs

## Project Structure

```text
fraud-detection-system/
│
├── models/
│   ├── fraud_detection_model.joblib
│   ├── preprocessing.joblib
│   └── xgboost_model.json
│
├── script/
│   ├── Fraud_detection.ipynb
│   └── data/
│
├── src/
│   └── fraud_detection/
│       ├── __init__.py
│       ├── main.py
│       └── train.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Deployment

The application is deployed as a Docker-based web service on Render.

The Docker container runs the FastAPI application and loads the saved XGBoost model and preprocessing artifacts.

**Deployment URL:** https://fraud-detection-system-g2co.onrender.com

## Disclaimer

This project is intended for educational and demonstration purposes. Predictions should not be used as the sole basis for real-world financial decisions without further validation, monitoring and appropriate production safeguards.

## Author

**George Analytics**

GitHub: https://github.com/IamGeorge254
