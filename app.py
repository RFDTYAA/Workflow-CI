from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import dagshub
import os
from typing import Dict, Any

dagshub.init(
    repo_owner="RFDTYAA",
    repo_name="credit-scoring-mlops",
    mlflow=True
)

app = FastAPI(
    title="Credit Scoring API",
    description="API untuk prediksi credit scoring",
    version="1.0.0"
)

def get_latest_model():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Credit_Scoring_Advanced_RafiAditya")
    if experiment is None:
        return None
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=10
    )
    for run in runs:
        if "model" in run.data.artifacts:
            return mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")
    return None

model = get_latest_model()
if model:
    print("✅ Model berhasil dimuat (latest dari DagsHub)")
else:
    print("⚠️ Model belum ditemukan, cek DagsHub Experiments")

class CreditData(BaseModel):
    checking_status: float
    duration: float
    credit_history: float
    purpose: float
    credit_amount: float
    savings_status: float
    employment: float
    installment_commitment: float
    personal_status: float
    other_parties: float
    residence_since: float
    property_magnitude: float
    age: float
    other_payment_plans: float
    housing: float
    existing_credits: float
    job: float
    num_dependents: float
    own_telephone: float
    foreign_worker: float

@app.get("/")
def root():
    return {"message": "Credit Scoring API running", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post(
    "/predict",
    responses={
        200: {
            "description": "Prediksi berhasil",
            "content": {
                "application/json": {
                    "example": {"prediction": 1, "label": "good"}
                }
            }
        },
        503: {
            "description": "Model belum dimuat / tidak tersedia"
        },
        400: {
            "description": "Input tidak valid atau terjadi error saat prediksi"
        }
    }
)
def predict(data: CreditData):
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model belum siap"
        )
    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)

        return {
            "prediction": int(prediction[0]),
            "label": "good" if prediction[0] == 1 else "bad"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))