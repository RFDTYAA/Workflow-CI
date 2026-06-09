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
    description="API untuk prediksi credit scoring menggunakan model yang dilatih di DagsHub MLflow",
    version="1.0.0"
)

MODEL_URI = os.getenv(
    "MODEL_URI", 
    "runs:/691f9216291f4642ada774c006d7642a/model"
)

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print(f"✅ Model berhasil dimuat dari: {MODEL_URI}")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    model = None


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
    return {
        "message": "Credit Scoring API is running",
        "model_uri": MODEL_URI
    }


@app.post(
    "/predict",
    responses={
        200: {"description": "Prediction successful"},
        400: {"description": "Invalid input data"},
        500: {"description": "Model failed to load or internal server error"}
    }
)
def predict(data: CreditData) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model belum berhasil dimuat"
        )

    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)

        return {
            "prediction": int(prediction[0]),
            "message": "good" if prediction[0] == 1 else "bad"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))