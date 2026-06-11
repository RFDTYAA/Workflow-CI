from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import joblib
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time

app = FastAPI(title="Credit Scoring Inference API with Monitoring")

MODEL_URI = "runs:/cd72eed415334815a69e7109713f0f36/model"

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    preprocessor = joblib.load("preprocessor.pkl")
    MODEL_LOADED = True
    print("✅ Model dan Preprocessor berhasil dimuat")
except Exception as e:
    model = None
    preprocessor = None
    MODEL_LOADED = False
    print(f"❌ Gagal memuat model/preprocessor: {e}")

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "Request latency", ["endpoint"])
PREDICTION_COUNT = Counter("prediction_total", "Total predictions", ["result"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
MODEL_PREDICTION_GOOD = Counter("model_predictions_good_total", "Total good predictions")
MODEL_PREDICTION_BAD = Counter("model_predictions_bad_total", "Total bad predictions")
HTTP_ERRORS = Counter("http_errors_total", "Total HTTP errors", ["status_code"])
ACTIVE_REQUESTS = Gauge("active_requests", "Currently active requests")
MODEL_STATUS = Gauge("model_load_status", "Model load status (1=loaded, 0=not loaded)")

MODEL_STATUS.set(1 if MODEL_LOADED else 0)

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

@app.middleware("http")
async def add_prometheus_middleware(request, call_next):
    start_time = time.time()
    ACTIVE_REQUESTS.inc()
    response = await call_next(request)
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(time.time() - start_time)
    ACTIVE_REQUESTS.dec()
    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
def root():
    return {"message": "Credit Scoring API with Monitoring is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL_LOADED}

@app.post("/predict")
def predict(data: CreditData):
    if not MODEL_LOADED or model is None or preprocessor is None:
        HTTP_ERRORS.labels(status_code=503).inc()
        raise HTTPException(status_code=503, detail="Model atau Preprocessor belum siap")

    start_time = time.time()
    try:
        # Buat DataFrame dari input mentah
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])

        # Lakukan preprocessing
        input_processed = preprocessor.transform(input_df)

        # Prediksi
        prediction = model.predict(input_processed)
        result = int(prediction[0])
        label = "good" if result == 1 else "bad"

        PREDICTION_COUNT.labels(result=label).inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)

        if result == 1:
            MODEL_PREDICTION_GOOD.inc()
        else:
            MODEL_PREDICTION_BAD.inc()

        return {
            "prediction": result,
            "label": label
        }
    except Exception as e:
        HTTP_ERRORS.labels(status_code=400).inc()
        raise HTTPException(status_code=400, detail=str(e))