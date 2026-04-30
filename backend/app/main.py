from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .schemas import HouseInput, PredictionResponse
from .model import train_model, get_metrics, predict_price, load_or_create_dataset, feature_impact

app = FastAPI(title="House Price Prediction API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"message": "House Price Prediction API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/train")
def train():
    return train_model()

@app.get("/metrics")
def metrics():
    return get_metrics()

@app.get("/sample-data")
def sample_data(limit: int = 8):
    df = load_or_create_dataset().head(max(1, min(limit, 25)))
    return df.to_dict(orient="records")

@app.get("/feature-impact")
def impacts():
    return {"top_features": feature_impact()}

@app.post("/predict", response_model=PredictionResponse)
def predict(house: HouseInput):
    price = predict_price(house.model_dump())
    metrics_data = get_metrics()
    return {"predicted_price_inr": round(price, 2), "predicted_price_lakh": round(price / 100000, 2), "model_name": metrics_data.get("best_model", "Regression Model"), "confidence_note": "Prediction is based on a trained synthetic housing dataset for project demonstration."}
