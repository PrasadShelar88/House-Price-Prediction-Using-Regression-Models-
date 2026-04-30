from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = DATA_DIR / "housing_data.csv"
MODEL_PATH = MODEL_DIR / "house_price_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

NUMERIC_FEATURES = ["area_sqft", "bedrooms", "bathrooms", "property_age", "parking", "floors", "distance_to_city_km"]
CATEGORICAL_FEATURES = ["location", "furnishing", "has_garden"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_data(rows: int = 1600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    locations = np.array(["Metro", "Urban", "Suburban", "Rural"])
    furnishings = np.array(["Unfurnished", "Semi-Furnished", "Fully-Furnished"])
    gardens = np.array(["Yes", "No"])

    df = pd.DataFrame({
        "area_sqft": rng.integers(500, 4500, rows),
        "bedrooms": rng.integers(1, 6, rows),
        "bathrooms": rng.integers(1, 5, rows),
        "location": rng.choice(locations, rows, p=[0.20, 0.40, 0.30, 0.10]),
        "property_age": rng.integers(0, 45, rows),
        "parking": rng.integers(0, 4, rows),
        "furnishing": rng.choice(furnishings, rows, p=[0.30, 0.45, 0.25]),
        "floors": rng.integers(1, 5, rows),
        "has_garden": rng.choice(gardens, rows, p=[0.28, 0.72]),
        "distance_to_city_km": np.round(rng.uniform(0.5, 35, rows), 2),
    })

    location_bonus = df["location"].map({"Metro": 4500000, "Urban": 2800000, "Suburban": 1500000, "Rural": 550000})
    furnishing_bonus = df["furnishing"].map({"Unfurnished": 0, "Semi-Furnished": 450000, "Fully-Furnished": 950000})
    garden_bonus = df["has_garden"].map({"Yes": 600000, "No": 0})
    noise = rng.normal(0, 450000, rows)

    price = (
        1200000 + df["area_sqft"] * 4200 + df["bedrooms"] * 180000 + df["bathrooms"] * 260000
        + df["parking"] * 300000 + df["floors"] * 160000 - df["property_age"] * 45000
        - df["distance_to_city_km"] * 85000 + location_bonus + furnishing_bonus + garden_bonus + noise
    )
    df["price_inr"] = np.maximum(price, 800000).round(0)
    return df


def load_or_create_dataset() -> pd.DataFrame:
    ensure_dirs()
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    df = generate_synthetic_data()
    df.to_csv(DATA_PATH, index=False)
    return df


def build_pipeline(model_type: str = "random_forest") -> Pipeline:
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", numeric_transformer, NUMERIC_FEATURES), ("cat", categorical_transformer, CATEGORICAL_FEATURES)])
    regressor = LinearRegression() if model_type == "linear" else RandomForestRegressor(n_estimators=250, random_state=42, max_depth=16, min_samples_leaf=2)
    return Pipeline([("preprocessor", preprocessor), ("model", regressor)])


def train_model() -> Dict[str, Any]:
    df = load_or_create_dataset()
    X = df[FEATURE_COLUMNS]
    y = df["price_inr"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    candidates = {"Linear Regression": build_pipeline("linear"), "Random Forest Regressor": build_pipeline("random_forest")}
    results: Dict[str, Any] = {}
    best_name, best_model, best_rmse = None, None, float("inf")
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        results[name] = {"mae": float(mean_absolute_error(y_test, preds)), "rmse": rmse, "r2": float(r2_score(y_test, preds))}
        if rmse < best_rmse:
            best_name, best_model, best_rmse = name, pipeline, rmse
    joblib.dump(best_model, MODEL_PATH)
    metrics = {"best_model": best_name, "rows": int(len(df)), "features": FEATURE_COLUMNS, "results": results}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def get_model() -> Pipeline:
    ensure_dirs()
    if not MODEL_PATH.exists():
        train_model()
    return joblib.load(MODEL_PATH)


def get_metrics() -> Dict[str, Any]:
    if not METRICS_PATH.exists():
        return train_model()
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def predict_price(payload: Dict[str, Any]) -> float:
    model = get_model()
    df = pd.DataFrame([payload])[FEATURE_COLUMNS]
    return float(model.predict(df)[0])


def feature_impact() -> list[dict[str, Any]]:
    model = get_model()
    regressor = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]
    if not hasattr(regressor, "feature_importances_"):
        return []
    names = preprocessor.get_feature_names_out()
    impacts = sorted(zip(names, regressor.feature_importances_), key=lambda x: x[1], reverse=True)[:10]
    return [{"feature": str(name).replace("num__", "").replace("cat__", ""), "impact": round(float(value), 4)} for name, value in impacts]
