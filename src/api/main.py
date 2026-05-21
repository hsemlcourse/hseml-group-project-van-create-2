from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import DeviceFeatures, PredictionResponse
from src.models.train import build_feature_matrix

MODEL_PATH = Path("models/final_model.joblib")

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run training first.")
    _model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="Device Price Predictor", lifespan=lifespan)


def _row_to_df(features: DeviceFeatures) -> pd.DataFrame:
    data = features.model_dump()
    df = pd.DataFrame([data])
    return df


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    CONDITION_MAP = {"excellent": 3, "good": 2, "fair": 1, "poor": 0}
    df = df.copy()
    df["is_laptop"] = (df["device_type"] == "laptop").astype(int)
    df["condition_enc"] = df.get("condition", pd.Series([None])).map(CONDITION_MAP).fillna(-1).astype(int)
    df["age_years"] = (df.get("days_used", pd.Series([np.nan])) / 365).round(1)
    rear = pd.to_numeric(df.get("rear_camera_mp", pd.Series([np.nan])), errors="coerce")
    front = pd.to_numeric(df.get("front_camera_mp", pd.Series([np.nan])), errors="coerce").replace(0, np.nan)
    df["camera_ratio"] = rear / front
    df["log_price"] = np.nan

    top_brands = [
        "Samsung", "Apple", "Huawei", "Xiaomi", "OnePlus", "Oppo", "Vivo", "Realme",
        "Nokia", "Sony", "Lenovo", "LG", "Motorola", "Dell", "HP", "Asus",
        "Acer", "MSI", "Toshiba", "Razer",
    ]
    df["brand_clean"] = df["brand"].where(df["brand"].isin(top_brands), other="Other")

    df["storage_bucket"] = pd.cut(
        pd.to_numeric(df.get("storage_gb", pd.Series([np.nan])), errors="coerce"),
        bins=[0, 32, 64, 128, 256, 512, np.inf],
        labels=["≤32", "64", "128", "256", "512", "512+"],
    ).astype(str)

    df["ram_bucket"] = pd.cut(
        pd.to_numeric(df.get("ram_gb", pd.Series([np.nan])), errors="coerce"),
        bins=[0, 2, 4, 6, 8, 12, np.inf],
        labels=["≤2", "4", "6", "8", "12", "12+"],
    ).astype(str)

    return df


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: DeviceFeatures):
    df = _row_to_df(features)
    df = _preprocess(df)

    try:
        X, _, _, _ = build_feature_matrix(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    price = float(_model.predict(X)[0])
    price = max(price, 0)

    return PredictionResponse(
        price_rub=round(price, 2),
        price_rub_formatted=f"{price:,.0f} руб.",
    )
