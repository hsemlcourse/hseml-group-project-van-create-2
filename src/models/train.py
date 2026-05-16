"""
Feature matrix builder and final model training script.
Run: python3 -m src.models.train
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURES_PATH = Path("data/processed/listings_features.csv")
MODEL_PATH = Path("models/final_model.joblib")

NUM_COLS = [
    "screen_inch", "storage_gb", "ram_gb", "is_laptop",
    "condition_enc", "age_years", "camera_ratio",
    "battery_mah", "weight_g", "weight_kg", "release_year",
    "rear_camera_mp", "front_camera_mp",
]
CAT_COLS = ["brand_clean", "os", "cpu_brand", "gpu_brand", "storage_type",
            "storage_bucket", "ram_bucket", "laptop_type"]


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[int], list[str]]:
    """Return (X, y, cat_col_indices, all_col_names)."""
    available_num = [c for c in NUM_COLS if c in df.columns]
    available_cat = [c for c in CAT_COLS if c in df.columns]
    all_cols = available_num + available_cat

    X = df[all_cols].copy()
    X[available_num] = X[available_num].apply(pd.to_numeric, errors="coerce")
    for col in available_cat:
        X[col] = X[col].fillna("unknown").astype("category")

    y = df["price"].values
    cat_indices = list(range(len(available_num), len(all_cols)))
    return X, y, cat_indices, all_cols


def save_model(model, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


if __name__ == "__main__":
    import lightgbm as lgb

    from src.models.evaluate import print_metrics, regression_metrics, split_data

    df = pd.read_csv(FEATURES_PATH)
    X, y, cat_indices, col_names = build_feature_matrix(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, random_state=RANDOM_SEED)

    model = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, random_state=RANDOM_SEED, n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    print_metrics("LightGBM val", regression_metrics(y_val, model.predict(X_val)))
    print_metrics("LightGBM test", regression_metrics(y_test, model.predict(X_test)))
    save_model(model)
