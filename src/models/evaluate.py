"""Metrics and train/val/test split utilities."""

import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 4)}


def print_metrics(name: str, metrics: dict) -> None:
    print(f"{name:35s} | MAE={metrics['mae']:>10.2f} | RMSE={metrics['rmse']:>10.2f} | MAPE={metrics['mape']:>7.2f}%")


def split_data(X, y, val_size: float = 0.15, test_size: float = 0.15, random_state: int = RANDOM_SEED):
    """Train / val / test split. Stratification not used (regression task)."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=val_size + test_size, random_state=random_state
    )
    relative_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=relative_test, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
