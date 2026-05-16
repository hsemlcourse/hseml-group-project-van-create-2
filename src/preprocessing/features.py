"""
Feature engineering on top of cleaned data.
Input: data/processed/listings_clean.csv
Output: data/processed/listings_features.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_PATH = Path("data/processed/listings_clean.csv")
FEATURES_PATH = Path("data/processed/listings_features.csv")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Encode device type ────────────────────────────────────────────────────
    df["is_laptop"] = (df["device_type"] == "laptop").astype(int)

    # ── Condition → ordinal (smartphones only) ────────────────────────────────
    condition_map = {"excellent": 3, "good": 2, "fair": 1, "poor": 0}
    df["condition_enc"] = df["condition"].map(condition_map).fillna(-1).astype(int)

    # ── Age proxy (smartphones) ───────────────────────────────────────────────
    df["age_years"] = (df.get("days_used", pd.Series(np.nan, index=df.index)) / 365).round(1)

    # ── Phone camera ratio ────────────────────────────────────────────────────
    df["camera_ratio"] = (
        df.get("rear_camera_mp", pd.Series(np.nan, index=df.index)) /
        df.get("front_camera_mp", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    )

    # ── Log price (useful for modelling) ─────────────────────────────────────
    df["log_price"] = np.log1p(df["price"])

    # ── Brand: keep top-N, rest → "Other" ────────────────────────────────────
    top_brands = df["brand"].value_counts().nlargest(20).index
    df["brand_clean"] = df["brand"].where(df["brand"].isin(top_brands), other="Other")

    # ── Storage buckets ───────────────────────────────────────────────────────
    df["storage_bucket"] = pd.cut(
        df["storage_gb"],
        bins=[0, 32, 64, 128, 256, 512, np.inf],
        labels=["≤32", "64", "128", "256", "512", "512+"],
    ).astype(str)

    # ── RAM buckets ───────────────────────────────────────────────────────────
    df["ram_bucket"] = pd.cut(
        df["ram_gb"],
        bins=[0, 2, 4, 6, 8, 12, np.inf],
        labels=["≤2", "4", "6", "8", "12", "12+"],
    ).astype(str)

    return df


def main() -> None:
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN_PATH)
    df_feat = build_features(df)
    df_feat.to_csv(FEATURES_PATH, index=False)
    print(f"Features saved: {FEATURES_PATH}  shape={df_feat.shape}")
    print(f"Columns: {list(df_feat.columns)}")


if __name__ == "__main__":
    main()
