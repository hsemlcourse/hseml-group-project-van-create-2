import re
from pathlib import Path

import numpy as np
import pandas as pd

PHONES_PATH = Path("data/raw/used_device_data.csv")
LAPTOPS_PATH = Path("data/raw/laptop_price.csv")
PROCESSED_PATH = Path("data/processed/listings_clean.csv")

EUR_TO_RUB = 100.0


def _denormalize_phone_price(norm_price: float) -> float:
    # normalized_used_price is ln-scaled; * 500 gives plausible RUB range
    return float(np.exp(norm_price) * 500)


def clean_phones(path: Path = PHONES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df = df.dropna(subset=["normalized_used_price"])
    df["price"] = df["normalized_used_price"].apply(_denormalize_phone_price)
    df = df.rename(columns={
        "device_brand": "brand",
        "screen_size": "screen_inch",
        "internal_memory": "storage_gb",
        "ram": "ram_gb",
        "battery": "battery_mah",
        "weight": "weight_g",
    })
    df["device_type"] = "smartphone"
    df["condition"] = pd.cut(
        df["days_used"],
        bins=[-1, 90, 365, 730, np.inf],
        labels=["excellent", "good", "fair", "poor"],
    ).astype(str)
    keep = [
        "device_type", "brand", "os", "screen_inch", "storage_gb", "ram_gb",
        "battery_mah", "weight_g", "rear_camera_mp", "front_camera_mp",
        "release_year", "days_used", "condition", "price",
    ]
    return df[keep].copy()


def _parse_ram_laptop(s: str) -> float | None:
    if not isinstance(s, str):
        return None
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else None


def _parse_storage_laptop(s: str) -> tuple[float | None, str]:
    if not isinstance(s, str):
        return None, "Unknown"
    total_gb = 0.0
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(TB|GB)", s, re.IGNORECASE):
        val = float(match.group(1))
        if match.group(2).upper() == "TB":
            val *= 1024
        total_gb += val
    storage_type = "SSD" if "SSD" in s.upper() or "Flash" in s else "HDD"
    return (total_gb if total_gb > 0 else None), storage_type


def _parse_weight_laptop(s: str) -> float | None:
    if not isinstance(s, str):
        return None
    m = re.search(r"(\d+[.,]\d+|\d+)", s)
    return float(m.group(1).replace(",", ".")) if m else None


def _parse_cpu_brand(s: str) -> str:
    if not isinstance(s, str):
        return "Other"
    s_lower = s.lower()
    if "intel" in s_lower:
        return "Intel"
    if "amd" in s_lower:
        return "AMD"
    return "Other"


def _parse_gpu_brand(s: str) -> str:
    if not isinstance(s, str):
        return "Other"
    s_lower = s.lower()
    if "nvidia" in s_lower:
        return "Nvidia"
    if "amd" in s_lower or "radeon" in s_lower:
        return "AMD"
    if "intel" in s_lower:
        return "Intel"
    return "Other"


def clean_laptops(path: Path = LAPTOPS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df = df.dropna(subset=["Price_euros"])
    df["price"] = df["Price_euros"] * EUR_TO_RUB
    df["ram_gb"] = df["Ram"].apply(_parse_ram_laptop)
    storage_parsed = df["Memory"].apply(_parse_storage_laptop)
    df["storage_gb"] = [x[0] for x in storage_parsed]
    df["storage_type"] = [x[1] for x in storage_parsed]
    df["weight_kg"] = df["Weight"].apply(_parse_weight_laptop)
    df["screen_inch"] = pd.to_numeric(df["Inches"], errors="coerce")
    df["cpu_brand"] = df["Cpu"].apply(_parse_cpu_brand)
    df["gpu_brand"] = df["Gpu"].apply(_parse_gpu_brand)
    df = df.rename(columns={"Company": "brand", "TypeName": "laptop_type", "OpSys": "os"})
    df["device_type"] = "laptop"
    keep = [
        "device_type", "brand", "os", "screen_inch", "storage_gb", "storage_type",
        "ram_gb", "weight_kg", "cpu_brand", "gpu_brand", "laptop_type", "price",
    ]
    return df[keep].copy()


def clean_all() -> pd.DataFrame:
    df = pd.concat([clean_phones(), clean_laptops()], ignore_index=True)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if (dropped := before - len(df)) > 0:
        print(f"Dropped {dropped} duplicate rows")
    lo, hi = df["price"].quantile(0.01), df["price"].quantile(0.99)
    return df[(df["price"] >= lo) & (df["price"] <= hi)].reset_index(drop=True)


def main() -> None:
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = clean_all()
    print(f"Clean shape: {df.shape}")
    print(df["device_type"].value_counts())
    print(df[["price"]].describe())
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved → {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
