"""
Unit tests for preprocessing and model utilities.
Run: pytest tests/
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.clean import (
    _denormalize_phone_price,
    _parse_ram_laptop,
    _parse_storage_laptop,
    _parse_weight_laptop,
    _parse_cpu_brand,
    _parse_gpu_brand,
    clean_phones,
    clean_laptops,
    clean_all,
)
from src.preprocessing.features import build_features
from src.models.evaluate import regression_metrics, split_data
from src.models.train import build_feature_matrix


# ── clean.py ──────────────────────────────────────────────────────────────────

class TestPhoneParsers:
    def test_denormalize_positive(self):
        result = _denormalize_phone_price(4.0)
        assert result > 0

    def test_denormalize_monotone(self):
        assert _denormalize_phone_price(3.0) < _denormalize_phone_price(5.0)

    def test_parse_ram_normal(self):
        assert _parse_ram_laptop("8GB") == 8.0

    def test_parse_ram_none(self):
        assert _parse_ram_laptop(None) is None

    def test_parse_ram_no_digits(self):
        assert _parse_ram_laptop("Unknown") is None

    def test_parse_storage_ssd(self):
        gb, stype = _parse_storage_laptop("256GB SSD")
        assert gb == 256.0
        assert stype == "SSD"

    def test_parse_storage_tb(self):
        gb, stype = _parse_storage_laptop("1TB HDD")
        assert gb == 1024.0
        assert stype == "HDD"

    def test_parse_storage_combo(self):
        gb, _ = _parse_storage_laptop("128GB SSD + 1TB HDD")
        assert gb == 128.0 + 1024.0

    def test_parse_storage_none(self):
        gb, _ = _parse_storage_laptop(None)
        assert gb is None

    def test_parse_weight(self):
        assert _parse_weight_laptop("1.37kg") == pytest.approx(1.37)

    def test_parse_weight_none(self):
        assert _parse_weight_laptop(None) is None

    def test_cpu_brand_intel(self):
        assert _parse_cpu_brand("Intel Core i5 2.3GHz") == "Intel"

    def test_cpu_brand_amd(self):
        assert _parse_cpu_brand("AMD Ryzen 5") == "AMD"

    def test_cpu_brand_other(self):
        assert _parse_cpu_brand("Apple M1") == "Other"

    def test_gpu_brand_nvidia(self):
        assert _parse_gpu_brand("Nvidia GeForce GTX 1050") == "Nvidia"

    def test_gpu_brand_intel(self):
        assert _parse_gpu_brand("Intel HD Graphics 620") == "Intel"


class TestCleanPipeline:
    def test_clean_phones_shape(self):
        df = clean_phones()
        assert len(df) > 0
        assert "price" in df.columns
        assert "device_type" in df.columns

    def test_clean_phones_price_positive(self):
        df = clean_phones()
        assert (df["price"] > 0).all()

    def test_clean_phones_device_type(self):
        df = clean_phones()
        assert (df["device_type"] == "smartphone").all()

    def test_clean_laptops_shape(self):
        df = clean_laptops()
        assert len(df) > 0
        assert "price" in df.columns

    def test_clean_laptops_price_positive(self):
        df = clean_laptops()
        assert (df["price"] > 0).all()

    def test_clean_all_both_types(self):
        df = clean_all()
        types = df["device_type"].unique()
        assert "smartphone" in types
        assert "laptop" in types

    def test_clean_all_no_extreme_outliers(self):
        df = clean_all()
        assert (df["price"] > 0).all()


# ── features.py ───────────────────────────────────────────────────────────────

class TestFeatures:
    @pytest.fixture
    def clean_df(self):
        return clean_all()

    def test_build_features_adds_columns(self, clean_df):
        df = build_features(clean_df)
        for col in ["is_laptop", "log_price", "brand_clean", "storage_bucket", "ram_bucket"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_is_laptop_binary(self, clean_df):
        df = build_features(clean_df)
        assert set(df["is_laptop"].unique()).issubset({0, 1})

    def test_log_price_positive(self, clean_df):
        df = build_features(clean_df)
        assert (df["log_price"] > 0).all()

    def test_brand_clean_limits_categories(self, clean_df):
        df = build_features(clean_df)
        n_brands = df["brand_clean"].nunique()
        assert n_brands <= 21  # top 20 + "Other"


# ── evaluate.py ───────────────────────────────────────────────────────────────

class TestMetrics:
    def test_perfect_prediction(self):
        y = np.array([100.0, 200.0, 300.0])
        m = regression_metrics(y, y)
        assert m["mae"] == pytest.approx(0.0)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["mape"] == pytest.approx(0.0)

    def test_metrics_positive(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 320.0])
        m = regression_metrics(y_true, y_pred)
        assert m["mae"] > 0
        assert m["rmse"] > 0
        assert m["mape"] > 0

    def test_split_sizes(self):
        X = pd.DataFrame({"a": range(1000)})
        y = np.arange(1000)
        X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X, y)
        assert len(X_tr) + len(X_v) + len(X_te) == 1000
        assert 100 < len(X_v) < 250
        assert 100 < len(X_te) < 250

    def test_split_no_overlap(self):
        X = pd.DataFrame({"a": range(1000)})
        y = np.arange(1000, dtype=float)
        X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X, y)
        tr_set = set(y_tr)
        assert len(tr_set & set(y_v)) == 0
        assert len(tr_set & set(y_te)) == 0


# ── train.py ──────────────────────────────────────────────────────────────────

class TestBuildFeatureMatrix:
    @pytest.fixture
    def features_df(self):
        df = clean_all()
        return build_features(df)

    def test_returns_correct_shapes(self, features_df):
        X, y, cat_idx, cols = build_feature_matrix(features_df)
        assert len(X) == len(y)
        assert len(cols) == X.shape[1]

    def test_y_all_positive(self, features_df):
        _, y, _, _ = build_feature_matrix(features_df)
        assert (y > 0).all()

    def test_cat_indices_in_range(self, features_df):
        X, _, cat_idx, cols = build_feature_matrix(features_df)
        for i in cat_idx:
            assert 0 <= i < len(cols)
