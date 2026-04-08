"""
Tests for the Kaggle data downloader and disaster training data generator.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSyntheticFallback:
    """Test synthetic disaster data generation (no Kaggle auth needed)."""

    def test_build_synthetic_fallback_shape(self):
        from data.kaggle_downloader import _build_synthetic_fallback
        df = _build_synthetic_fallback()
        assert not df.empty
        assert len(df) >= 9000, f"Expected at least 9000 rows, got {len(df)}"

    def test_event_types_present(self):
        from data.kaggle_downloader import _build_synthetic_fallback
        df = _build_synthetic_fallback()
        event_types = set(df["event_type"].unique())
        assert "flood" in event_types
        assert "drought" in event_types
        assert "wildfire" in event_types

    def test_has_crop_impact_columns(self):
        from data.kaggle_downloader import _build_synthetic_fallback
        df = _build_synthetic_fallback()
        assert "crop_type" in df.columns
        assert "crop_impact" in df.columns
        assert "crop_yield_loss_pct" in df.columns

    def test_crop_impact_values_valid(self):
        from data.kaggle_downloader import _build_synthetic_fallback
        df = _build_synthetic_fallback()
        valid_impacts = {"high", "medium", "low"}
        actual = set(df["crop_impact"].unique())
        assert actual.issubset(valid_impacts), f"Invalid crop impacts: {actual - valid_impacts}"

    def test_global_coverage(self):
        from data.kaggle_downloader import _build_synthetic_fallback
        df = _build_synthetic_fallback()
        regions = set(df["region"].unique())
        # Should have India, Africa, Brazil, SE Asia
        region_str = " ".join(regions)
        assert "India" in region_str, "Missing India coverage"
        assert "Africa" in region_str, "Missing Africa coverage"
        assert "Brazil" in region_str, "Missing Brazil coverage"
        assert "Indonesia" in region_str, "Missing Indonesia coverage"
        assert "USA" in region_str or "California" in region_str, "Missing USA coverage"

    def test_yield_loss_range(self):
        from data.kaggle_downloader import _build_synthetic_fallback
        df = _build_synthetic_fallback()
        assert df["crop_yield_loss_pct"].min() >= 0
        assert df["crop_yield_loss_pct"].max() <= 100

    def test_latitude_longitude_valid(self):
        from data.kaggle_downloader import _build_synthetic_fallback
        df = _build_synthetic_fallback()
        assert df["latitude"].min() > -90
        assert df["latitude"].max() < 90
        assert df["longitude"].min() > -180
        assert df["longitude"].max() < 180

    def test_core_features_present(self):
        from data.kaggle_downloader import _build_synthetic_fallback
        df = _build_synthetic_fallback()
        required = ["temperature", "precipitation", "humidity", "wind_speed",
                     "soil_moisture", "severity", "latitude", "longitude"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"


class TestGenerateDisasterTrainingData:
    """Test the full training data generation pipeline."""

    def test_generate_disaster_data_returns_dataframe(self, tmp_path):
        from data.kaggle_downloader import generate_disaster_training_data
        df = generate_disaster_training_data(data_dir=str(tmp_path))
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) >= 9000

    def test_saves_combined_csv(self, tmp_path):
        from data.kaggle_downloader import generate_disaster_training_data
        generate_disaster_training_data(data_dir=str(tmp_path))
        assert os.path.exists(os.path.join(str(tmp_path), "combined_disaster_training.csv"))

    def test_cached_loading(self, tmp_path):
        from data.kaggle_downloader import generate_disaster_training_data
        # First call generates
        df1 = generate_disaster_training_data(data_dir=str(tmp_path))
        # Second call uses cache
        df2 = generate_disaster_training_data(data_dir=str(tmp_path))
        assert len(df1) == len(df2)


class TestCropImpactGeneration:
    """Test crop impact label generation."""

    def test_generate_crop_impact(self):
        from data.kaggle_downloader import _generate_crop_impact
        df = pd.DataFrame({
            "event_type": ["flood", "drought", "wildfire"] * 10,
            "precipitation": [50, 2, 0] * 10,
            "flood_probability": [0.7, 0.1, 0.0] * 10,
            "drought_score": [0, 3, 1] * 10
        })
        result = _generate_crop_impact(df)
        assert "crop_type" in result.columns
        assert "crop_impact" in result.columns
        assert "crop_yield_loss_pct" in result.columns
        assert len(result) == 30


class TestKaggleAuth:
    """Test Kaggle auth setup."""

    def test_setup_kaggle_auth_no_key(self):
        from data.kaggle_downloader import _setup_kaggle_auth
        original = os.environ.get("KAGGLE_KEY")
        os.environ.pop("KAGGLE_KEY", None)
        result = _setup_kaggle_auth()
        assert result is False
        if original:
            os.environ["KAGGLE_KEY"] = original
