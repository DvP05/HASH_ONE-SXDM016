"""
Tests for the expanded API tools — global regions, EONET, POWER, FIRMS global.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGlobalRegions:
    """Test the GLOBAL_REGIONS constant."""

    def test_global_regions_defined(self):
        from tools.api_tools import GLOBAL_REGIONS
        assert isinstance(GLOBAL_REGIONS, dict)
        assert len(GLOBAL_REGIONS) >= 15, f"Expected at least 15 regions, got {len(GLOBAL_REGIONS)}"

    def test_india_regions_present(self):
        from tools.api_tools import GLOBAL_REGIONS
        india_keys = [k for k in GLOBAL_REGIONS if "india" in k]
        assert len(india_keys) >= 3, f"Expected at least 3 India regions, got {india_keys}"

    def test_africa_regions_present(self):
        from tools.api_tools import GLOBAL_REGIONS
        africa_keys = [k for k in GLOBAL_REGIONS if "africa" in k]
        assert len(africa_keys) >= 2, f"Expected at least 2 Africa regions, got {africa_keys}"

    def test_brazil_regions_present(self):
        from tools.api_tools import GLOBAL_REGIONS
        brazil_keys = [k for k in GLOBAL_REGIONS if "brazil" in k]
        assert len(brazil_keys) >= 1

    def test_se_asia_regions_present(self):
        from tools.api_tools import GLOBAL_REGIONS
        se_asia = [k for k in GLOBAL_REGIONS if k in ("indonesia", "thailand", "vietnam")]
        assert len(se_asia) >= 2

    def test_region_has_bbox_and_centroid(self):
        from tools.api_tools import GLOBAL_REGIONS
        for key, info in GLOBAL_REGIONS.items():
            assert "bbox" in info, f"Region {key} missing bbox"
            assert "centroid" in info, f"Region {key} missing centroid"
            assert "name" in info, f"Region {key} missing name"
            assert len(info["centroid"]) == 2, f"Region {key} centroid must be (lat, lon)"


class TestFIRMSFallback:
    """Test FIRMS fallback data with global coverage."""

    def test_fallback_returns_dataframe(self):
        from tools.api_tools import _build_firms_fallback_dataframe
        df = _build_firms_fallback_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_fallback_has_global_regions(self):
        from tools.api_tools import _build_firms_fallback_dataframe
        df = _build_firms_fallback_dataframe()
        assert "region" in df.columns
        regions = " ".join(df["region"].unique())
        assert "India" in regions, "Missing India in FIRMS fallback"
        assert "Africa" in regions or "Kenya" in regions, "Missing Africa in FIRMS fallback"
        assert "Amazon" in regions or "Brazil" in regions, "Missing Brazil in FIRMS fallback"
        assert "Indonesia" in regions, "Missing Indonesia in FIRMS fallback"

    def test_fallback_has_weather_features(self):
        from tools.api_tools import _build_firms_fallback_dataframe
        df = _build_firms_fallback_dataframe()
        weather_cols = ["temperature_2m", "precipitation", "relative_humidity",
                        "wind_speed", "soil_moisture", "vegetation_index",
                        "drought_severity_index", "flood_risk_score"]
        for col in weather_cols:
            assert col in df.columns, f"Missing weather feature: {col}"


class TestEONETFallback:
    """Test EONET fallback data."""

    def test_eonet_fallback_returns_dataframe(self):
        from tools.api_tools import _build_eonet_fallback
        df = _build_eonet_fallback()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) >= 20

    def test_eonet_fallback_has_categories(self):
        from tools.api_tools import _build_eonet_fallback
        df = _build_eonet_fallback()
        categories = set(df["eonet_category"].unique())
        assert "Wildfires" in categories
        assert "Floods" in categories


class TestPOWERFallback:
    """Test POWER fallback data."""

    def test_power_fallback_returns_dataframe(self):
        from tools.api_tools import _build_power_fallback
        df = _build_power_fallback(28.6, 77.2)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) == 7  # 7 days

    def test_power_fallback_has_weather_params(self):
        from tools.api_tools import _build_power_fallback
        df = _build_power_fallback(28.6, 77.2)
        assert "t2m" in df.columns
        assert "prectotcorr" in df.columns
        assert "rh2m" in df.columns
        assert "ws2m" in df.columns
        assert "gwetroot" in df.columns


class TestFIRMSGlobal:
    """Test FIRMS global fetch (falls back to sample data without API key)."""

    def test_firms_global_fallback(self):
        """Without NASA_API_KEY, should return fallback data."""
        # Temporarily remove key
        original = os.environ.pop("NASA_API_KEY", None)
        try:
            from tools.api_tools import fetch_nasa_firms_global
            df = fetch_nasa_firms_global()
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert len(df) >= 20
        finally:
            if original:
                os.environ["NASA_API_KEY"] = original
