"""
Kaggle Dataset Downloader & Preprocessor
Downloads and preprocesses disaster datasets from Kaggle for model training:
  1. Flood Prediction Dataset
  2. Predict Droughts using Weather & Soil Data
  3. Wildfire Risk Dataset

Also generates synthetic crop-impact labels to train crop damage prediction.
"""

from __future__ import annotations

import json
import logging
import os
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Kaggle Dataset Slugs
# ─────────────────────────────────────────────

KAGGLE_DATASETS = {
    "flood": {
        "slug": "brijlalraj/flood-prediction",
        "description": "Flood prediction factors dataset",
        "fallback_slug": "esraa208/flood-prediction-factors",
    },
    "drought": {
        "slug": "cdminix/us-drought-meteorological-data",
        "description": "US drought meteorological data with 18 weather indicators",
    },
    "wildfire": {
        "slug": "capcloudcoder/wildfire-risk-dataset-2024-2025-7-regions",
        "description": "Wildfire Risk Dataset 2024-2025 across 7 regions",
        "fallback_slug": "rtatman/188-million-us-wildfires",
    },
}

KAGGLE_DATA_DIR = os.path.join(os.path.dirname(__file__), "kaggle")


def _setup_kaggle_auth():
    """Set up Kaggle API authentication from environment variables."""
    kaggle_key = os.getenv("KAGGLE_KEY", "")
    kaggle_username = os.getenv("KAGGLE_USERNAME", "")

    if not kaggle_key:
        logger.warning("KAGGLE_KEY not set in environment. Kaggle downloads may fail.")
        return False

    # Set environment variables that the kaggle library expects
    os.environ["KAGGLE_KEY"] = kaggle_key
    if kaggle_username:
        os.environ["KAGGLE_USERNAME"] = kaggle_username

    # Also write to kaggle.json if it doesn't exist
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(kaggle_json):
        os.makedirs(kaggle_dir, exist_ok=True)
        creds = {"username": kaggle_username, "key": kaggle_key}
        with open(kaggle_json, "w") as f:
            json.dump(creds, f)
        # Kaggle expects restricted permissions (on Unix)
        try:
            os.chmod(kaggle_json, 0o600)
        except (OSError, AttributeError):
            pass  # Windows doesn't need chmod
        logger.info(f"Created Kaggle credentials at {kaggle_json}")

    return True


def download_kaggle_dataset(slug: str, dest_dir: str) -> str | None:
    """
    Download a Kaggle dataset by its slug (owner/dataset-name).
    Returns the directory path where files were extracted.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        os.makedirs(dest_dir, exist_ok=True)
        logger.info(f"Downloading Kaggle dataset: {slug} -> {dest_dir}")
        api.dataset_download_files(slug, path=dest_dir, unzip=True)
        logger.info(f"Successfully downloaded: {slug}")
        return dest_dir

    except SystemExit:
        logger.warning(f"Kaggle auth failed (SystemExit) for '{slug}'. Check KAGGLE_KEY/KAGGLE_USERNAME.")
        return None
    except Exception as e:
        logger.warning(f"Kaggle download failed for '{slug}': {e}")
        return None


def download_all_datasets(data_dir: str = "") -> dict:
    """
    Download all three disaster datasets from Kaggle.
    Returns a dict mapping dataset name -> local directory path.
    """
    if not data_dir:
        data_dir = KAGGLE_DATA_DIR

    os.makedirs(data_dir, exist_ok=True)

    _setup_kaggle_auth()

    results = {}
    for name, info in KAGGLE_DATASETS.items():
        dest = os.path.join(data_dir, name)
        path = download_kaggle_dataset(info["slug"], dest)

        # Try fallback slug if primary failed
        if path is None and "fallback_slug" in info:
            logger.info(f"Trying fallback slug for {name}: {info['fallback_slug']}")
            path = download_kaggle_dataset(info["fallback_slug"], dest)

        results[name] = path
        if path:
            files = os.listdir(path)
            logger.info(f"  {name}: {len(files)} files downloaded")
        else:
            logger.warning(f"  {name}: download failed, will use synthetic fallback")

    return results


def _load_flood_data(data_dir: str) -> pd.DataFrame | None:
    """Load and normalize the flood prediction dataset."""
    flood_dir = os.path.join(data_dir, "flood")
    if not os.path.exists(flood_dir):
        return None

    # Search for CSV files
    csv_files = [f for f in os.listdir(flood_dir) if f.endswith(".csv")]
    if not csv_files:
        return None

    try:
        df = pd.read_csv(os.path.join(flood_dir, csv_files[0]))
        logger.info(f"Loaded flood data: {df.shape}")

        # Normalize columns
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if "rain" in lower or "precip" in lower:
                col_map[col] = "precipitation"
            elif "temp" in lower:
                col_map[col] = "temperature"
            elif "humid" in lower:
                col_map[col] = "humidity"
            elif "flood" in lower and ("prob" in lower or "risk" in lower or "pred" in lower):
                col_map[col] = "flood_probability"
            elif "river" in lower or "discharge" in lower:
                col_map[col] = "river_discharge"

        df = df.rename(columns=col_map)
        df["event_type"] = "flood"
        df["data_source"] = "kaggle_flood"
        return df

    except Exception as e:
        logger.warning(f"Failed to load flood data: {e}")
        return None


def _load_drought_data(data_dir: str) -> pd.DataFrame | None:
    """Load and normalize the drought meteorological dataset."""
    drought_dir = os.path.join(data_dir, "drought")
    if not os.path.exists(drought_dir):
        return None

    # The dataset has train_timeseries/ and validation_timeseries/ dirs, or a CSV
    csv_files = []
    for root, dirs, files in os.walk(drought_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        return None

    try:
        # Try loading the smallest CSV first (validation set is smaller)
        csv_files.sort(key=lambda x: os.path.getsize(x))

        # Load a sample — drought dataset can be huge (900MB+)
        # Read only first 50K rows to keep things manageable
        df = pd.read_csv(csv_files[0], nrows=50000)
        logger.info(f"Loaded drought data: {df.shape} from {os.path.basename(csv_files[0])}")

        # The dataset has columns like: fips, date, score (drought level 0-5),
        # PRECTOT, PS, QV2M, T2M, T2MDEW, T2MWET, T2M_MAX, T2M_MIN, etc.
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if lower == "prectot" or lower == "prectotcorr":
                col_map[col] = "precipitation"
            elif lower == "t2m":
                col_map[col] = "temperature"
            elif lower in ("t2m_max", "t2mmax"):
                col_map[col] = "temperature_max"
            elif lower in ("t2m_min", "t2mmin"):
                col_map[col] = "temperature_min"
            elif lower == "qv2m":
                col_map[col] = "specific_humidity"
            elif lower in ("ws10m", "ws2m"):
                col_map[col] = "wind_speed"
            elif lower == "score":
                col_map[col] = "drought_score"

        df = df.rename(columns=col_map)
        df["event_type"] = "drought"
        df["data_source"] = "kaggle_drought"
        return df

    except Exception as e:
        logger.warning(f"Failed to load drought data: {e}")
        return None


def _load_wildfire_data(data_dir: str) -> pd.DataFrame | None:
    """Load and normalize the wildfire risk dataset."""
    wildfire_dir = os.path.join(data_dir, "wildfire")
    if not os.path.exists(wildfire_dir):
        return None

    csv_files = []
    for root, dirs, files in os.walk(wildfire_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        return None

    try:
        csv_files.sort(key=lambda x: os.path.getsize(x))
        df = pd.read_csv(csv_files[0], nrows=50000)
        logger.info(f"Loaded wildfire data: {df.shape} from {os.path.basename(csv_files[0])}")

        # Normalize columns
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if "lat" in lower:
                col_map[col] = "latitude"
            elif "lon" in lower:
                col_map[col] = "longitude"
            elif "temp" in lower and "max" not in lower and "min" not in lower:
                col_map[col] = "temperature"
            elif "wind" in lower:
                col_map[col] = "wind_speed"
            elif "humid" in lower:
                col_map[col] = "humidity"
            elif "precip" in lower or "rain" in lower:
                col_map[col] = "precipitation"
            elif "risk" in lower or "fire" in lower and "level" in lower:
                col_map[col] = "fire_risk_level"
            elif "region" in lower:
                col_map[col] = "region"

        df = df.rename(columns=col_map)
        df["event_type"] = "wildfire"
        df["data_source"] = "kaggle_wildfire"
        return df

    except Exception as e:
        logger.warning(f"Failed to load wildfire data: {e}")
        return None


def _generate_crop_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate realistic crop impact scores based on disaster type and severity.
    Based on FAO research data on crop yield loss from disasters.
    """
    np.random.seed(42)
    df = df.copy()

    crop_impact = []
    crop_type = []
    yield_loss_pct = []

    crops = ["rice", "wheat", "maize", "soybean", "cotton", "sugarcane", "vegetables"]

    for _, row in df.iterrows():
        event = row.get("event_type", "unknown")
        crop = np.random.choice(crops)

        if event == "flood":
            # Floods cause 30-90% crop loss depending on severity
            severity = row.get("flood_probability", row.get("precipitation", 50))
            if isinstance(severity, (str,)):
                severity = 50
            severity = float(severity) if not pd.isna(severity) else 50
            base_loss = np.clip(severity * 0.6 + np.random.normal(10, 15), 5, 95)
            impact = "high" if base_loss > 60 else "medium" if base_loss > 30 else "low"

        elif event == "drought":
            # Droughts cause 20-80% crop loss
            score = row.get("drought_score", 2)
            if pd.isna(score):
                score = 2
            base_loss = np.clip(float(score) * 15 + np.random.normal(5, 10), 5, 85)
            impact = "high" if base_loss > 50 else "medium" if base_loss > 25 else "low"

        elif event == "wildfire":
            # Wildfires cause 50-100% crop loss in affected areas
            base_loss = np.clip(65 + np.random.normal(0, 20), 20, 100)
            impact = "high" if base_loss > 70 else "medium" if base_loss > 40 else "low"

        else:
            base_loss = np.random.uniform(5, 30)
            impact = "low"

        crop_impact.append(impact)
        crop_type.append(crop)
        yield_loss_pct.append(round(float(base_loss), 1))

    df["crop_type"] = crop_type
    df["crop_impact"] = crop_impact
    df["crop_yield_loss_pct"] = yield_loss_pct

    return df


def _build_synthetic_fallback() -> pd.DataFrame:
    """
    Build comprehensive synthetic disaster training data when Kaggle
    downloads fail. Based on real-world statistical distributions.
    Covers all disaster types globally with crop impact.
    """
    np.random.seed(42)
    n_per_type = 3000  # 3000 per disaster type = 9000 total
    records = []

    # ── Global regions (latitude, longitude, region_name) ──
    regions = [
        (28.6, 77.2, "India — North"),
        (13.0, 80.2, "India — South"),
        (22.5, 88.3, "India — East (Bengal)"),
        (19.1, 72.9, "India — West (Maharashtra)"),
        (-1.3, 36.8, "East Africa — Kenya"),
        (9.1, 7.5, "West Africa — Nigeria"),
        (-14.3, 28.3, "Southern Africa — Zambia"),
        (-15.8, -47.9, "Brazil — Central"),
        (-3.1, -60.0, "Brazil — Amazon"),
        (-6.2, 106.8, "Indonesia — Java"),
        (13.8, 100.5, "Thailand — Central"),
        (21.0, 105.8, "Vietnam — North"),
        (37.5, -122.0, "USA — California"),
        (34.0, -118.2, "USA — Los Angeles"),
        (40.7, -74.0, "USA — East Coast"),
        (-33.9, 151.2, "Australia — Sydney"),
        (-12.5, 130.8, "Australia — NT"),
        (37.4, 15.1, "Europe — Mediterranean"),
        (35.7, 51.4, "Middle East — Iran"),
        (30.0, 31.2, "North Africa — Egypt"),
    ]

    crops = ["rice", "wheat", "maize", "soybean", "cotton", "sugarcane", "vegetables"]

    # ── Flood events ──
    for i in range(n_per_type):
        lat, lon, region = regions[i % len(regions)]
        lat += np.random.normal(0, 1.5)
        lon += np.random.normal(0, 1.5)
        month = np.random.choice([6, 7, 8, 9, 10, 11], p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.1])
        year = np.random.randint(2015, 2026)

        precip = round(float(np.clip(np.random.gamma(3, 20), 5, 300)), 1)
        temp = round(float(25 + np.random.normal(0, 5)), 1)
        humidity = round(float(np.clip(75 + np.random.normal(0, 10), 30, 100)), 1)
        wind = round(float(max(0.5, np.random.lognormal(1.0, 0.5))), 1)
        soil_moisture = round(float(np.clip(0.7 + np.random.normal(0, 0.1), 0.3, 1.0)), 3)
        flood_severity = round(float(np.clip(precip * 0.03 + np.random.normal(0, 0.5), 0, 10)), 2)

        crop = np.random.choice(crops)
        yield_loss = round(float(np.clip(flood_severity * 8 + np.random.normal(5, 10), 0, 100)), 1)
        impact = "high" if yield_loss > 50 else "medium" if yield_loss > 20 else "low"

        records.append({
            "event_type": "flood",
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "region": region,
            "year": year,
            "month": month,
            "temperature": temp,
            "precipitation": precip,
            "humidity": humidity,
            "wind_speed": wind,
            "soil_moisture": soil_moisture,
            "severity": flood_severity,
            "crop_type": crop,
            "crop_impact": impact,
            "crop_yield_loss_pct": yield_loss,
            "data_source": "synthetic_flood",
        })

    # ── Drought events ──
    for i in range(n_per_type):
        lat, lon, region = regions[i % len(regions)]
        lat += np.random.normal(0, 1.5)
        lon += np.random.normal(0, 1.5)
        month = np.random.choice([3, 4, 5, 6, 7, 8], p=[0.1, 0.15, 0.2, 0.25, 0.2, 0.1])
        year = np.random.randint(2015, 2026)

        precip = round(float(max(0, np.random.exponential(2))), 1)
        temp = round(float(35 + np.random.normal(0, 6)), 1)
        humidity = round(float(np.clip(25 + np.random.normal(0, 10), 5, 60)), 1)
        wind = round(float(max(0.5, np.random.lognormal(0.8, 0.4))), 1)
        soil_moisture = round(float(np.clip(0.15 + np.random.normal(0, 0.08), 0.01, 0.5)), 3)
        drought_score = round(float(np.clip(np.random.choice([0, 1, 2, 3, 4, 5],
                                    p=[0.1, 0.15, 0.25, 0.25, 0.15, 0.1]), 0, 5)), 0)

        crop = np.random.choice(crops)
        yield_loss = round(float(np.clip(drought_score * 14 + np.random.normal(3, 8), 0, 90)), 1)
        impact = "high" if yield_loss > 45 else "medium" if yield_loss > 20 else "low"

        records.append({
            "event_type": "drought",
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "region": region,
            "year": year,
            "month": month,
            "temperature": temp,
            "precipitation": precip,
            "humidity": humidity,
            "wind_speed": wind,
            "soil_moisture": soil_moisture,
            "severity": drought_score,
            "crop_type": crop,
            "crop_impact": impact,
            "crop_yield_loss_pct": yield_loss,
            "data_source": "synthetic_drought",
        })

    # ── Wildfire events ──
    for i in range(n_per_type):
        lat, lon, region = regions[i % len(regions)]
        lat += np.random.normal(0, 1.5)
        lon += np.random.normal(0, 1.5)
        month = np.random.choice([1, 2, 3, 4, 5, 10, 11, 12],
                                 p=[0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.1])
        year = np.random.randint(2015, 2026)

        precip = round(float(max(0, np.random.exponential(1))), 1)
        temp = round(float(38 + np.random.normal(0, 7)), 1)
        humidity = round(float(np.clip(20 + np.random.normal(0, 8), 5, 50)), 1)
        wind = round(float(max(0.5, np.random.lognormal(1.3, 0.6))), 1)
        soil_moisture = round(float(np.clip(0.1 + np.random.normal(0, 0.06), 0.01, 0.4)), 3)
        frp = round(float(max(1, np.random.gamma(3, 8))), 1)
        brightness = round(float(300 + frp * 1.2 + np.random.normal(0, 10)), 1)

        crop = np.random.choice(crops)
        yield_loss = round(float(np.clip(55 + frp * 0.5 + np.random.normal(0, 18), 5, 100)), 1)
        impact = "high" if yield_loss > 60 else "medium" if yield_loss > 30 else "low"

        records.append({
            "event_type": "wildfire",
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "region": region,
            "year": year,
            "month": month,
            "temperature": temp,
            "precipitation": precip,
            "humidity": humidity,
            "wind_speed": wind,
            "soil_moisture": soil_moisture,
            "severity": round(frp, 2),
            "brightness": brightness,
            "frp": frp,
            "crop_type": crop,
            "crop_impact": impact,
            "crop_yield_loss_pct": yield_loss,
            "data_source": "synthetic_wildfire",
        })

    return pd.DataFrame(records)


def generate_disaster_training_data(data_dir: str = "", force_download: bool = False) -> pd.DataFrame:
    """
    Main entry point: download Kaggle datasets, preprocess, merge with
    crop impact data, and return a unified training DataFrame.

    Falls back to high-quality synthetic data if Kaggle downloads fail.
    """
    if not data_dir:
        data_dir = KAGGLE_DATA_DIR

    combined_path = os.path.join(data_dir, "combined_disaster_training.csv")

    # Return cached if exists and not forcing re-download
    if os.path.exists(combined_path) and not force_download:
        logger.info(f"Loading cached training data from {combined_path}")
        return pd.read_csv(combined_path)

    os.makedirs(data_dir, exist_ok=True)

    # Try downloading from Kaggle
    logger.info("Attempting to download datasets from Kaggle...")
    downloaded = download_all_datasets(data_dir)

    frames = []

    # Load each dataset
    flood_df = _load_flood_data(data_dir)
    if flood_df is not None:
        flood_df = _generate_crop_impact(flood_df)
        frames.append(flood_df)
        logger.info(f"Loaded Kaggle flood data: {len(flood_df)} rows")

    drought_df = _load_drought_data(data_dir)
    if drought_df is not None:
        drought_df = _generate_crop_impact(drought_df)
        frames.append(drought_df)
        logger.info(f"Loaded Kaggle drought data: {len(drought_df)} rows")

    wildfire_df = _load_wildfire_data(data_dir)
    if wildfire_df is not None:
        wildfire_df = _generate_crop_impact(wildfire_df)
        frames.append(wildfire_df)
        logger.info(f"Loaded Kaggle wildfire data: {len(wildfire_df)} rows")

    # Always supplement with synthetic data for comprehensive coverage
    logger.info("Generating synthetic supplementary training data...")
    synthetic_df = _build_synthetic_fallback()
    frames.append(synthetic_df)

    # Combine all
    combined = pd.concat(frames, ignore_index=True)

    # Select and standardize the key training columns
    key_cols = [
        "event_type", "latitude", "longitude", "region",
        "year", "month", "temperature", "precipitation",
        "humidity", "wind_speed", "soil_moisture", "severity",
        "crop_type", "crop_impact", "crop_yield_loss_pct", "data_source",
    ]

    # Keep only columns that exist
    available = [c for c in key_cols if c in combined.columns]
    combined = combined[available].copy()

    # Fill missing values
    for col in combined.select_dtypes(include="number").columns:
        combined[col] = combined[col].fillna(combined[col].median())
    for col in combined.select_dtypes(include="object").columns:
        combined[col] = combined[col].fillna("unknown")

    # Save combined dataset
    combined.to_csv(combined_path, index=False)
    logger.info(f"Saved combined training data: {combined.shape} to {combined_path}")

    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from dotenv import load_dotenv
    load_dotenv()

    df = generate_disaster_training_data(force_download=True)
    print(f"\nCombined dataset shape: {df.shape}")
    print(f"\nEvent type distribution:")
    print(df["event_type"].value_counts())
    print(f"\nCrop impact distribution:")
    if "crop_impact" in df.columns:
        print(df["crop_impact"].value_counts())
    print(f"\nSample rows:")
    print(df.head(10))
