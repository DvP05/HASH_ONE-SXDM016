"""
Flask web dashboard for the Autonomous Analysis System.
Serves a premium dark-themed Geo Caster dashboard
with pipeline results, interactive map, charts, and insights.
"""

from __future__ import annotations

import json
import os

import numpy as np

from flask import Flask, render_template, jsonify, send_from_directory, request


def create_app(output_dir: str = "output") -> Flask:
    app = Flask(__name__,
                template_folder="templates",
                static_folder="static")
    app.config["OUTPUT_DIR"] = os.path.abspath(output_dir)

    @app.route("/")
    def index():
        """Serve the main dashboard."""
        data = _load_dashboard_data(app.config["OUTPUT_DIR"])
        # Add disaster summary to template data
        data["disaster_summary"] = _compute_disaster_summary()
        return render_template("index.html", data=data)

    @app.route("/api/pipeline-result")
    def api_pipeline_result():
        """API: Get pipeline result JSON."""
        path = os.path.join(app.config["OUTPUT_DIR"], "pipeline_result.json")
        if os.path.exists(path):
            with open(path) as f:
                return jsonify(json.load(f))
        return jsonify({"error": "No pipeline results found"}), 404

    @app.route("/api/cleaning-log")
    def api_cleaning_log():
        path = os.path.join(app.config["OUTPUT_DIR"], "cleaning_log.json")
        if os.path.exists(path):
            with open(path) as f:
                return jsonify(json.load(f))
        return jsonify([])

    @app.route("/api/eda-report")
    def api_eda_report():
        path = os.path.join(app.config["OUTPUT_DIR"], "eda_report.json")
        if os.path.exists(path):
            with open(path) as f:
                return jsonify(json.load(f))
        return jsonify({})

    @app.route("/api/feature-importance")
    def api_feature_importance():
        path = os.path.join(app.config["OUTPUT_DIR"], "feature_importance.json")
        if os.path.exists(path):
            with open(path) as f:
                return jsonify(json.load(f))
        return jsonify({})

    @app.route("/api/fire-locations")
    def api_fire_locations():
        """API: Get fire location data for the map view."""
        locations = _build_fire_locations()
        return jsonify(locations)

    @app.route("/api/disaster-summary")
    def api_disaster_summary():
        """API: Get summary statistics broken down by disaster type (Fire, Flood, Drought)."""
        summary = _compute_disaster_summary()
        return jsonify(summary)

    @app.route("/charts/<path:filename>")
    def serve_chart(filename):
        charts_dir = os.path.join(app.config["OUTPUT_DIR"], "charts")
        return send_from_directory(charts_dir, filename)

    @app.route("/api/insight-report")
    def api_insight_report():
        path = os.path.join(app.config["OUTPUT_DIR"], "insight_report.md")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return jsonify({"content": f.read()})
        return jsonify({"content": ""})

    @app.route("/api/model-card")
    def api_model_card():
        path = os.path.join(app.config["OUTPUT_DIR"], "model_card.md")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return jsonify({"content": f.read()})
        return jsonify({"content": ""})

    @app.route("/api/refresh-status")
    def api_refresh_status():
        """API: Get the scheduler refresh status."""
        try:
            from scheduler import load_latest_cache
            cache = load_latest_cache()
            cache_info = {}
            for source in ("firms", "eonet", "power"):
                latest_path = os.path.join("data", "live_cache", f"{source}_latest.csv")
                if os.path.exists(latest_path):
                    import time as _t
                    mtime = os.path.getmtime(latest_path)
                    cache_info[source] = {
                        "available": True,
                        "rows": len(cache.get(source, [])) if cache.get(source) is not None else 0,
                        "last_updated": _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(mtime)),
                    }
                else:
                    cache_info[source] = {"available": False, "rows": 0, "last_updated": None}
            return jsonify({"status": "ok", "cache": cache_info})
        except Exception as e:
            return jsonify({"status": "no_scheduler", "error": str(e)})

    @app.route("/api/global-regions")
    def api_global_regions():
        """API: Get the list of monitored global regions."""
        from tools.api_tools import GLOBAL_REGIONS
        regions = []
        for key, info in GLOBAL_REGIONS.items():
            regions.append({
                "key": key,
                "name": info["name"],
                "centroid_lat": info["centroid"][0],
                "centroid_lon": info["centroid"][1],
                "bbox": info["bbox"],
            })
        return jsonify(regions)

    return app


def _compute_disaster_summary() -> dict:
    """Compute summary statistics for each disaster type from fire location data."""
    locations = _build_fire_locations()

    summary = {
        "total": len(locations),
        "fire": {"count": 0, "high_count": 0, "avg_severity": 0, "max_severity": 0,
                 "avg_brightness": 0, "avg_frp": 0},
        "flood": {"count": 0, "high_count": 0, "avg_severity": 0, "max_severity": 0,
                  "avg_water_level": 0, "avg_precipitation": 0},
        "drought": {"count": 0, "high_count": 0, "avg_severity": 0, "max_severity": 0,
                    "avg_pdsi": 0, "avg_soil_moisture": 0},
    }

    fire_sevs, flood_sevs, drought_sevs = [], [], []
    fire_brightness, fire_frp = [], []
    flood_water, flood_precip = [], []
    drought_pdsi, drought_moisture = [], []

    for loc in locations:
        dtype = loc.get("disaster_type", "Fire").lower()
        sev = abs(float(loc.get("severity_value", 0)))
        is_high = loc.get("confidence") == "high"

        if dtype == "fire":
            summary["fire"]["count"] += 1
            if is_high:
                summary["fire"]["high_count"] += 1
            fire_sevs.append(sev)
            fire_brightness.append(float(loc.get("brightness", 0)))
            fire_frp.append(float(loc.get("frp", 0)))
        elif dtype == "flood":
            summary["flood"]["count"] += 1
            if is_high:
                summary["flood"]["high_count"] += 1
            flood_sevs.append(sev)
            flood_water.append(sev)
            flood_precip.append(float(loc.get("precipitation", 0)))
        elif dtype == "drought":
            summary["drought"]["count"] += 1
            if is_high:
                summary["drought"]["high_count"] += 1
            drought_sevs.append(sev)
            drought_pdsi.append(float(loc.get("severity_value", 0)))
            drought_moisture.append(float(loc.get("soil_moisture", 0)))

    # Compute averages
    if fire_sevs:
        summary["fire"]["avg_severity"] = round(np.mean(fire_sevs), 2)
        summary["fire"]["max_severity"] = round(max(fire_sevs), 2)
        summary["fire"]["avg_brightness"] = round(np.mean(fire_brightness), 1)
        summary["fire"]["avg_frp"] = round(np.mean(fire_frp), 1)

    if flood_sevs:
        summary["flood"]["avg_severity"] = round(np.mean(flood_sevs), 2)
        summary["flood"]["max_severity"] = round(max(flood_sevs), 2)
        summary["flood"]["avg_water_level"] = round(np.mean(flood_water), 2)
        summary["flood"]["avg_precipitation"] = round(np.mean(flood_precip), 2) if flood_precip else 0

    if drought_sevs:
        summary["drought"]["avg_severity"] = round(np.mean(drought_sevs), 2)
        summary["drought"]["max_severity"] = round(max(drought_sevs), 2)
        summary["drought"]["avg_pdsi"] = round(np.mean(drought_pdsi), 2)
        summary["drought"]["avg_soil_moisture"] = round(np.mean(drought_moisture), 3) if drought_moisture else 0

    return summary


def _build_fire_locations() -> list[dict]:
    """Build fire location data with ALL disaster types (Fire, Flood, Drought) and global coverage."""

    np.random.seed(42)

    base_rows = [
        # Australia
        {"id": 0, "latitude": -12.45, "longitude": 130.84, "brightness": 327.4,
         "scan": 0.51, "track": 0.41, "acq_date": "2026-04-05", "acq_time": "00:45",
         "satellite": "VIIRS SNPP", "confidence": "high", "bright_t31": 299.2,
         "frp": 21.7, "daynight": "Day", "region": "Northern Territory, Australia"},
        {"id": 1, "latitude": -33.87, "longitude": 151.21, "brightness": 306.8,
         "scan": 0.40, "track": 0.35, "acq_date": "2026-04-05", "acq_time": "00:50",
         "satellite": "VIIRS SNPP", "confidence": "nominal", "bright_t31": 288.5,
         "frp": 8.9, "daynight": "Day", "region": "Southeast Australia"},
        # USA
        {"id": 2, "latitude": 38.51, "longitude": -121.42, "brightness": 302.8,
         "scan": 0.37, "track": 0.33, "acq_date": "2026-04-05", "acq_time": "00:53",
         "satellite": "VIIRS SNPP", "confidence": "low", "bright_t31": 287.3,
         "frp": 5.2, "daynight": "Night", "region": "California, USA"},
        {"id": 3, "latitude": 31.0, "longitude": -83.5, "brightness": 311.1,
         "scan": 0.42, "track": 0.36, "acq_date": "2026-04-05", "acq_time": "00:55",
         "satellite": "VIIRS SNPP", "confidence": "nominal", "bright_t31": 291.5,
         "frp": 10.2, "daynight": "Night", "region": "Southeast USA"},
        # India
        {"id": 4, "latitude": 28.61, "longitude": 77.21, "brightness": 310.5,
         "scan": 0.45, "track": 0.39, "acq_date": "2026-04-05", "acq_time": "02:10",
         "satellite": "VIIRS SNPP", "confidence": "high", "bright_t31": 294.8,
         "frp": 18.2, "daynight": "Day", "region": "India — North (Delhi)"},
        {"id": 5, "latitude": 13.08, "longitude": 80.27, "brightness": 305.2,
         "scan": 0.42, "track": 0.35, "acq_date": "2026-04-05", "acq_time": "02:20",
         "satellite": "VIIRS SNPP", "confidence": "nominal", "bright_t31": 290.5,
         "frp": 12.1, "daynight": "Day", "region": "India — South (Chennai)"},
        {"id": 6, "latitude": 22.57, "longitude": 88.36, "brightness": 315.8,
         "scan": 0.48, "track": 0.40, "acq_date": "2026-04-05", "acq_time": "02:15",
         "satellite": "VIIRS SNPP", "confidence": "high", "bright_t31": 296.3,
         "frp": 16.5, "daynight": "Day", "region": "India — East (Kolkata)"},
        {"id": 7, "latitude": 19.08, "longitude": 72.88, "brightness": 308.3,
         "scan": 0.44, "track": 0.37, "acq_date": "2026-04-05", "acq_time": "02:25",
         "satellite": "VIIRS SNPP", "confidence": "nominal", "bright_t31": 292.1,
         "frp": 11.4, "daynight": "Day", "region": "India — West (Mumbai)"},
        # Africa
        {"id": 8, "latitude": -1.29, "longitude": 36.82, "brightness": 322.1,
         "scan": 0.52, "track": 0.42, "acq_date": "2026-04-05", "acq_time": "03:20",
         "satellite": "VIIRS SNPP", "confidence": "high", "bright_t31": 298.7,
         "frp": 22.8, "daynight": "Day", "region": "East Africa — Kenya"},
        {"id": 9, "latitude": 9.06, "longitude": 7.49, "brightness": 318.4,
         "scan": 0.50, "track": 0.41, "acq_date": "2026-04-05", "acq_time": "03:30",
         "satellite": "VIIRS SNPP", "confidence": "nominal", "bright_t31": 295.2,
         "frp": 15.9, "daynight": "Day", "region": "West Africa — Nigeria"},
        {"id": 10, "latitude": -25.0, "longitude": 28.5, "brightness": 309.7,
         "scan": 0.43, "track": 0.36, "acq_date": "2026-04-05", "acq_time": "03:40",
         "satellite": "VIIRS SNPP", "confidence": "low", "bright_t31": 289.1,
         "frp": 7.2, "daynight": "Day", "region": "Southern Africa"},
        # Brazil / Amazon
        {"id": 11, "latitude": -3.12, "longitude": -59.98, "brightness": 340.2,
         "scan": 0.56, "track": 0.45, "acq_date": "2026-04-05", "acq_time": "05:00",
         "satellite": "VIIRS SNPP", "confidence": "high", "bright_t31": 302.4,
         "frp": 32.5, "daynight": "Day", "region": "Brazil — Amazon"},
        {"id": 12, "latitude": -15.8, "longitude": -47.9, "brightness": 315.6,
         "scan": 0.47, "track": 0.39, "acq_date": "2026-04-05", "acq_time": "05:10",
         "satellite": "VIIRS SNPP", "confidence": "nominal", "bright_t31": 293.8,
         "frp": 14.7, "daynight": "Day", "region": "Brazil — Central"},
        # Southeast Asia
        {"id": 13, "latitude": -6.17, "longitude": 106.83, "brightness": 325.7,
         "scan": 0.53, "track": 0.43, "acq_date": "2026-04-05", "acq_time": "08:20",
         "satellite": "VIIRS SNPP", "confidence": "high", "bright_t31": 300.1,
         "frp": 25.3, "daynight": "Day", "region": "Indonesia — Java"},
        {"id": 14, "latitude": 13.8, "longitude": 100.5, "brightness": 312.4,
         "scan": 0.46, "track": 0.38, "acq_date": "2026-04-05", "acq_time": "08:30",
         "satellite": "VIIRS SNPP", "confidence": "nominal", "bright_t31": 291.8,
         "frp": 11.6, "daynight": "Day", "region": "Thailand — Bangkok"},
        {"id": 15, "latitude": 21.03, "longitude": 105.85, "brightness": 308.9,
         "scan": 0.44, "track": 0.37, "acq_date": "2026-04-05", "acq_time": "08:40",
         "satellite": "VIIRS SNPP", "confidence": "low", "bright_t31": 289.4,
         "frp": 6.8, "daynight": "Day", "region": "Vietnam — Hanoi"},
        # Europe / Middle East
        {"id": 16, "latitude": 37.39, "longitude": 15.09, "brightness": 308.3,
         "scan": 0.43, "track": 0.37, "acq_date": "2026-04-05", "acq_time": "10:00",
         "satellite": "VIIRS SNPP", "confidence": "nominal", "bright_t31": 289.5,
         "frp": 9.7, "daynight": "Day", "region": "Mediterranean — Sicily"},
        {"id": 17, "latitude": 35.69, "longitude": 51.39, "brightness": 312.6,
         "scan": 0.46, "track": 0.38, "acq_date": "2026-04-05", "acq_time": "11:00",
         "satellite": "VIIRS SNPP", "confidence": "low", "bright_t31": 292.1,
         "frp": 7.4, "daynight": "Day", "region": "Middle East — Iran"},
        # Central America
        {"id": 18, "latitude": 13.0, "longitude": -86.0, "brightness": 319.8,
         "scan": 0.51, "track": 0.42, "acq_date": "2026-04-05", "acq_time": "06:00",
         "satellite": "VIIRS SNPP", "confidence": "high", "bright_t31": 297.4,
         "frp": 19.5, "daynight": "Day", "region": "Central America — Honduras"},
    ]

    locations = []
    idx = 0
    disaster_types = ["Fire", "Flood", "Drought"]
    for i in range(3):
        for row in base_rows:
            r = row.copy()
            r["id"] = idx

            # Rotate disaster types
            dtype = disaster_types[idx % 3]
            r["disaster_type"] = dtype

            # Crop Impact Logic
            if r["confidence"] == "high" or r["frp"] > 25:
                r["crop_impact"] = "Total Destruction Expected"
            elif r["confidence"] == "nominal":
                r["crop_impact"] = "Moderate Damage Expected"
            else:
                r["crop_impact"] = "Minimal To No Harm Expected"

            # Severity metrics mapping
            if dtype == "Flood":
                r["severity_value"] = round(float(r["frp"]) * 0.1 + i * 0.15, 2)
                r["severity_unit"] = "Water Level (m)"
            elif dtype == "Drought":
                r["severity_value"] = round(float(r["frp"]) * -0.2 - i * 0.1, 2)
                r["severity_unit"] = "PDSI Index"
            else:
                r["severity_value"] = round(float(r["frp"]) + i * 0.9, 2)
                r["severity_unit"] = "MW (Radiative Power)"

            r["latitude"] = round(float(r["latitude"]) + (i * 0.07), 5)
            r["longitude"] = round(float(r["longitude"]) - (i * 0.05), 5)
            r["brightness"] = round(float(r["brightness"]) + i * 1.2, 2)
            r["frp"] = round(float(r["frp"]) + i * 0.9, 2)
            r["bright_t31"] = round(float(r["bright_t31"]) + i * 0.4, 2)

            # ── Enriched weather/climate features for ALL disaster types ──
            base_temp = 28 + abs(float(r["latitude"])) * 0.1
            r["temperature_2m"] = round(base_temp + np.random.normal(0, 3), 2)
            r["precipitation"] = round(max(0, np.random.exponential(2.5 if dtype == "Flood" else 1.0)), 2)
            r["relative_humidity"] = round(float(np.clip(
                (70 if dtype == "Flood" else 35) + np.random.normal(0, 10), 10, 98)), 2)
            r["wind_speed"] = round(max(0.1, np.random.lognormal(1.0, 0.5)), 2)
            r["soil_moisture"] = round(float(np.clip(
                (0.8 if dtype == "Flood" else 0.15 if dtype == "Drought" else 0.4) + np.random.normal(0, 0.1),
                0.02, 0.98)), 4)
            r["vegetation_index"] = round(float(np.clip(
                (0.6 if dtype == "Flood" else 0.2 if dtype == "Drought" else 0.4) + np.random.normal(0, 0.1),
                0.05, 0.9)), 4)
            r["drought_severity_index"] = round(float(np.clip(
                (-3.5 if dtype == "Drought" else 1.0 if dtype == "Flood" else -0.5) + np.random.normal(0, 0.8),
                -6.0, 6.0)), 2)
            r["flood_risk_score"] = round(float(np.clip(
                (7.5 if dtype == "Flood" else 1.0 if dtype == "Drought" else 2.0) + np.random.normal(0, 1.0),
                0, 10)), 2)

            locations.append(r)
            idx += 1

    return locations


def _load_dashboard_data(output_dir: str) -> dict:
    """Load all pipeline outputs for the dashboard template."""
    data = {
        "pipeline_result": {},
        "cleaning_log": [],
        "eda_report": {},
        "feature_importance": {},
        "insight_report": "",
        "model_card": "",
        "charts": [],
    }

    # Pipeline result
    path = os.path.join(output_dir, "pipeline_result.json")
    if os.path.exists(path):
        with open(path) as f:
            data["pipeline_result"] = json.load(f)

    # Cleaning log
    path = os.path.join(output_dir, "cleaning_log.json")
    if os.path.exists(path):
        with open(path) as f:
            data["cleaning_log"] = json.load(f)

    # EDA report
    path = os.path.join(output_dir, "eda_report.json")
    if os.path.exists(path):
        with open(path) as f:
            data["eda_report"] = json.load(f)

    # Feature importance
    path = os.path.join(output_dir, "feature_importance.json")
    if os.path.exists(path):
        with open(path) as f:
            data["feature_importance"] = json.load(f)

    # Insight report
    path = os.path.join(output_dir, "insight_report.md")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data["insight_report"] = f.read()

    # Model card
    path = os.path.join(output_dir, "model_card.md")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data["model_card"] = f.read()

    # Charts
    charts_dir = os.path.join(output_dir, "charts")
    if os.path.exists(charts_dir):
        for f in os.listdir(charts_dir):
            if f.endswith(".html"):
                data["charts"].append(f)

    return data


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
