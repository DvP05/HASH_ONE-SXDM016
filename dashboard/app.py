"""
Flask web dashboard for the Autonomous Analysis System.
Serves a premium dark-themed single-page dashboard
with pipeline results, charts, and insights.
"""

from __future__ import annotations

import json
import os

from flask import Flask, render_template, jsonify, send_from_directory


def create_app(output_dir: str = "output") -> Flask:
    app = Flask(__name__,
                template_folder="templates",
                static_folder="static")
    app.config["OUTPUT_DIR"] = os.path.abspath(output_dir)

    @app.route("/")
    def index():
        """Serve the main dashboard."""
        data = _load_dashboard_data(app.config["OUTPUT_DIR"])
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

    return app


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
