# 🛰️ Luminus-X — Autonomous Analysis System

> **Global Environmental Resilience & Crop Protection Command Center**
>
> An end-to-end, agentic AI pipeline that ingests live NASA satellite telemetry, enriches it with weather data and historical disaster datasets, trains a fire-confidence classification model, and presents real-time insights through an interactive web dashboard — all powered by your choice of LLM backend (OpenAI, Anthropic, Gemini, Ollama, or zero-dependency rule-based fallback).

---

## 📋 Table of Contents

1. [What This System Does](#what-this-system-does)
2. [Architecture Overview](#architecture-overview)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Environment Configuration](#environment-configuration)
7. [Running the System](#running-the-system)
   - [Quick Start (No API Keys Needed)](#quick-start-no-api-keys-needed)
   - [Run with Live NASA Data](#run-with-live-nasa-data)
   - [Launch the Web Dashboard](#launch-the-web-dashboard)
   - [Enable the Hourly Scheduler](#enable-the-hourly-scheduler)
   - [Download Kaggle Disaster Datasets](#download-kaggle-disaster-datasets)
   - [Generate Sample Data Only](#generate-sample-data-only)
   - [Use a Custom Dataset](#use-a-custom-dataset)
8. [Command-Line Reference](#command-line-reference)
9. [LLM Provider Setup](#llm-provider-setup)
10. [Pipeline Stages Explained](#pipeline-stages-explained)
11. [Data Sources](#data-sources)
12. [Configuration File](#configuration-file)
13. [Output Files](#output-files)
14. [Dashboard Features](#dashboard-features)
15. [Scheduler](#scheduler)
16. [Testing](#testing)
17. [Troubleshooting](#troubleshooting)
18. [Frequently Asked Questions](#frequently-asked-questions)

---

## What This System Does

Luminus-X autonomously:

1. **Collects** live fire/disaster data from three free NASA APIs — FIRMS (active fire detections at 375 m resolution), EONET (natural event tracker), and POWER (daily climate/weather)
2. **Augments** the live telemetry with historical flood, drought and wildfire training data from Kaggle
3. **Cleans and validates** all data, resolving missing values, capping outliers, and deduplicating records
4. **Runs exploratory analysis** (distributions, correlations, geo-temporal patterns, anomaly detection)
5. **Engineers features** — log transforms, drought/flood derived indices, soil moisture, vegetation index, and categorical encodings
6. **Trains an ML model** (XGBoost / LightGBM / Random Forest, with Optuna hyperparameter search) to predict fire-detection *confidence* (`l` / `n` / `h`)
7. **Generates LLM-powered insights** — executive summaries, risk maps, recommendations, and model cards
8. **Serves results** through a Flask web dashboard with interactive Plotly charts, a live world map, and scheduler status

The pipeline is 100 % autonomous — it makes its own decisions at every stage, with an optional LLM brain driving the analytical reasoning.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         main.py (entry point)                        │
│   CLI args → load config/pipeline_config.yaml → PipelineOrchestrator│
└──────────────────────────────┬──────────────────────────────────────┘
                               │ runs 6 sequential stages
        ┌──────────────────────▼──────────────────────┐
        │           PipelineOrchestrator               │
        │  (orchestrator/pipeline_orchestrator.py)     │
        │  • quality gates between every stage         │
        │  • shared MemoryStore for inter-agent state  │
        └───┬──────┬──────┬──────┬──────┬─────────────┘
            │      │      │      │      │
    Stage 1 │      │Stage2│Stage3│Stage4│Stage5     Stage 6
            ▼      ▼      ▼      ▼      ▼           ▼
      Data        Data   EDA   Feature  Modeling  Insight
    Collection  Cleaning      Engineer  Agent     Agent
     Agent      Agent  Agent  Agent
    ─────────────────────────────────────────────────────────────
     NASA        Handle  Plots  Log/scale Train     Executive
     FIRMS       missing corrs  drought   XGBoost   summaries
     EONET       values  hists  index     + SHAP    Risk map
     POWER       outliers geo   flood     CV eval   Recommend-
     Kaggle      dupes   trend  score             ations
    ─────────────────────────────────────────────────────────────
                      ↕ every agent talks to ↕
                 LLMClient (core/llm_client.py)
                  auto-detects: OpenAI | Anthropic
                               Gemini | Ollama
                               Rule-based fallback
```

---

## Project Structure

```
geo-caster/
│
├── main.py                    # ← Entry point / CLI
├── scheduler.py               # Hourly NASA data refresh loop
├── requirements.txt           # Python dependencies
├── .env                       # Your secret API keys (create from .env.example)
├── .env.example               # Template for environment variables
│
├── config/
│   ├── pipeline_config.yaml   # Pipeline settings (target, task type, LLM, sources)
│   └── quality_gates.yaml     # Pass/fail thresholds for each stage
│
├── core/
│   ├── llm_client.py          # Unified LLM interface
│   ├── llm_providers.py       # OpenAI / Anthropic / Gemini / Ollama / RuleBased
│   ├── models.py              # Pydantic data models (PipelineConfig, LLMConfig…)
│   └── sandbox.py             # Safe code execution environment
│
├── orchestrator/
│   ├── pipeline_orchestrator.py  # Runs all 6 stages, enforces quality gates
│   └── quality_gates.py          # Quality gate evaluation logic
│
├── agents/
│   ├── base_agent.py              # Shared agent functionality, tool execution
│   ├── data_collection_agent.py   # Stage 1 — ingest & validate data
│   ├── data_cleaning_agent.py     # Stage 2 — clean & impute
│   ├── eda_agent.py               # Stage 3 — exploratory data analysis
│   ├── feature_engineering_agent.py # Stage 4 — transform & encode
│   ├── modeling_agent.py          # Stage 5 — train, tune, evaluate models
│   └── insight_agent.py           # Stage 6 — LLM-powered insights & reports
│
├── tools/
│   ├── registry.py            # @tool decorator — registers callable tools
│   ├── api_tools.py           # NASA FIRMS / EONET / POWER API wrappers
│   ├── data_tools.py          # CSV reader, statistics, deduplication
│   ├── ml_tools.py            # Model training, SHAP, cross-validation
│   └── viz_tools.py           # Plotly / Seaborn chart generators
│
├── memory/
│   └── memory_store.py        # In-memory key-value store shared across agents
│
├── dashboard/
│   ├── app.py                 # Flask web application
│   ├── templates/             # Jinja2 HTML templates
│   └── static/                # CSS / JS assets
│
└── data/
    ├── generate_sample_data.py   # Generates synthetic SaaS churn dataset
    ├── kaggle_downloader.py      # Downloads & merges Kaggle disaster datasets
    ├── sample_churn_data.csv     # Pre-generated sample dataset
    ├── kaggle/                   # Downloaded Kaggle CSVs (auto-created)
    └── live_cache/               # Timestamped NASA API cache (auto-created)
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| **Python** | 3.10 or higher | 3.11+ recommended |
| **pip** | latest | comes with Python |
| **Internet access** | — | for NASA API calls |
| **NASA API Key** | optional — free | only needed for live FIRMS data |
| **LLM API Key** | optional | OpenAI / Anthropic / Gemini — or use Ollama locally / rules fallback |
| **Kaggle API Key** | optional | only if you want to download Kaggle training datasets |

> **Windows users:** all commands below should be run in **PowerShell** or **Command Prompt**. The system uses UTF-8 output by default and handles Windows cp1252 encoding automatically.

---

## Installation

### Step 1 — Clone the repository

```powershell
git clone https://github.com/your-username/geo-caster.git
cd geo-caster
```

Or if you already have the folder:
```powershell
cd .\geo-caster
```

---

### Step 2 — Create a virtual environment

It is strongly recommended to use a virtual environment to avoid dependency conflicts.

```powershell
python -m venv .venv
```

---

### Step 3 — Activate the virtual environment

**PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Command Prompt:**
```cmd
.venv\Scripts\activate.bat
```

You should see `(.venv)` appear at the start of your terminal prompt.

---

### Step 4 — Install dependencies

```powershell
pip install -r requirements.txt
```

This installs all **required** packages:

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `pydantic` | Data models and validation |
| `pyyaml` | Config file parsing |
| `python-dotenv` | Loading `.env` files |
| `requests` | HTTP calls to NASA APIs |
| `scikit-learn` | ML algorithms, preprocessing |
| `xgboost`, `lightgbm` | Gradient boosting models |
| `optuna` | Hyperparameter optimization |
| `shap` | Model explainability |
| `plotly`, `seaborn`, `matplotlib` | Visualization |
| `flask` | Web dashboard server |
| `kaggle` | Kaggle dataset downloads |
| `schedule` | Background task scheduling |

**Optional LLM packages** (install as needed):
```powershell
pip install openai>=1.0          # for OpenAI GPT-4o
pip install anthropic>=0.25      # for Claude
pip install google-generativeai>=0.5  # for Gemini
```

---

### Step 5 — Set up environment variables

```powershell
copy .env.example .env
```

Then open `.env` in a text editor and fill in your keys (see [Environment Configuration](#environment-configuration) below).

---

## Environment Configuration

Edit `.env` in the project root. All values are optional — the system works without any keys using enriched sample data and rule-based AI reasoning.

```env
# ──────────────────────────────────────────
# LLM Provider — pick ONE (or none for fallback)
# ──────────────────────────────────────────

# Option A: OpenAI (GPT-4o, GPT-4o-mini)
OPENAI_API_KEY=sk-proj-...

# Option B: Anthropic (Claude Sonnet / Haiku)
ANTHROPIC_API_KEY=sk-ant-...

# Option C: Google Gemini
GOOGLE_API_KEY=AIza...

# Option D: Ollama (local, no cost)
# Make sure Ollama is running: https://ollama.ai
OLLAMA_BASE_URL=http://localhost:11434

# Force a specific provider (default: auto-detect)
# LLM_PROVIDER=auto   # auto | openai | anthropic | gemini | ollama | rule_based
# LLM_MODEL=          # Override the default model name for the chosen provider

# ──────────────────────────────────────────
# NASA FIRMS API Key (free from https://firms.modaps.eosdis.nasa.gov/api/area/)
# Without this key, the pipeline uses a built-in global sample dataset.
# ──────────────────────────────────────────
NASA_API_KEY=your_nasa_firms_map_key_here

# ──────────────────────────────────────────
# Kaggle Credentials (for disaster dataset downloads)
# Create from https://www.kaggle.com/settings → API → Create New Token
# ──────────────────────────────────────────
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key

# ──────────────────────────────────────────
# Scheduler
# ──────────────────────────────────────────
REFRESH_INTERVAL_SECONDS=3600   # How often to refresh NASA data (default: 1 hour)
```

### LLM Provider Auto-Detection Priority

When `LLM_PROVIDER=auto` (the default), the system checks for credentials in this order:

1. `OPENAI_API_KEY` → uses **GPT-4o-mini**
2. `ANTHROPIC_API_KEY` → uses **Claude Sonnet**
3. `GOOGLE_API_KEY` → uses **Gemini 2.0 Flash**
4. Ollama running at `localhost:11434` → uses **best available local model**
5. **Rule-based fallback** → deterministic heuristics, no LLM required

---

## Running the System

### Quick Start (No API Keys Needed)

Run the full 6-stage pipeline using built-in sample data and rule-based AI:

```powershell
python main.py
```

The pipeline will:
1. Detect that no dataset exists and auto-generate `data/sample_churn_data.csv`
2. Run all 6 agentic stages (collection → cleaning → EDA → features → modeling → insights)
3. Save all outputs to `output/`
4. Print final recommendations in the terminal

Expected runtime: **2–5 minutes** on a modern machine.

---

### Run with Live NASA Data

Provide your NASA API key in `.env`, then:

```powershell
python main.py --config config/pipeline_config.yaml
```

The pipeline uses the YAML config which points to `nasa_api` sources. It will:
- Pull live FIRMS satellite fire detections from **20 global regions**
- Fetch EONET natural disaster events (last 30 days)
- Enrich each detection with NASA POWER weather data
- Merge with Kaggle historical training data (if `KAGGLE_USERNAME` is set)

---

### Launch the Web Dashboard

Run the pipeline **and** open the dashboard at the end:

```powershell
python main.py --dashboard
```

Or, if you've already run the pipeline once and want to re-open the dashboard:

```powershell
python main.py --dashboard --config config/pipeline_config.yaml
```

The dashboard starts at **http://localhost:5000**

Open a browser and navigate to that URL. It shows:
- 📊 Model performance metrics
- 🗺️ Interactive global fire/disaster map
- 📈 Feature importance chart (SHAP values)
- 📋 Executive insight report
- 🔄 Scheduler status and last-refresh timestamp

---

### Enable the Hourly Scheduler

Keep the system running and refresh NASA data every hour automatically:

```powershell
python main.py --scheduler
```

Or combine with the dashboard:

```powershell
python main.py --dashboard --scheduler
```

Press `Ctrl+C` to stop the scheduler gracefully.

Change the refresh interval via the environment variable:

```powershell
# Refresh every 30 minutes instead of 1 hour
$env:REFRESH_INTERVAL_SECONDS = "1800"
python main.py --scheduler
```

---

### Download Kaggle Disaster Datasets

Before running the pipeline, you can pre-download historical disaster data:

```powershell
python main.py --download-kaggle
```

**Requirements:**
- `KAGGLE_USERNAME` and `KAGGLE_KEY` must be set in `.env`
- The `kaggle` package must be installed (included in `requirements.txt`)

This downloads and merges flood, drought, and wildfire datasets from Kaggle into `data/kaggle/`.

---

### Generate Sample Data Only

If you just want to create the synthetic sample dataset without running the pipeline:

```powershell
python main.py --generate-data
```

This produces `data/sample_churn_data.csv` — a synthetic SaaS customer churn dataset used when no NASA or custom data is available.

---

### Use a Custom Dataset

Point the pipeline at any CSV, Parquet, or JSON file:

```powershell
python main.py --data path/to/your/data.csv --target your_target_column
```

Example — classify wildfire risk using your own data:

```powershell
python main.py --data data/wildfire_data.csv --target fire_risk --task-type classification
```

---

## Command-Line Reference

```
python main.py [OPTIONS]

Options:
  --data PATH           Path to a CSV, Parquet, or JSON dataset
  --config PATH         Pipeline config YAML (default: config/pipeline_config.yaml)
  --target COLUMN       Target column name (overrides config)
  --task-type TYPE      classification | regression (overrides config)
  --dashboard           Launch the web dashboard after the pipeline completes
  --generate-data       Generate the built-in synthetic sample dataset and exit
  --download-kaggle     Download Kaggle disaster datasets and exit
  --scheduler           Start the hourly NASA data refresh scheduler
  --provider PROVIDER   Force LLM provider: openai | anthropic | gemini | ollama | rule_based
  --output DIR          Output directory (overrides config; default: output/)
  --verbose, -v         Enable DEBUG-level logging
  -h, --help            Show this help message
```

### Usage Examples

```powershell
# 1. Most basic — let everything auto-detect
python main.py

# 2. Run pipeline and open dashboard
python main.py --dashboard

# 3. Use live NASA data + dashboard + hourly scheduler
python main.py --dashboard --scheduler

# 4. Force Gemini as LLM, use live NASA data
python main.py --provider gemini

# 5. Analyze your own dataset
python main.py --data my_data.csv --target churn --task-type classification

# 6. Verbose debug output
python main.py --verbose

# 7. Use rule-based AI (no LLM needed, fully offline after install)
python main.py --provider rule_based

# 8. Download Kaggle data first, then run main pipeline
python main.py --download-kaggle
python main.py --dashboard
```

---

## LLM Provider Setup

### Option A: OpenAI

1. Sign up at https://platform.openai.com
2. Create an API key under **API keys**
3. Add to `.env`:
   ```env
   OPENAI_API_KEY=sk-proj-...
   ```
4. Default model: `gpt-4o-mini`. To use GPT-4o:
   ```env
   LLM_MODEL=gpt-4o
   ```

---

### Option B: Anthropic (Claude)

1. Sign up at https://console.anthropic.com
2. Create an API key
3. Add to `.env`:
   ```env
   ANTHROPIC_API_KEY=sk-ant-...
   ```
4. Default model: `claude-sonnet-4-20250514`

---

### Option C: Google Gemini

1. Sign up at https://ai.google.dev
2. Create an API key in Google AI Studio
3. Install the package:
   ```powershell
   pip install google-generativeai>=0.5
   ```
4. Add to `.env`:
   ```env
   GOOGLE_API_KEY=AIza...
   ```
5. Default model: `gemini-2.0-flash`

---

### Option D: Ollama (Local, Free, No Internet Required for Inference)

1. Download and install Ollama from https://ollama.ai
2. Pull a model (e.g., Llama 3.1):
   ```powershell
   ollama pull llama3.1
   ```
3. Ensure Ollama is running (it starts automatically on Windows after install)
4. Add to `.env` (optional, this is the default):
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   ```

The system auto-detects the best available Ollama model from this priority list: `llama3.1` → `llama3` → `mistral` → `qwen2.5` → `codellama` → `deepseek-coder` → `phi3`

---

### Option E: Rule-Based Fallback (No LLM Required)

No setup needed. Run:

```powershell
python main.py --provider rule_based
```

The built-in heuristic engine produces deterministic responses for all agent prompts — cleaning plans, feature recommendations, model selection, and insight generation — without any external API call. This is ideal for testing the pipeline structure or running in air-gapped environments.

---

## Pipeline Stages Explained

### Stage 1 — Data Collection Agent
**File:** `agents/data_collection_agent.py`

- Detects the data source type from config (`nasa_api` or `file`)
- For NASA sources: fetches from **FIRMS global** (20 regions), **EONET** (last 30 days), and loads Kaggle training data
- Validates schema, logs missing-value warnings
- Stores the raw merged DataFrame in shared `MemoryStore`

---

### Stage 2 — Data Cleaning Agent
**File:** `agents/data_cleaning_agent.py`

- Asks the LLM to generate a JSON cleaning plan tailored to the dataset
- Executes the plan: duplicate removal, median/mode imputation, IQR outlier capping
- Each operation is sandboxed and rolled back if it causes data loss
- Reports before/after statistics

---

### Stage 3 — EDA Agent
**File:** `agents/eda_agent.py`

- Computes distributions, correlations, skewness, and cardinality for all columns
- Generates Plotly/Seaborn charts saved to `output/plots/`
- Runs geo-temporal analysis — fires by latitude/longitude cluster and acquisition time
- Produces a LLM-generated narrative summarizing findings and hypotheses

---

### Stage 4 — Feature Engineering Agent
**File:** `agents/feature_engineering_agent.py`

- Log-transforms skewed numeric features (FRP, brightness, precipitation)
- Generates derived features: drought severity index, flood risk score, soil moisture class
- Encodes categorical columns (satellite, day/night, region) using label/one-hot encoding
- Performs mutual-information-based feature selection to reduce noise

---

### Stage 5 — Modeling Agent
**File:** `agents/modeling_agent.py`

- Asks LLM to recommend model candidates (XGBoost, LightGBM, Random Forest, Logistic Regression)
- Trains and cross-validates all candidates (5-fold stratified CV)
- Runs Optuna hyperparameter search on the best candidate
- Computes SHAP values for global feature importance
- Saves the trained model to `output/model.pkl`

---

### Stage 6 — Insight Agent
**File:** `agents/insight_agent.py`

- Synthesizes outputs from all prior stages through the LLM
- Generates an executive summary, risk assessment, and prioritized recommendations
- Creates a model card documenting performance, limitations, and intended use
- Saves `output/insights.json` and `output/report.html`

---

## Data Sources

### NASA FIRMS (Fire Information for Resource Management System)
- **API:** `https://firms.modaps.eosdis.nasa.gov/api/area/csv/`
- **Sensor:** VIIRS SNPP NRT (Visible Infrared Imaging Radiometer Suite, 375 m resolution)
- **Requires:** Free NASA FIRMS API key
- **Covered regions (20+):** India North/South/East/West, East Africa, West Africa, Southern Africa, Brazil Central/Amazon, Indonesia, Thailand, Vietnam, California, Southeast USA, Northern Territory Australia, Southeast Australia, Mediterranean Europe, Central America, Middle East, Southern China

### NASA EONET (Earth Observatory Natural Event Tracker)
- **API:** `https://eonet.gsfc.nasa.gov/api/v3/events`
- **Requires:** No API key
- **Data:** Wildfires, floods, severe storms, droughts — last 30 days, up to 100 events

### NASA POWER (Prediction of Worldwide Energy Resources)
- **API:** `https://power.larc.nasa.gov/api/temporal/daily/point`
- **Requires:** No API key
- **Parameters fetched:** T2M (temperature), T2M_MAX, T2M_MIN, PRECTOTCORR (precipitation), RH2M (humidity), WS2M (wind speed), ALLSKY_SFC_SW_DWN (solar radiation), GWETROOT (soil moisture)

### Kaggle Disaster Datasets (Optional)
Managed by `data/kaggle_downloader.py`. Merges multiple historical datasets:
- Flood records with severity indicators
- Drought data with PDSI indices
- Wildfire occurrences with acreage burned
- Crop impact datasets

### Built-in Fallback
When no API key is present, `api_tools.py` serves a fully enriched, deterministic global sample dataset covering all 20 regions — no internet connection required.

---

## Configuration File

`config/pipeline_config.yaml` controls the pipeline behavior:

```yaml
# What problem to solve
problem_statement: |
  Predict fire-detection confidence using NASA satellite telemetry.

# Background knowledge for the LLM agents
domain_context: |
  Global Environmental Resilience & Crop Protection Command Center.

# Target column and task type
target_column: "confidence"    # values: l (low), n (nominal), h (high)
task_type: "classification"    # or "regression"

# Data source
sources:
  - source_type: "nasa_api"
    map_key: "VIIRS_SNPP_NRT"
    area_coords: "world"

# LLM settings
llm:
  provider: "auto"     # auto | openai | anthropic | gemini | ollama | rule_based
  model: ""            # Leave blank to use provider default
  temperature: 0.1
  max_tokens: 4096

# Output directory
output_dir: "output"

# Quality gates config
quality_gates_path: "config/quality_gates.yaml"
```

To change to regression (e.g., predict fire radiative power):

```yaml
target_column: "frp"
task_type: "regression"
```

---

## Output Files

After a successful run, the `output/` directory contains:

| File/Folder | Description |
|---|---|
| `model.pkl` | Serialized trained ML model |
| `insights.json` | Full structured output from the Insight Agent |
| `report.html` | Human-readable HTML report |
| `metrics.json` | Model cross-validation scores, confusion matrix |
| `plots/` | Folder of Plotly/Seaborn chart HTML and PNG files |
| `shap_summary.png` | SHAP beeswarm plot showing feature importance |
| `feature_importance.json` | Ranked feature importances |

---

## Dashboard Features

Start the dashboard with:

```powershell
python main.py --dashboard
```

Then open **http://localhost:5000** in your browser.

| Section | Description |
|---|---|
| **Overview** | Pipeline run summary, dataset shape, model accuracy |
| **Global Map** | Interactive Plotly world map with fire detections color-coded by confidence |
| **Feature Importance** | SHAP value waterfall and beeswarm charts |
| **Insights** | LLM-generated executive summary and recommendations |
| **Scheduler** | Live status showing last refresh time and next scheduled refresh |
| **Data Explorer** | Sortable/filterable table of raw predictions |

The dashboard reads pre-computed results from `output/` — it is stateless and can be restarted independently of the pipeline.

---

## Scheduler

The background scheduler (`scheduler.py`) runs on a configurable interval (default: **1 hour**) and:

1. Calls `fetch_nasa_firms_global()` for all 20 regions
2. Calls `fetch_nasa_eonet()` for the last 30 events
3. Calls `fetch_nasa_power_global()` for weather data across global region centroids
4. Saves timestamped CSVs to `data/live_cache/`
5. Always maintains a `*_latest.csv` symlink for the most recent fetch
6. Automatically cleans up old cache files (keeps last 24 cycles per source)

Start the scheduler standalone (without running the full pipeline):

```powershell
# Run pipeline first, then keep refreshing in background
python main.py --scheduler
```

Or as part of the full flow:

```powershell
python main.py --dashboard --scheduler
```

---

## Testing

Run the test suite:

```powershell
python -m pytest tests/ -v
```

Quick API connectivity tests (no keys required):

```powershell
python tmp_test_api.py
python tmp_test_api_v3.py
```

---

## Troubleshooting

### ❌ `Unable to run it on VS_Code`

This may be caused due to the Python version or the virtual environment not being activated or version clash issues.

Open the file in Google Antigravity and prompt - "run the program and show the webpage" for the webpage and "update the database from nasa api" for the database updation (the data updated depends on the data stored in the NASA servers which may be changed once an hour to once a week) 


---

### ❌ `ModuleNotFoundError: No module named 'yaml'`

Your virtual environment is not activated, or dependencies are not installed.

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

### ❌ `ModuleNotFoundError: No module named 'dotenv'`

```powershell
pip install python-dotenv
```

---

### ❌ NASA FIRMS returns empty data / falls back to sample

- Verify `NASA_API_KEY` is set correctly in `.env`
- Get a free key at: https://firms.modaps.eosdis.nasa.gov/api/area/
- Confirm the key is active — there may be a short provisioning delay after creation
- The pipeline still works without the key using the built-in enriched fallback dataset

---

### ❌ Ollama connection refused / LLM falling back to rule-based

- Make sure Ollama is installed: https://ollama.ai
- On Windows, Ollama runs as a background service — check the taskbar
- Pull a model if none are installed:
  ```powershell
  ollama pull llama3.1
  ```
- Verify Ollama is reachable:
  ```powershell
  curl http://localhost:11434/api/tags
  ```

---

### ❌ `UnicodeEncodeError` on Windows terminal

The system forces UTF-8 output internally. If you still see encoding errors, set:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python main.py
```

---

### ❌ Dashboard shows "No data found"

Run the pipeline at least once before opening the dashboard:

```powershell
python main.py          # generate output/
python main.py --dashboard   # then launch dashboard
```

---

### ❌ Kaggle download fails

1. Make sure `KAGGLE_USERNAME` and `KAGGLE_KEY` are in `.env`
2. Accept the dataset license on the Kaggle website for each dataset
3. Test your credentials:
   ```powershell
   kaggle datasets list
   ```

---

### ❌ `optuna` or `shap` installation fails on Windows

Try installing binary wheels:

```powershell
pip install optuna shap --only-binary :all:
```

If that doesn't work, install Visual C++ Build Tools from:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

---

## Frequently Asked Questions

**Q: Do I need an API key to run this?**
A: No. Without any API keys, the pipeline uses a built-in global dataset and a rule-based AI engine. Everything works end-to-end offline after `pip install`.

**Q: How do I get a free NASA FIRMS key?**
A: Register at https://firms.modaps.eosdis.nasa.gov/api/area/ — it's instant and free.

**Q: Can I use this on my own dataset unrelated to wildfires?**
A: Yes. Use `--data your_file.csv --target your_column --task-type classification`. Adjust `target_column` and `task_type` in the YAML config for persistent settings.

**Q: The pipeline takes too long. How do I speed it up?**
A: Use `--provider rule_based` to skip LLM calls. Optuna hyperparameter search is the slowest step — reduce trials by editing `n_trials` in `agents/modeling_agent.py`. Use a smaller dataset with `--data`.

**Q: Where does the model get saved?**
A: `output/model.pkl`. Load it with `import pickle; model = pickle.load(open('output/model.pkl', 'rb'))`.

**Q: Can I run this on a server / headless machine?**
A: Yes. The dashboard binds to `0.0.0.0:5000` so it is accessible from other machines on the network. For headless operation without a dashboard, just run `python main.py`.

**Q: Why is the confidence prediction accuracy around 60–70%?**
A: Fire detection confidence (`l`/`n`/`h`) is a noisy label derived from satellite sensor thresholds. The model learns real-world signal but the ceiling accuracy for this 3-class problem on satellite telemetry is naturally limited by measurement uncertainty.

---

## License

This project is provided for research and educational purposes.

---

## Acknowledgements

- **NASA FIRMS** — Fire Information for Resource Management System
- **NASA EONET** — Earth Observatory Natural Event Tracker
- **NASA POWER** — Prediction of Worldwide Energy Resources
- **Kaggle** — Open disaster and climate datasets
- **SHAP** — SHapley Additive exPlanations (Lundberg & Lee)
- **Optuna** — Hyperparameter optimization framework

---

*Built with ❤️ — Geo Caster - Autonomous Analysis System*
