# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Streamlit app (Databricks App) that predicts FIFA World Cup 2026 match outcomes with a match-level LightGBM model served from Databricks Model Serving. The user picks two teams; the app builds 26 pre-match features (dynamic Elo, last-5 form, head-to-head, squad quality) from state tables in Unity Catalog and shows calibrated-ish win/draw/loss probabilities with explanatory visuals. **The Kaggle dataset is synthetic — predictions hover near chance; this is documented in-app and is a data limitation, not a bug.**

## Commands

```bash
# Tests (12, pure logic — no network)
.venv/Scripts/python.exe -m pytest tests -q

# Run app locally (uses local model instead of the serving endpoint)
PYTHONUTF8=1 MODEL_LOCAL_URI="models:/workspace.fifa_wc_gold.fifa_team_win_predictor/5" \
  .venv/Scripts/python.exe -m streamlit run app.py

# Retrain + register a new model version (LOCAL ONLY — see constraint below)
PYTHONUTF8=1 .venv/Scripts/python.exe training/train_local.py
```

Always set `PYTHONUTF8=1` on Windows — MLflow prints emoji that crash cp1252.

## Hard infrastructure constraint

**Serverless jobs in this workspace (Free Edition) cannot read or write Unity Catalog model artifacts** (S3 explicit deny). Model training/registration must run locally (`training/train_local.py`, `training/register_v5.py`) via the `.venv`; data moves through SQL Warehouse `f9bb0b517b9fc8ba`. When registering models locally: pin `python=3.12` in `conda_env` (serving doesn't support 3.14) and pass `skops_trusted_types`.

## Architecture

- **`fifa_features.py`** — the contract between training and serving: `FEATURE_COLUMNS` order and `build_feature_row()` must produce identical rows in both. Any feature change requires retraining AND redeploying together.
- **`app.py`** — Streamlit UI. Reads `team_state_current` + `h2h_state` (cached 10 min), builds features, queries the endpoint (`MODEL_ENDPOINT` env) or a local model (`MODEL_LOCAL_URI` env). Predictions are symmetrized: A-vs-B and mirrored B-vs-A averaged, so selection order doesn't matter. Charts are hand-rolled HTML/CSS (palette: blue #2a78d6 team A, orange #eb6834 team B, gray draw).
- **`app_helpers.py`** — pure logic extracted for testability (input dtypes, mirror-averaging, friendly errors, verdicts). Model signature has 4 `long` columns (`INT_COLS`) — dtype mismatches fail schema enforcement at the endpoint.
- **`training/train_local.py`** — full pipeline: SQL aggregation → per-match features built chronologically (windows strictly exclude the current match — the original model's fatal flaw was score leakage) → temporal split 70/10/20 → LogReg/XGB/LGBM tuning on TimeSeriesSplit → MLflow → UC registration → writes the state tables.
- **`training/train_match_predictor.py`** — same pipeline as a Databricks notebook; kept for reference but **cannot register models** (see constraint).
- **Model lineage**: v1 = old leaky player-level model; v4 = match-level + Platt (calibration hurt: 105-row cal set); **v5 = match-level uncalibrated, the production candidate**. Metrics in `training/phase1_results.json`.

## Deployment

`deploy/app_resources.json` (endpoint CAN_QUERY + warehouse CAN_USE via `databricks apps update`), `deploy/grants.sql` (SELECT on state tables for the app's service principal `13d449ab-c727-44b6-88f9-89eb887379a0`), then `databricks sync` to the workspace folder and `databricks apps deploy`. `app.yaml` injects `MODEL_ENDPOINT`/`WAREHOUSE_ID` from the declared resources via `valueFrom`.
