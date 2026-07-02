# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-page Streamlit app (`app.py`) that predicts FIFA World Cup 2026 match outcomes. The user enters two team names; the app queries historical player/team stats from Databricks Unity Catalog, builds a 79-feature vector per team, and calls a Databricks ML serving endpoint to get win probabilities, which are normalized and displayed head-to-head.

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app requires Databricks credentials to be configured (via `databricks-sdk`'s standard auth resolution — e.g. `DATABRICKS_HOST`/`DATABRICKS_TOKEN` env vars or a configured CLI profile) since it makes live calls to a Databricks workspace. There are no local tests, lint config, or build step in this repo.

In production it runs as a Databricks App, launched per `app.yaml` via `streamlit run app.py --server.port $DATABRICKS_APP_PORT --server.headless true`.

## Architecture

Everything lives in `app.py`. The flow per comparison is:

1. **`get_team_features(team_name)`** — runs a SQL aggregation against `workspace.fifa_wc_gold.player_performance_ml` (via `w.statement_execution.execute_statement`, polling for completion), averaging/summing player stats for the team. It then hand-builds a fixed 79-key feature dict (`features`) that the ML model expects — team-level aggregates (age, market value, xG, pass accuracy, etc.) are mapped onto per-player-shaped fields, and many fields the query doesn't produce (crosses, aerial duels, sprint distance, etc.) are hardcoded placeholder constants. Any missing/null SQL value falls back to a hardcoded default (e.g. `avg_age = 28.0`).
2. **`convert_features_to_numeric(features_dict)`** — coerces the feature dict to numeric types before sending to the model; `position` and `preferred_foot` are mapped from strings to ints via fixed lookup tables.
3. **`predict_endpoint(endpoint_name, features_dict)`** — calls the `fifa-team-win-predictor` Databricks serving endpoint (`w.serving_endpoints.query`) with the numeric feature row and returns the raw prediction.
4. The two teams' raw predictions are normalized against each other (`prob_a / (prob_a + prob_b)`) since the model doesn't produce a naturally paired probability, then rendered with Streamlit metrics/progress bars and a favorite/underdog verdict based on the margin.

Key coupling to be aware of when editing: the feature dict keys and count (79) in `get_team_features` must exactly match what the `fifa-team-win-predictor` model endpoint expects — the SQL query, the hardcoded warehouse ID (`f9bb0b517b9fc8ba`), the Unity Catalog table name, and the model's feature schema are all implicitly coupled and live only in this one file.

Note: team name lookups are done via raw f-string SQL interpolation (`WHERE LOWER(team) = LOWER('{team_name}')`), not parameterized queries.
