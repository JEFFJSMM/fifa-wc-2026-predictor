"""Entrena el modelo de predicción a nivel partido desde la máquina local.

Ejecuta el mismo pipeline que training/train_match_predictor.py (notebook),
pero corre localmente porque el compute serverless de jobs de este workspace
tiene denegado el acceso al almacenamiento de modelos de Unity Catalog.
Los datos se leen/escriben vía SQL Warehouse y el modelo se registra en UC.
"""
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
import lightgbm
import xgboost
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROFILE = "jsierram96"
WAREHOUSE_ID = "f9bb0b517b9fc8ba"
CATALOG, SCHEMA_GOLD = "workspace", "fifa_wc_gold"
GOLD_TABLE = f"{CATALOG}.{SCHEMA_GOLD}.player_performance_ml"
MODEL_UC_NAME = f"{CATALOG}.{SCHEMA_GOLD}.fifa_team_win_predictor"
EXPERIMENT = "/Users/jsierram96@gmail.com/fifa-wc-2026-match-model"
ENDPOINT_OLD = "fifa-team-win-predictor"
SEED = 42
POINTS = {"W": 3, "D": 1, "L": 0}
ELO_K = 32
FORM_N = 5

w = WorkspaceClient(profile=PROFILE)
mlflow.set_tracking_uri(f"databricks://{PROFILE}")
mlflow.set_registry_uri(f"databricks-uc://{PROFILE}")
mlflow.set_experiment(EXPERIMENT)


def run_sql(stmt, timeout=300):
    st = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=stmt, wait_timeout="50s")
    waited = 0
    while st.status.state in (StatementState.PENDING, StatementState.RUNNING) and waited < timeout:
        time.sleep(3)
        waited += 3
        st = w.statement_execution.get_statement(st.statement_id)
    if st.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL falló: {st.status}")
    cols = [c.name for c in st.manifest.schema.columns]
    rows = list(st.result.data_array or [])
    for i in range(1, st.manifest.total_chunk_count):
        chunk = w.statement_execution.get_statement_result_chunk_n(st.statement_id, i)
        rows.extend(chunk.data_array or [])
    return pd.DataFrame(rows, columns=cols)


print("1/8 Leyendo agregados equipo-partido desde el warehouse...")
team_match = run_sql(f"""
    SELECT match_id, CAST(match_date AS STRING) AS match_date, team, opponent_team,
        MAX(match_result) AS result,
        MAX(CAST(goals_team AS DOUBLE)) AS goals_for,
        MAX(CAST(goals_opponent AS DOUBLE)) AS goals_against,
        SUM(CAST(expected_goals_xg AS DOUBLE)) AS xg,
        SUM(CAST(shots AS DOUBLE)) AS shots,
        AVG(CAST(player_rating AS DOUBLE)) AS rating,
        AVG(CAST(age AS DOUBLE)) AS squad_age,
        AVG(CAST(height_cm AS DOUBLE)) AS squad_height,
        AVG(CAST(market_value_eur AS DOUBLE)) AS squad_value,
        MAX(tournament_stage) AS stage
    FROM {GOLD_TABLE}
    GROUP BY match_id, match_date, team, opponent_team
""")
for c in ["goals_for", "goals_against", "xg", "shots", "rating", "squad_age", "squad_height", "squad_value"]:
    team_match[c] = pd.to_numeric(team_match[c])
team_match["match_date"] = pd.to_datetime(team_match["match_date"])
team_match = team_match.sort_values("match_date").reset_index(drop=True)
assert team_match.groupby("match_id").size().eq(2).all()
print(f"   {len(team_match)} filas equipo-partido, {team_match['match_id'].nunique()} partidos")

# ---------------------------------------------------------------- features
print("2/8 Construyendo features pre-partido (Elo, forma, descanso, H2H)...")
matches = (team_match.sort_values(["match_date", "match_id"])
           .groupby("match_id", sort=False).agg(list).reset_index())

elo, team_hist, h2h, last_played = {}, {}, {}, {}
rng = np.random.default_rng(SEED)
rows = []


def team_state(team, date):
    hist = team_hist.get(team, [])
    recent = hist[-FORM_N:]
    return {
        "elo": elo.get(team, 1500.0),
        "matches_played": len(hist),
        "winrate": (np.mean([h["won"] for h in hist]) if hist else 0.5),
        "rest_days": ((date - last_played[team]).days if team in last_played else 7.0),
        "form5_ppg": (np.mean([h["pts"] for h in recent]) if recent else 1.33),
        "form5_gf": (np.mean([h["gf"] for h in recent]) if recent else 1.3),
        "form5_ga": (np.mean([h["ga"] for h in recent]) if recent else 1.3),
        "form5_xg": (np.mean([h["xg"] for h in recent]) if recent else 1.3),
        "form5_rating": (np.mean([h["rating"] for h in recent]) if recent else 7.0),
        "form5_shots": (np.mean([h["shots"] for h in recent]) if recent else 12.0),
    }


def build_feature_row(sa, sb, h2h_stats, stage_knockout):
    return {
        "elo_a": sa["elo"], "elo_b": sb["elo"], "elo_diff": sa["elo"] - sb["elo"],
        "form5_ppg_a": sa["form5_ppg"], "form5_ppg_b": sb["form5_ppg"],
        "form5_ppg_diff": sa["form5_ppg"] - sb["form5_ppg"],
        "form5_gf_diff": sa["form5_gf"] - sb["form5_gf"],
        "form5_ga_diff": sa["form5_ga"] - sb["form5_ga"],
        "form5_xg_diff": sa["form5_xg"] - sb["form5_xg"],
        "form5_rating_diff": sa["form5_rating"] - sb["form5_rating"],
        "form5_shots_diff": sa["form5_shots"] - sb["form5_shots"],
        "winrate_a": sa["winrate"], "winrate_b": sb["winrate"],
        "winrate_diff": sa["winrate"] - sb["winrate"],
        "rest_days_a": sa["rest_days"], "rest_days_b": sb["rest_days"],
        "rest_diff": sa["rest_days"] - sb["rest_days"],
        "matches_played_a": sa["matches_played"], "matches_played_b": sb["matches_played"],
        "h2h_matches": h2h_stats["n"], "h2h_winrate_a": h2h_stats["winrate_a"],
        "h2h_gd_a": h2h_stats["gd_a"],
        "squad_value_diff": sa["squad_value"] - sb["squad_value"],
        "squad_age_diff": sa["squad_age"] - sb["squad_age"],
        "squad_height_diff": sa["squad_height"] - sb["squad_height"],
        "stage_knockout": stage_knockout,
    }


for _, m in matches.iterrows():
    date, stage = m["match_date"][0], m["stage"][0]
    stage_knockout = 0 if str(stage).lower().startswith("group") else 1
    idx = [0, 1] if rng.random() < 0.5 else [1, 0]
    ta, tb = m["team"][idx[0]], m["team"][idx[1]]
    res_a = m["result"][idx[0]]
    gf_a, ga_a = m["goals_for"][idx[0]], m["goals_against"][idx[0]]

    sa, sb = team_state(ta, date), team_state(tb, date)
    for s, i in ((sa, idx[0]), (sb, idx[1])):
        s["squad_value"] = m["squad_value"][i]
        s["squad_age"] = m["squad_age"][i]
        s["squad_height"] = m["squad_height"][i]

    pair = frozenset((ta, tb))
    h = h2h.get(pair, {"n": 0, ta: 0, tb: 0, "draws": 0, ("gd", ta): 0.0, ("gd", tb): 0.0})
    h2h_stats = {"n": h["n"],
                 "winrate_a": (h.get(ta, 0) / h["n"]) if h["n"] else 0.5,
                 "gd_a": (h.get(("gd", ta), 0.0) / h["n"]) if h["n"] else 0.0}

    row = build_feature_row(sa, sb, h2h_stats, stage_knockout)
    row.update({"match_id": m["match_id"], "match_date": date, "team_a": ta, "team_b": tb,
                "target": 0 if res_a == "W" else (1 if res_a == "D" else 2)})
    rows.append(row)

    ra, rb = elo.get(ta, 1500.0), elo.get(tb, 1500.0)
    exp_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
    score_a = 1.0 if res_a == "W" else (0.5 if res_a == "D" else 0.0)
    elo[ta] = ra + ELO_K * (score_a - exp_a)
    elo[tb] = rb + ELO_K * ((1.0 - score_a) - (1.0 - exp_a))

    for t, i in ((ta, idx[0]), (tb, idx[1])):
        team_hist.setdefault(t, []).append({
            "won": 1.0 if m["result"][i] == "W" else 0.0, "pts": POINTS[m["result"][i]],
            "gf": m["goals_for"][i], "ga": m["goals_against"][i],
            "xg": m["xg"][i], "rating": m["rating"][i], "shots": m["shots"][i]})
        last_played[t] = date

    h["n"] += 1
    if res_a == "D":
        h["draws"] += 1
    else:
        winner = ta if res_a == "W" else tb
        h[winner] = h.get(winner, 0) + 1
    h[("gd", ta)] = h.get(("gd", ta), 0.0) + (gf_a - ga_a)
    h[("gd", tb)] = h.get(("gd", tb), 0.0) + (ga_a - gf_a)
    h2h[pair] = h

df = pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)
_dummy = team_state("_", pd.Timestamp("2026-01-01")) | {"squad_value": 0, "squad_age": 0, "squad_height": 0}
FEATURES = list(build_feature_row(_dummy, _dummy, {"n": 0, "winrate_a": 0.5, "gd_a": 0.0}, 0).keys())
print(f"   dataset: {len(df)} partidos, {len(FEATURES)} features")

# ---------------------------------------------------------------- split
n = len(df)
i_train, i_cal = int(n * 0.70), int(n * 0.80)
train, cal, test = df.iloc[:i_train], df.iloc[i_train:i_cal], df.iloc[i_cal:]
X_tr, y_tr = train[FEATURES], train["target"]
X_cal, y_cal = cal[FEATURES], cal["target"]
X_te, y_te = test[FEATURES], test["target"]
print(f"3/8 Split temporal: train={len(train)} cal={len(cal)} test={len(test)}")

# ---------------------------------------------------------------- selección
print("4/8 Comparando algoritmos con tuning temporal (MLflow)...")
tscv = TimeSeriesSplit(n_splits=3)
candidates = {
    "logistic_regression": (
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=3000, random_state=SEED))]),
        {"clf__C": np.logspace(-3, 2, 12)}, 12),
    "xgboost": (
        XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                      random_state=SEED, n_jobs=-1, verbosity=0),
        {"n_estimators": [100, 200, 300, 500], "max_depth": [2, 3, 4, 5],
         "learning_rate": [0.01, 0.03, 0.05, 0.1], "subsample": [0.6, 0.8, 1.0],
         "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 5, 10],
         "reg_lambda": [0.1, 1.0, 10.0]}, 20),
    "lightgbm": (
        LGBMClassifier(objective="multiclass", random_state=SEED, n_jobs=-1, verbose=-1),
        {"n_estimators": [100, 200, 300, 500], "max_depth": [2, 3, 4, 5],
         "learning_rate": [0.01, 0.03, 0.05, 0.1], "num_leaves": [7, 15, 31],
         "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
         "min_child_samples": [10, 30, 50], "reg_lambda": [0.1, 1.0, 10.0]}, 20),
}
results = {}
with mlflow.start_run(run_name="match_model_selection_local") as parent:
    for name, (estimator, grid, n_iter) in candidates.items():
        with mlflow.start_run(run_name=name, nested=True):
            search = RandomizedSearchCV(estimator, grid, n_iter=n_iter, cv=tscv,
                                        scoring="neg_log_loss", random_state=SEED,
                                        n_jobs=-1, refit=True)
            search.fit(X_tr, y_tr)
            cv_ll = -search.best_score_
            mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
            mlflow.log_metric("cv_log_loss", cv_ll)
            results[name] = {"model": search.best_estimator_, "cv_log_loss": cv_ll,
                             "params": search.best_params_}
            print(f"   {name}: CV log-loss = {cv_ll:.4f}")
    best_name = min(results, key=lambda k: results[k]["cv_log_loss"])
    best_model = results[best_name]["model"]
    mlflow.log_param("selected_algorithm", best_name)
    mlflow.log_metric("selected_cv_log_loss", results[best_name]["cv_log_loss"])
    parent_run_id = parent.info.run_id
print(f"   🏆 mejor algoritmo: {best_name}")

# ---------------------------------------------------------------- calibración
print("5/8 Calibrando (Platt) y evaluando en test...")


def ece_binary(y_true, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.sum():
            ece += mask.mean() * abs(y_true[mask].mean() - p[mask].mean())
    return ece


def multiclass_brier(y_true, proba):
    onehot = np.eye(proba.shape[1])[y_true]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def eval_3class(proba, y):
    y = np.asarray(y)
    pred = proba.argmax(axis=1)
    p_win_a = proba[:, 0] / (proba[:, 0] + proba[:, 2] + 1e-12)
    nd = y != 1
    yb = (y[nd] == 0).astype(int)
    return {
        "accuracy_3class": float(accuracy_score(y, pred)),
        "log_loss_3class": float(log_loss(y, proba, labels=[0, 1, 2])),
        "brier_3class": multiclass_brier(y, proba),
        "accuracy_binary_nondraw": float(accuracy_score(yb, (p_win_a[nd] > 0.5).astype(int))),
        "log_loss_binary_nondraw": float(log_loss(yb, np.clip(p_win_a[nd], 1e-6, 1 - 1e-6), labels=[0, 1])),
        "brier_binary_nondraw": float(brier_score_loss(yb, p_win_a[nd])),
        "ece_binary_nondraw": float(ece_binary(yb, p_win_a[nd])),
    }


try:
    from sklearn.frozen import FrozenEstimator
    calibrated = CalibratedClassifierCV(FrozenEstimator(best_model), method="sigmoid")
    calibrated.fit(X_cal, y_cal)
except ImportError:
    calibrated = CalibratedClassifierCV(best_model, method="sigmoid", cv="prefit")
    calibrated.fit(X_cal, y_cal)

metrics_uncal = eval_3class(best_model.predict_proba(X_te), y_te)
metrics_cal = eval_3class(calibrated.predict_proba(X_te), y_te)

# ---------------------------------------------------------------- modelo viejo
print("6/8 Evaluando el modelo viejo contra el endpoint de producción...")
app_agg = run_sql(f"""
    SELECT team,
        AVG(CAST(age AS DOUBLE)) avg_age, AVG(CAST(height_cm AS DOUBLE)) avg_height,
        AVG(CAST(weight_kg AS DOUBLE)) avg_weight, AVG(CAST(market_value_eur AS DOUBLE)) avg_market_value,
        AVG(CAST(minutes_played AS DOUBLE)) avg_minutes, AVG(CAST(shots AS DOUBLE)) avg_shots,
        AVG(CAST(shots_on_target AS DOUBLE)) avg_shots_on_target, AVG(CAST(expected_goals_xg AS DOUBLE)) avg_xg,
        AVG(CAST(expected_assists_xa AS DOUBLE)) avg_xa, AVG(CAST(successful_passes AS DOUBLE)) avg_passes,
        AVG(CAST(pass_accuracy AS DOUBLE)) avg_pass_acc, AVG(CAST(dribbles_attempted AS DOUBLE)) avg_dribbles,
        AVG(CAST(successful_dribbles AS DOUBLE)) avg_succ_dribbles, AVG(CAST(tackles AS DOUBLE)) avg_tackles,
        AVG(CAST(interceptions AS DOUBLE)) avg_interceptions, AVG(CAST(player_rating AS DOUBLE)) avg_rating,
        AVG(CAST(performance_score AS DOUBLE)) avg_performance, COUNT(*) games_played,
        SUM(CAST(goals AS INT)) total_goals, SUM(CAST(assists AS INT)) total_assists,
        SUM(CASE WHEN match_result = 'W' THEN 1 ELSE 0 END) total_wins,
        AVG(CAST(goals_team AS DOUBLE)) avg_goals_per_game
    FROM {GOLD_TABLE} GROUP BY team
""").set_index("team")
for c in app_agg.columns:
    app_agg[c] = pd.to_numeric(app_agg[c])


def app_features_for_team(team):
    r = app_agg.loc[team]
    games_played = int(r["games_played"])
    total_goals, total_assists, total_wins = int(r["total_goals"]), int(r["total_assists"]), int(r["total_wins"])
    avg_pass_acc, avg_shots = float(r["avg_pass_acc"]), float(r["avg_shots"])
    avg_rating, avg_xa = float(r["avg_rating"]), float(r["avg_xa"])
    avg_market_value = float(r["avg_market_value"])
    win_rate = total_wins / games_played if games_played else 0.5
    return {
        "position": 3, "age": float(r["avg_age"]), "jersey_number": 10,
        "height_cm": int(r["avg_height"]), "weight_kg": int(r["avg_weight"]), "preferred_foot": 1,
        "market_value_eur": avg_market_value / 11, "goals_team": int(r["avg_goals_per_game"]),
        "goals_opponent": 1, "minutes_played": int(r["avg_minutes"]),
        "assists": int(total_assists / games_played) if games_played else 0,
        "shots": int(avg_shots), "shots_on_target": int(r["avg_shots_on_target"]),
        "expected_goals_xg": float(r["avg_xg"]), "expected_assists_xa": avg_xa,
        "key_passes": int(r["avg_passes"] * 0.1), "successful_passes": int(r["avg_passes"]),
        "total_passes": int(r["avg_passes"] / (avg_pass_acc / 100)) if avg_pass_acc > 0 else int(r["avg_passes"]),
        "pass_accuracy": avg_pass_acc, "dribbles_attempted": int(r["avg_dribbles"]),
        "successful_dribbles": int(r["avg_succ_dribbles"]), "crosses": 15, "successful_crosses": 5,
        "tackles": int(r["avg_tackles"]), "interceptions": int(r["avg_interceptions"]), "clearances": 15,
        "blocks": 8, "aerial_duels_won": 20, "aerial_duels_lost": 15, "recoveries": 25,
        "defensive_actions": int(r["avg_tackles"] + r["avg_interceptions"]), "fouls_committed": 12,
        "fouls_suffered": 10, "yellow_cards": 2, "red_cards": 0, "offsides": 3, "saves": 0,
        "save_percentage": 0.0, "punches": 0, "clean_sheet": 0, "goals_conceded": 1,
        "penalty_saves": 0, "distance_covered_km": 110.0, "sprint_distance_km": 15.0,
        "top_speed_kmh": 32.0, "accelerations": 150, "decelerations": 150, "stamina_score": 80.0,
        "player_rating": avg_rating, "performance_score": float(r["avg_performance"]),
        "offensive_contribution": avg_rating * 0.6, "defensive_contribution": avg_rating * 0.4,
        "possession_impact": avg_pass_acc / 100.0, "pressure_resistance": 0.75,
        "creativity_score": avg_xa * 10, "consistency_score": 0.75, "clutch_performance_score": 0.7,
        "goal_difference": int(total_goals - games_played) if games_played else 0,
        "goal_per_shot": float(r["avg_shots_on_target"]) / avg_shots if avg_shots > 0 else 0.15,
        "assist_per_key_pass": 0.05, "high_value_player": 1 if avg_market_value > 5e7 else 0,
        "player_games_played": games_played,
        "player_total_goals": int(total_goals / 11) if games_played else 0,
        "player_total_assists": int(total_assists / 11) if games_played else 0,
        "player_avg_rating": avg_rating, "player_avg_minutes": float(r["avg_minutes"]),
        "player_goals_per_game": total_goals / games_played if games_played else 0.0,
        "team_total_wins": total_wins, "team_total_goals": total_goals,
        "team_games_played": games_played, "team_win_rate": win_rate,
        "player_recent_goals": int(total_goals / games_played * 5) if games_played else 0,
        "player_recent_avg_rating": avg_rating, "player_goals_vs_opponent": 2,
        "position_avg_goals": 0.2, "position_avg_assists": 0.15, "position_avg_rating": 7.3,
        "goals_vs_position_avg": 0.0, "rating_vs_position_avg": 0.2,
    }


teams_in_test = sorted(set(test["team_a"]) | set(test["team_b"]))
team_records = [app_features_for_team(t) for t in teams_in_test]
resp = None
for attempt in range(20):
    try:
        resp = w.serving_endpoints.query(name=ENDPOINT_OLD, dataframe_records=team_records)
        break
    except Exception as e:
        print(f"   endpoint no listo (intento {attempt + 1}/20): {str(e)[:120]}")
        time.sleep(30)
if resp is None:
    raise RuntimeError("El endpoint no respondió tras 10 minutos")
old_score = dict(zip(teams_in_test, [float(p) for p in resp.predictions]))

pa = test["team_a"].map(old_score).to_numpy()
pb = test["team_b"].map(old_score).to_numpy()
p_old = np.clip(pa / (pa + pb + 1e-12), 1e-6, 1 - 1e-6)
y_arr = test["target"].to_numpy()
nd = y_arr != 1
y_bin = (y_arr[nd] == 0).astype(int)
metrics_old = {
    "accuracy_binary_nondraw": float(accuracy_score(y_bin, (p_old[nd] > 0.5).astype(int))),
    "log_loss_binary_nondraw": float(log_loss(y_bin, p_old[nd], labels=[0, 1])),
    "brier_binary_nondraw": float(brier_score_loss(y_bin, p_old[nd])),
    "ece_binary_nondraw": float(ece_binary(y_bin, p_old[nd])),
    "accuracy_3class": float(accuracy_score(y_arr, np.where(p_old > 0.5, 0, 2))),
}

# ---------------------------------------------------------------- curva + log
proba_new = calibrated.predict_proba(X_te)
p_new = proba_new[:, 0] / (proba_new[:, 0] + proba_new[:, 2] + 1e-12)
fig, ax = plt.subplots(figsize=(6, 6))
for label, probs in [("modelo viejo (v1)", p_old[nd]), ("modelo nuevo calibrado", p_new[nd])]:
    edges = np.linspace(0, 1, 11)
    mids, obs = [], []
    bidx = np.clip(np.digitize(probs, edges) - 1, 0, 9)
    for b in range(10):
        mask = bidx == b
        if mask.sum() >= 5:
            mids.append(probs[mask].mean())
            obs.append(y_bin[mask].mean())
    ax.plot(mids, obs, marker="o", label=label)
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="calibración perfecta")
ax.set_xlabel("probabilidad predicha (gana A)")
ax.set_ylabel("frecuencia observada")
ax.legend()
ax.set_title("Curva de calibración — test temporal (sin empates)")

with mlflow.start_run(run_id=parent_run_id):
    for prefix, m in [("new_cal_", metrics_cal), ("new_uncal_", metrics_uncal), ("old_", metrics_old)]:
        mlflow.log_metrics({prefix + k: v for k, v in m.items()})
    try:
        mlflow.log_figure(fig, "calibration_curve.png")
    except Exception as e:
        print(f"   (no se pudo subir la figura: {str(e)[:100]})")
fig.savefig("training/calibration_curve.png", dpi=120, bbox_inches="tight")

# ---------------------------------------------------------------- registro
print("7/8 Registrando modelo calibrado como nueva versión en UC...")
from mlflow.models.signature import infer_signature

conda_env = {
    "name": "mlflow-env",
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.12", "pip",
        {"pip": [
            f"mlflow=={mlflow.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            f"lightgbm=={lightgbm.__version__}",
            f"xgboost=={xgboost.__version__}",
            f"numpy=={np.__version__}",
            f"pandas=={pd.__version__}",
        ]},
    ],
}
signature = infer_signature(X_te, calibrated.predict_proba(X_te))
with mlflow.start_run(run_name=f"register_match_model_{best_name}"):
    mlflow.log_params({str(k): str(v) for k, v in results[best_name]["params"].items()})
    mlflow.log_param("algorithm", best_name)
    mlflow.log_param("calibration", "sigmoid_platt")
    mlflow.log_param("feature_columns", json.dumps(FEATURES))
    mlflow.log_metrics(metrics_cal)
    model_info = mlflow.sklearn.log_model(
        calibrated, name="model", signature=signature,
        input_example=X_te.head(3), registered_model_name=MODEL_UC_NAME,
        pyfunc_predict_fn="predict_proba", conda_env=conda_env,
        skops_trusted_types=[
            "collections.OrderedDict",
            "lightgbm.basic.Booster",
            "lightgbm.sklearn.LGBMClassifier",
            "sklearn.calibration._CalibratedClassifier",
            "sklearn.calibration._SigmoidCalibration",
        ])
new_version = model_info.registered_model_version
print(f"   ✅ {MODEL_UC_NAME} versión {new_version} registrada (producción sigue en v1)")

# ---------------------------------------------------------------- tablas estado
print("8/8 Escribiendo tablas de estado para servir...")


def esc(s):
    return str(s).replace("'", "''")


ts_values = []
for t in sorted(team_hist):
    s = team_state(t, pd.Timestamp("2026-08-01"))
    tm_rows = team_match[team_match["team"] == t]
    ts_values.append(
        f"('{esc(t)}', {s['elo']:.2f}, {s['matches_played']}, {s['winrate']:.4f}, "
        f"{s['form5_ppg']:.4f}, {s['form5_gf']:.4f}, {s['form5_ga']:.4f}, {s['form5_xg']:.4f}, "
        f"{s['form5_rating']:.4f}, {s['form5_shots']:.4f}, {tm_rows['squad_value'].mean():.2f}, "
        f"{tm_rows['squad_age'].mean():.3f}, {tm_rows['squad_height'].mean():.3f}, "
        f"DATE'{last_played[t].date().isoformat()}', {int(sum(h['won'] for h in team_hist[t]))})")
run_sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA_GOLD}.team_state_current AS
    SELECT * FROM (VALUES {", ".join(ts_values)}) AS v(
        team, elo, matches_played, winrate, form5_ppg, form5_gf, form5_ga, form5_xg,
        form5_rating, form5_shots, squad_value, squad_age, squad_height, last_match_date, wins)
""")

h2h_values = []
for pair, h in h2h.items():
    t1, t2 = sorted(pair)
    h2h_values.append(
        f"('{esc(t1)}', '{esc(t2)}', {h['n']}, {h.get(t1, 0)}, {h.get(t2, 0)}, "
        f"{h['draws']}, {h.get(('gd', t1), 0.0):.1f})")
run_sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA_GOLD}.h2h_state AS
    SELECT * FROM (VALUES {", ".join(h2h_values)}) AS v(
        team_1, team_2, matches, wins_1, wins_2, draws, gd_1)
""")

# ---------------------------------------------------------------- resumen
summary = {
    "best_algorithm": best_name,
    "best_params": {str(k): (v.item() if hasattr(v, "item") else v) for k, v in results[best_name]["params"].items()},
    "cv_log_loss_by_algorithm": {k: round(v["cv_log_loss"], 4) for k, v in results.items()},
    "registered_model_version": str(new_version),
    "n_train": len(train), "n_cal": len(cal), "n_test": len(test),
    "n_test_nondraw": int(nd.sum()),
    "metrics_new_calibrated": metrics_cal,
    "metrics_new_uncalibrated": metrics_uncal,
    "metrics_old_as_deployed": metrics_old,
    "mlflow_parent_run": parent_run_id,
}
out_path = sys.argv[1] if len(sys.argv) > 1 else "training/phase1_results.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
