# Databricks notebook source
# DBTITLE 1,Setup
# MAGIC %pip install xgboost lightgbm --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports y configuración
import json
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

CATALOG = "workspace"
SCHEMA_GOLD = "fifa_wc_gold"
GOLD_TABLE = f"{CATALOG}.{SCHEMA_GOLD}.player_performance_ml"
MODEL_UC_NAME = f"{CATALOG}.{SCHEMA_GOLD}.fifa_team_win_predictor"
EXPERIMENT = "/Users/jsierram96@gmail.com/fifa-wc-2026-match-model"
SEED = 42

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT)

# COMMAND ----------

# DBTITLE 1,Agregación a nivel equipo-partido
gold = spark.table(GOLD_TABLE).toPandas()
gold["match_date"] = pd.to_datetime(gold["match_date"])

numeric_cols = [
    "age", "height_cm", "market_value_eur", "expected_goals_xg", "shots",
    "player_rating", "goals_team", "goals_opponent",
]
for c in numeric_cols:
    gold[c] = pd.to_numeric(gold[c], errors="coerce")

team_match = (
    gold.groupby(["match_id", "match_date", "team", "opponent_team"], as_index=False)
    .agg(
        result=("match_result", "first"),
        goals_for=("goals_team", "max"),
        goals_against=("goals_opponent", "max"),
        xg=("expected_goals_xg", "sum"),
        shots=("shots", "sum"),
        rating=("player_rating", "mean"),
        squad_age=("age", "mean"),
        squad_height=("height_cm", "mean"),
        squad_value=("market_value_eur", "mean"),
        stage=("tournament_stage", "first"),
    )
    .sort_values("match_date")
    .reset_index(drop=True)
)
assert team_match.groupby("match_id").size().eq(2).all(), "cada partido debe tener 2 filas"
print(f"team-match rows: {len(team_match)}, matches: {team_match['match_id'].nunique()}")
print("stages:", sorted(team_match["stage"].unique()))

# COMMAND ----------

# DBTITLE 1,Construcción de features pre-partido (Elo, forma, descanso, H2H)
POINTS = {"W": 3, "D": 1, "L": 0}
ELO_K = 32
FORM_N = 5

matches = (
    team_match.sort_values(["match_date", "match_id"])
    .groupby("match_id", sort=False)
    .agg(list)
    .reset_index()
)

elo = {}                    # team -> rating actual
team_hist = {}              # team -> lista de dicts de partidos previos
h2h = {}                    # frozenset(pair) -> {team: wins, "draws": n, ("gd", team): sum}
last_played = {}            # team -> fecha del último partido

rng = np.random.default_rng(SEED)
rows = []

def team_state(team, date):
    hist = team_hist.get(team, [])
    recent = hist[-FORM_N:]
    state = {
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
    return state

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
    date = m["match_date"][0]
    stage = m["stage"][0]
    stage_knockout = 0 if str(stage).lower().startswith("group") else 1
    # asignación aleatoria de lado para no sesgar la clase
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
    h2h_stats = {
        "n": h["n"],
        "winrate_a": (h.get(ta, 0) / h["n"]) if h["n"] else 0.5,
        "gd_a": (h.get(("gd", ta), 0.0) / h["n"]) if h["n"] else 0.0,
    }

    row = build_feature_row(sa, sb, h2h_stats, stage_knockout)
    row.update({
        "match_id": m["match_id"], "match_date": date,
        "team_a": ta, "team_b": tb,
        "target": 0 if res_a == "W" else (1 if res_a == "D" else 2),
    })
    rows.append(row)

    # --- actualizar estados DESPUÉS de registrar las features ---
    ra, rb = elo.get(ta, 1500.0), elo.get(tb, 1500.0)
    exp_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
    score_a = 1.0 if res_a == "W" else (0.5 if res_a == "D" else 0.0)
    elo[ta] = ra + ELO_K * (score_a - exp_a)
    elo[tb] = rb + ELO_K * ((1.0 - score_a) - (1.0 - exp_a))

    for t, i in ((ta, idx[0]), (tb, idx[1])):
        team_hist.setdefault(t, []).append({
            "won": 1.0 if m["result"][i] == "W" else 0.0,
            "pts": POINTS[m["result"][i]],
            "gf": m["goals_for"][i], "ga": m["goals_against"][i],
            "xg": m["xg"][i], "rating": m["rating"][i], "shots": m["shots"][i],
        })
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
FEATURES = list(build_feature_row(
    team_state("_", pd.Timestamp("2026-01-01")) | {"squad_value": 0, "squad_age": 0, "squad_height": 0},
    team_state("_", pd.Timestamp("2026-01-01")) | {"squad_value": 0, "squad_age": 0, "squad_height": 0},
    {"n": 0, "winrate_a": 0.5, "gd_a": 0.0}, 0).keys())
print(f"dataset: {len(df)} partidos, {len(FEATURES)} features")
print(df["target"].value_counts(normalize=True).rename({0: "A gana", 1: "empate", 2: "B gana"}))

# COMMAND ----------

# DBTITLE 1,Split temporal train / calibración / test
n = len(df)
i_train, i_cal = int(n * 0.70), int(n * 0.80)
train, cal, test = df.iloc[:i_train], df.iloc[i_train:i_cal], df.iloc[i_cal:]
X_tr, y_tr = train[FEATURES], train["target"]
X_cal, y_cal = cal[FEATURES], cal["target"]
X_te, y_te = test[FEATURES], test["target"]
print(f"train: {len(train)} (hasta {train['match_date'].max().date()})")
print(f"cal:   {len(cal)} (hasta {cal['match_date'].max().date()})")
print(f"test:  {len(test)} ({test['match_date'].min().date()} → {test['match_date'].max().date()})")

# COMMAND ----------

# DBTITLE 1,Comparación de algoritmos con tuning temporal (MLflow)
tscv = TimeSeriesSplit(n_splits=3)

candidates = {
    "logistic_regression": (
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, random_state=SEED))]),
        {"clf__C": np.logspace(-3, 2, 12)},
        12,
    ),
    "xgboost": (
        XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                      random_state=SEED, n_jobs=-1, verbosity=0),
        {"n_estimators": [100, 200, 300, 500], "max_depth": [2, 3, 4, 5],
         "learning_rate": [0.01, 0.03, 0.05, 0.1], "subsample": [0.6, 0.8, 1.0],
         "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 5, 10],
         "reg_lambda": [0.1, 1.0, 10.0]},
        20,
    ),
    "lightgbm": (
        LGBMClassifier(objective="multiclass", random_state=SEED, n_jobs=-1, verbose=-1),
        {"n_estimators": [100, 200, 300, 500], "max_depth": [2, 3, 4, 5],
         "learning_rate": [0.01, 0.03, 0.05, 0.1], "num_leaves": [7, 15, 31],
         "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
         "min_child_samples": [10, 30, 50], "reg_lambda": [0.1, 1.0, 10.0]},
        20,
    ),
}

results = {}
with mlflow.start_run(run_name="match_model_selection") as parent:
    for name, (estimator, grid, n_iter) in candidates.items():
        with mlflow.start_run(run_name=name, nested=True):
            search = RandomizedSearchCV(
                estimator, grid, n_iter=n_iter, cv=tscv, scoring="neg_log_loss",
                random_state=SEED, n_jobs=-1, refit=True,
            )
            search.fit(X_tr, y_tr)
            cv_ll = -search.best_score_
            mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
            mlflow.log_metric("cv_log_loss", cv_ll)
            results[name] = {"model": search.best_estimator_, "cv_log_loss": cv_ll,
                             "params": search.best_params_}
            print(f"{name}: CV log-loss = {cv_ll:.4f}")

    best_name = min(results, key=lambda k: results[k]["cv_log_loss"])
    best_model = results[best_name]["model"]
    mlflow.log_param("selected_algorithm", best_name)
    mlflow.log_metric("selected_cv_log_loss", results[best_name]["cv_log_loss"])
    parent_run_id = parent.info.run_id

print(f"\n🏆 Mejor algoritmo: {best_name}")

# COMMAND ----------

# DBTITLE 1,Calibración (Platt) y métricas en test
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
    nd = y != 1  # partidos sin empate, para comparar contra el modelo viejo
    return {
        "accuracy_3class": float(accuracy_score(y, pred)),
        "log_loss_3class": float(log_loss(y, proba, labels=[0, 1, 2])),
        "brier_3class": multiclass_brier(y, proba),
        "accuracy_binary_nondraw": float(accuracy_score((y[nd] == 0).astype(int), (p_win_a[nd] > 0.5).astype(int))),
        "log_loss_binary_nondraw": float(log_loss((y[nd] == 0).astype(int), np.clip(p_win_a[nd], 1e-6, 1 - 1e-6), labels=[0, 1])),
        "brier_binary_nondraw": float(brier_score_loss((y[nd] == 0).astype(int), p_win_a[nd])),
        "ece_binary_nondraw": float(ece_binary((y[nd] == 0).astype(int), p_win_a[nd])),
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
print("Sin calibrar:", json.dumps(metrics_uncal, indent=2))
print("Calibrado:   ", json.dumps(metrics_cal, indent=2))

# COMMAND ----------

# DBTITLE 1,Evaluación del modelo viejo TAL COMO ESTÁ DESPLEGADO
# Replica exactamente get_team_features() de app.py (agregación sobre TODO el
# histórico + ~30 valores hardcodeados) y puntúa los mismos partidos de test
# consultando el endpoint real de producción (v1).

def app_features_for_team(team):
    g = gold[gold["team"].str.lower() == team.lower()]
    num = lambda s: pd.to_numeric(g[s], errors="coerce")
    games_played = len(g)
    total_goals = int(num("goals").sum())
    total_assists = int(num("assists").sum())
    total_wins = int((g["match_result"] == "W").sum())
    avg = lambda s, d: float(num(s).mean()) if not np.isnan(num(s).mean()) else d
    avg_age, avg_height, avg_weight = avg("age", 28.0), int(avg("height_cm", 180)), int(avg("weight_kg", 75))
    avg_market_value, avg_minutes = avg("market_value_eur", 5e7), avg("minutes_played", 85.0)
    avg_shots, avg_shots_target = avg("shots", 15.0), avg("shots_on_target", 9.0)
    avg_xg, avg_xa = avg("expected_goals_xg", 1.5), avg("expected_assists_xa", 0.5)
    avg_passes, avg_pass_acc = avg("successful_passes", 250.0), avg("pass_accuracy", 75.0)
    avg_dribbles, avg_succ_dribbles = avg("dribbles_attempted", 12.0), avg("successful_dribbles", 8.0)
    avg_tackles, avg_interceptions = avg("tackles", 18.0), avg("interceptions", 10.0)
    avg_rating, avg_performance = avg("player_rating", 7.5), avg("performance_score", 75.0)
    avg_goals_game = avg("goals_team", 2.0)
    win_rate = total_wins / games_played if games_played else 0.5
    return {
        "position": 3, "age": avg_age, "jersey_number": 10, "height_cm": avg_height,
        "weight_kg": avg_weight, "preferred_foot": 1, "market_value_eur": avg_market_value / 11,
        "goals_team": int(avg_goals_game), "goals_opponent": 1, "minutes_played": int(avg_minutes),
        "assists": int(total_assists / games_played) if games_played else 0,
        "shots": int(avg_shots), "shots_on_target": int(avg_shots_target),
        "expected_goals_xg": avg_xg, "expected_assists_xa": avg_xa,
        "key_passes": int(avg_passes * 0.1), "successful_passes": int(avg_passes),
        "total_passes": int(avg_passes / (avg_pass_acc / 100)) if avg_pass_acc > 0 else int(avg_passes),
        "pass_accuracy": avg_pass_acc, "dribbles_attempted": int(avg_dribbles),
        "successful_dribbles": int(avg_succ_dribbles), "crosses": 15, "successful_crosses": 5,
        "tackles": int(avg_tackles), "interceptions": int(avg_interceptions), "clearances": 15,
        "blocks": 8, "aerial_duels_won": 20, "aerial_duels_lost": 15, "recoveries": 25,
        "defensive_actions": int(avg_tackles + avg_interceptions), "fouls_committed": 12,
        "fouls_suffered": 10, "yellow_cards": 2, "red_cards": 0, "offsides": 3, "saves": 0,
        "save_percentage": 0.0, "punches": 0, "clean_sheet": 0, "goals_conceded": 1,
        "penalty_saves": 0, "distance_covered_km": 110.0, "sprint_distance_km": 15.0,
        "top_speed_kmh": 32.0, "accelerations": 150, "decelerations": 150, "stamina_score": 80.0,
        "player_rating": avg_rating, "performance_score": avg_performance,
        "offensive_contribution": avg_rating * 0.6, "defensive_contribution": avg_rating * 0.4,
        "possession_impact": avg_pass_acc / 100.0, "pressure_resistance": 0.75,
        "creativity_score": avg_xa * 10, "consistency_score": 0.75, "clutch_performance_score": 0.7,
        "goal_difference": int(total_goals - games_played) if games_played else 0,
        "goal_per_shot": avg_shots_target / avg_shots if avg_shots > 0 else 0.15,
        "assist_per_key_pass": 0.05, "high_value_player": 1 if avg_market_value > 5e7 else 0,
        "player_games_played": games_played,
        "player_total_goals": int(total_goals / 11) if games_played else 0,
        "player_total_assists": int(total_assists / 11) if games_played else 0,
        "player_avg_rating": avg_rating, "player_avg_minutes": avg_minutes,
        "player_goals_per_game": total_goals / games_played if games_played else 0.0,
        "team_total_wins": total_wins, "team_total_goals": total_goals,
        "team_games_played": games_played, "team_win_rate": win_rate,
        "player_recent_goals": int(total_goals / games_played * 5) if games_played else 0,
        "player_recent_avg_rating": avg_rating, "player_goals_vs_opponent": 2,
        "position_avg_goals": 0.2, "position_avg_assists": 0.15, "position_avg_rating": 7.3,
        "goals_vs_position_avg": 0.0, "rating_vs_position_avg": 0.2,
    }

import time
from databricks.sdk import WorkspaceClient

teams_in_test = sorted(set(test["team_a"]) | set(test["team_b"]))
team_records = [app_features_for_team(t) for t in teams_in_test]

w = WorkspaceClient()
resp = None
for attempt in range(20):  # el endpoint escala desde cero: puede tardar varios minutos
    try:
        resp = w.serving_endpoints.query(
            name="fifa-team-win-predictor", dataframe_records=team_records
        )
        break
    except Exception as e:
        print(f"endpoint no listo (intento {attempt + 1}/20): {str(e)[:150]}")
        time.sleep(30)
if resp is None:
    raise RuntimeError("El endpoint fifa-team-win-predictor no respondió tras 10 minutos")
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
    "accuracy_3class": float(accuracy_score(y_arr, np.where(p_old > 0.5, 0, 2))),  # nunca predice empate
    "note": "modelo v1 evaluado como lo usa la app (features agregadas + hardcoded, normalización p_a/(p_a+p_b))",
}
print(json.dumps(metrics_old, indent=2))

# COMMAND ----------

# DBTITLE 1,Curva de calibración y registro de resultados
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
    for prefix, m in [("new_cal_", metrics_cal), ("new_uncal_", metrics_uncal)]:
        mlflow.log_metrics({prefix + k: v for k, v in m.items()})
    mlflow.log_metrics({"old_" + k: v for k, v in metrics_old.items() if isinstance(v, float)})
    mlflow.log_figure(fig, "calibration_curve.png")

# COMMAND ----------

# DBTITLE 1,Registrar modelo calibrado como nueva versión (sin tocar producción)
input_example = X_te.head(3)
signature = infer_signature(X_te, calibrated.predict_proba(X_te))

with mlflow.start_run(run_name=f"register_match_model_{best_name}"):
    mlflow.log_params(results[best_name]["params"])
    mlflow.log_param("algorithm", best_name)
    mlflow.log_param("calibration", "sigmoid_platt")
    mlflow.log_param("feature_columns", json.dumps(FEATURES))
    mlflow.log_metrics({k: v for k, v in metrics_cal.items()})
    model_info = mlflow.sklearn.log_model(
        calibrated,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=MODEL_UC_NAME,
        pyfunc_predict_fn="predict_proba",
    )
new_version = model_info.registered_model_version
print(f"✅ Registrado {MODEL_UC_NAME} versión {new_version} (producción sigue en v1)")

# COMMAND ----------

# DBTITLE 1,Tablas de estado para servir (la app las consulta en vivo)
team_state_rows = []
for t in sorted(team_hist):
    s = team_state(t, pd.Timestamp("2026-08-01"))
    tm_rows = team_match[team_match["team"] == t]
    team_state_rows.append({
        "team": t, "elo": s["elo"], "matches_played": s["matches_played"],
        "winrate": s["winrate"], "form5_ppg": s["form5_ppg"], "form5_gf": s["form5_gf"],
        "form5_ga": s["form5_ga"], "form5_xg": s["form5_xg"], "form5_rating": s["form5_rating"],
        "form5_shots": s["form5_shots"],
        "squad_value": float(tm_rows["squad_value"].mean()),
        "squad_age": float(tm_rows["squad_age"].mean()),
        "squad_height": float(tm_rows["squad_height"].mean()),
        "last_match_date": last_played[t].date().isoformat(),
        "wins": int(sum(h["won"] for h in team_hist[t])),
    })
spark.createDataFrame(pd.DataFrame(team_state_rows)).write.mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{SCHEMA_GOLD}.team_state_current")

h2h_rows = []
for pair, h in h2h.items():
    t1, t2 = sorted(pair)
    h2h_rows.append({
        "team_1": t1, "team_2": t2, "matches": h["n"],
        "wins_1": h.get(t1, 0), "wins_2": h.get(t2, 0), "draws": h["draws"],
        "gd_1": float(h.get(("gd", t1), 0.0)),
    })
spark.createDataFrame(pd.DataFrame(h2h_rows)).write.mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{SCHEMA_GOLD}.h2h_state")
print("✅ Tablas team_state_current y h2h_state escritas")

# COMMAND ----------

# DBTITLE 1,Resumen final
summary = {
    "best_algorithm": best_name,
    "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in results[best_name]["params"].items()},
    "cv_log_loss_by_algorithm": {k: round(v["cv_log_loss"], 4) for k, v in results.items()},
    "registered_model_version": str(new_version),
    "n_train": len(train), "n_cal": len(cal), "n_test": len(test),
    "metrics_new_calibrated": metrics_cal,
    "metrics_new_uncalibrated": metrics_uncal,
    "metrics_old_as_deployed": {k: v for k, v in metrics_old.items() if isinstance(v, float)},
}
print(json.dumps(summary, indent=2))
dbutils.notebook.exit(json.dumps(summary))
