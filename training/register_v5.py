"""Registra como v5 la variante SIN calibrar del modelo de la Fase 1.

La calibración Platt empeoró las métricas (105 partidos de calibración son
muy pocos para 3 clases); el LightGBM sin calibrar ya estaba bien calibrado
(ECE 0.027). Reproduce el dataset y el fit exactos de train_local.py,
verifica las métricas contra training/phase1_results.json y registra.
También exporta la importancia de features para la app.
"""
import json
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import sklearn
import lightgbm
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from lightgbm import LGBMClassifier
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fifa_features import FEATURE_COLUMNS, build_feature_row  # noqa: E402

PROFILE = "jsierram96"
WAREHOUSE_ID = "f9bb0b517b9fc8ba"
GOLD_TABLE = "workspace.fifa_wc_gold.player_performance_ml"
MODEL_UC_NAME = "workspace.fifa_wc_gold.fifa_team_win_predictor"
EXPERIMENT = "/Users/jsierram96@gmail.com/fifa-wc-2026-match-model"
SEED = 42
POINTS = {"W": 3, "D": 1, "L": 0}
ELO_K = 32
FORM_N = 5

w = WorkspaceClient(profile=PROFILE)
mlflow.set_tracking_uri(f"databricks://{PROFILE}")
mlflow.set_registry_uri(f"databricks-uc://{PROFILE}")
mlflow.set_experiment(EXPERIMENT)

results_path = Path(__file__).parent / "phase1_results.json"
phase1 = json.loads(results_path.read_text())
BEST_PARAMS = phase1["best_params"]
EXPECTED = phase1["metrics_new_uncalibrated"]


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


print("1/4 Leyendo y reconstruyendo el dataset de partidos...")
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
    row.update({"match_date": date,
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
n = len(df)
i_train, i_cal = int(n * 0.70), int(n * 0.80)
train, test = df.iloc[:i_train], df.iloc[int(n * 0.80):]
X_tr, y_tr = train[FEATURE_COLUMNS], train["target"]
X_te, y_te = test[FEATURE_COLUMNS], test["target"]

print("2/4 Entrenando LightGBM con los mejores hiperparámetros...")
model = LGBMClassifier(objective="multiclass", random_state=SEED, n_jobs=-1,
                       verbose=-1, **BEST_PARAMS)
model.fit(X_tr, y_tr)

proba = model.predict_proba(X_te)
y_arr = np.asarray(y_te)
p_win_a = proba[:, 0] / (proba[:, 0] + proba[:, 2] + 1e-12)
nd = y_arr != 1
yb = (y_arr[nd] == 0).astype(int)
metrics = {
    "accuracy_3class": float(accuracy_score(y_arr, proba.argmax(axis=1))),
    "log_loss_3class": float(log_loss(y_arr, proba, labels=[0, 1, 2])),
    "accuracy_binary_nondraw": float(accuracy_score(yb, (p_win_a[nd] > 0.5).astype(int))),
    "log_loss_binary_nondraw": float(log_loss(yb, np.clip(p_win_a[nd], 1e-6, 1 - 1e-6), labels=[0, 1])),
    "brier_binary_nondraw": float(brier_score_loss(yb, p_win_a[nd])),
}
print(json.dumps(metrics, indent=2))
for k, v in metrics.items():
    if k in EXPECTED and abs(v - EXPECTED[k]) > 1e-6:
        print(f"⚠️ {k} difiere de la Fase 1: {v:.6f} vs {EXPECTED[k]:.6f}")

print("3/4 Registrando v5 (sin calibrar) en Unity Catalog...")
conda_env = {
    "name": "mlflow-env",
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.12", "pip",
        {"pip": [
            f"mlflow=={mlflow.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            f"lightgbm=={lightgbm.__version__}",
            f"numpy=={np.__version__}",
            f"pandas=={pd.__version__}",
        ]},
    ],
}
signature = infer_signature(X_te, model.predict_proba(X_te))
with mlflow.start_run(run_name="register_match_model_v5_uncalibrated"):
    mlflow.log_params({str(k): str(v) for k, v in BEST_PARAMS.items()})
    mlflow.log_param("algorithm", "lightgbm")
    mlflow.log_param("calibration", "none")
    mlflow.log_param("feature_columns", json.dumps(FEATURE_COLUMNS))
    mlflow.log_metrics(metrics)
    model_info = mlflow.sklearn.log_model(
        model, name="model", signature=signature,
        input_example=X_te.head(3), registered_model_name=MODEL_UC_NAME,
        pyfunc_predict_fn="predict_proba", conda_env=conda_env,
        skops_trusted_types=[
            "collections.OrderedDict",
            "lightgbm.basic.Booster",
            "lightgbm.sklearn.LGBMClassifier",
        ])
print(f"   ✅ versión {model_info.registered_model_version} registrada")

print("4/4 Exportando importancia de features para la app...")
importance = sorted(
    zip(FEATURE_COLUMNS, model.feature_importances_.tolist()),
    key=lambda x: -x[1])
(Path(__file__).parent / "feature_importance.json").write_text(
    json.dumps({"algorithm": "lightgbm", "importance_type": "split",
                "features": [{"feature": f, "importance": v} for f, v in importance]},
               indent=2))
print(json.dumps({"registered_version": str(model_info.registered_model_version),
                  "metrics": metrics}, indent=2))
