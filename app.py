"""FIFA World Cup 2026 — Predictor de enfrentamientos.

Compara dos selecciones con el modelo a nivel partido (LightGBM, v5 en
Unity Catalog). Las features (Elo, forma reciente, head-to-head, plantilla)
se leen de las tablas de estado y se construyen con fifa_features.py,
exactamente igual que en entrenamiento.

Config por variables de entorno:
  MODEL_ENDPOINT   nombre del serving endpoint (default fifa-team-win-predictor)
  MODEL_LOCAL_URI  URI mlflow para inferencia local en desarrollo
                   (ej. models:/workspace.fifa_wc_gold.fifa_team_win_predictor/5)
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

from app_helpers import (build_input_frame, combine_mirrored_proba,
                         friendly_error, verdict_for)
from fifa_features import (build_feature_row, h2h_stats_for,
                           state_from_table_row)

WAREHOUSE_ID = os.getenv("WAREHOUSE_ID", "f9bb0b517b9fc8ba")
ENDPOINT_NAME = os.getenv("MODEL_ENDPOINT", "fifa-team-win-predictor")
MODEL_LOCAL_URI = os.getenv("MODEL_LOCAL_URI", "")
STATE_TABLE = "workspace.fifa_wc_gold.team_state_current"
H2H_TABLE = "workspace.fifa_wc_gold.h2h_state"

# Paleta (slots validados de referencia: azul/naranja par cálido-frío, gris neutro)
C_TEAM_A, C_TEAM_B = "#2a78d6", "#eb6834"
C_DRAW, C_INK, C_INK2, C_MUTED = "#898781", "#0b0b0b", "#52514e", "#898781"
C_SURFACE, C_HAIRLINE = "#ffffff", "rgba(11,11,11,0.10)"

st.set_page_config(page_title="Predictor Mundial 2026", page_icon="⚽", layout="centered")

st.markdown(f"""
<style>
  .block-container {{ max-width: 880px; }}
  h1 {{ font-size: 1.9rem !important; letter-spacing: -0.02em; }}
  .subtitle {{ color: {C_INK2}; font-size: 0.95rem; margin-top: -0.6rem; }}
  .hero {{ font-size: 1.35rem; font-weight: 650; margin: 0.2rem 0 0.8rem; }}
  .hero small {{ display: block; font-size: 0.85rem; font-weight: 400; color: {C_INK2}; margin-top: 2px; }}
  .prob-bar {{ display: flex; gap: 2px; height: 34px; margin: 4px 0 6px; }}
  .prob-seg {{ display: flex; align-items: center; justify-content: center;
               font-size: 0.85rem; font-weight: 600; color: #fff; min-width: 2px; }}
  .prob-seg.first {{ border-radius: 6px 0 0 6px; }}
  .prob-seg.last {{ border-radius: 0 6px 6px 0; }}
  .legend {{ display: flex; gap: 16px; flex-wrap: wrap; font-size: 0.85rem; color: {C_INK2};
             margin-bottom: 0.6rem; }}
  .legend .chip {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px;
                   margin-right: 5px; }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px; margin: 0.4rem 0 0.8rem; }}
  .tile {{ background: {C_SURFACE}; border: 1px solid {C_HAIRLINE}; border-radius: 10px;
           padding: 10px 14px; border-top: 3px solid var(--accent, {C_MUTED}); }}
  .tile .v {{ font-size: 1.45rem; font-weight: 650; color: {C_INK}; }}
  .tile .l {{ font-size: 0.78rem; color: {C_INK2}; }}
  .sec {{ font-size: 1.02rem; font-weight: 650; margin: 1.1rem 0 0.4rem; }}
  .frow {{ display: grid; grid-template-columns: minmax(96px, 130px) 1fr; gap: 4px 12px;
           align-items: center; margin-bottom: 10px; }}
  .fname {{ font-size: 0.85rem; color: {C_INK2}; }}
  .fbars {{ display: flex; flex-direction: column; gap: 3px; }}
  .fbar-line {{ display: flex; align-items: center; gap: 8px; }}
  .fbar {{ height: 12px; border-radius: 0 4px 4px 0; }}
  .fval {{ font-size: 0.8rem; color: {C_INK}; white-space: nowrap; }}
  .imp-row {{ display: grid; grid-template-columns: minmax(120px, 190px) 1fr 44px;
              gap: 10px; align-items: center; margin-bottom: 5px; }}
  .imp-name {{ font-size: 0.82rem; color: {C_INK2}; text-align: right; }}
  .imp-bar {{ height: 12px; border-radius: 0 4px 4px 0; background: {C_TEAM_A}; }}
  .imp-val {{ font-size: 0.8rem; color: {C_INK2}; }}
  .note {{ color: {C_MUTED}; font-size: 0.8rem; margin-top: 1.2rem;
           border-top: 1px solid {C_HAIRLINE}; padding-top: 0.6rem; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_client():
    return WorkspaceClient()


def run_sql(statement):
    w = get_client()
    st_ = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=statement, wait_timeout="50s")
    waited = 0
    while st_.status.state in (StatementState.PENDING, StatementState.RUNNING) and waited < 120:
        time.sleep(2)
        waited += 2
        st_ = w.statement_execution.get_statement(st_.statement_id)
    if st_.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL statement failed: {st_.status}")
    cols = [c.name for c in st_.manifest.schema.columns]
    return pd.DataFrame(st_.result.data_array or [], columns=cols)


@st.cache_data(ttl=600, show_spinner=False)
def load_team_state():
    return run_sql(f"SELECT * FROM {STATE_TABLE}")


@st.cache_data(ttl=600, show_spinner=False)
def load_h2h():
    return run_sql(f"SELECT * FROM {H2H_TABLE}")


@st.cache_resource
def load_local_model(uri):
    import mlflow  # solo en desarrollo local
    mlflow.set_registry_uri("databricks-uc")
    return mlflow.pyfunc.load_model(uri)


def predict_match(row_ab, row_ba):
    """Predice con simetría forzada: promedia la predicción A-vs-B con la
    B-vs-A espejada, para que el orden de selección no cambie el resultado."""
    X = build_input_frame(row_ab, row_ba)
    if MODEL_LOCAL_URI:
        proba = np.asarray(load_local_model(MODEL_LOCAL_URI).predict(X))
    else:
        resp = get_client().serving_endpoints.query(
            name=ENDPOINT_NAME, dataframe_records=X.to_dict(orient="records"))
        proba = np.asarray(resp.predictions, dtype=float)
    return combine_mirrored_proba(proba)


def prob_bar_html(team_a, team_b, p):
    pcts = [p[0] * 100, p[1] * 100, p[2] * 100]
    labels = [team_a, "Empate", team_b]
    colors = [C_TEAM_A, C_DRAW, C_TEAM_B]
    segs = []
    for i, (pct, color) in enumerate(zip(pcts, colors)):
        cls = "prob-seg" + (" first" if i == 0 else "") + (" last" if i == 2 else "")
        inner = f"{pct:.0f}%" if pct >= 12 else ""
        segs.append(f'<div class="{cls}" style="width:{pct:.2f}%;background:{color}" '
                    f'title="{labels[i]}: {pct:.1f}%">{inner}</div>')
    legend = "".join(
        f'<span><span class="chip" style="background:{c}"></span>{l} '
        f'<strong>{pct:.1f}%</strong></span>'
        for l, c, pct in zip(labels, colors, pcts))
    return f'<div class="prob-bar">{"".join(segs)}</div><div class="legend">{legend}</div>'


def tiles_html(team, state, accent):
    tiles = [
        (f"{state['elo']:.0f}", "Elo actual"),
        (f"{state['winrate'] * 100:.0f}%", "Victorias (torneo)"),
        (f"{state['form5_ppg']:.2f}", "Puntos/partido (últ. 5)"),
        (f"€{state['squad_value'] / 1e6:.0f}M", "Valor medio jugador"),
    ]
    cells = "".join(f'<div class="tile" style="--accent:{accent}">'
                    f'<div class="v">{v}</div><div class="l">{l}</div></div>'
                    for v, l in tiles)
    return f'<div class="sec">{team}</div><div class="tiles">{cells}</div>'


def form_compare_html(team_a, team_b, sa, sb):
    metrics = [
        ("Puntos/partido", "form5_ppg", "{:.2f}"),
        ("Goles a favor", "form5_gf", "{:.2f}"),
        ("Goles en contra", "form5_ga", "{:.2f}"),
        ("xG generado", "form5_xg", "{:.2f}"),
        ("Rating medio", "form5_rating", "{:.2f}"),
    ]
    legend = (f'<div class="legend"><span><span class="chip" style="background:{C_TEAM_A}">'
              f'</span>{team_a}</span><span><span class="chip" style="background:{C_TEAM_B}">'
              f'</span>{team_b}</span></div>')
    rows = []
    for label, key, fmt in metrics:
        va, vb = sa[key], sb[key]
        mx = max(va, vb, 1e-9)
        bars = ""
        for v, color in ((va, C_TEAM_A), (vb, C_TEAM_B)):
            width = max(v / mx * 100, 1.5)
            bars += (f'<div class="fbar-line"><div class="fbar" '
                     f'style="width:{width:.1f}%;background:{color}"></div>'
                     f'<span class="fval">{fmt.format(v)}</span></div>')
        rows.append(f'<div class="frow"><div class="fname">{label}</div>'
                    f'<div class="fbars">{bars}</div></div>')
    return legend + "".join(rows)


def h2h_html(team_a, team_b, h2h_row):
    if h2h_row is None or int(h2h_row["matches"]) == 0:
        return f'<p class="fname">— {team_a} y {team_b} no se han enfrentado en este torneo.</p>'
    n = int(h2h_row["matches"])
    if team_a == h2h_row["team_1"]:
        wa, wb = int(h2h_row["wins_1"]), int(h2h_row["wins_2"])
    else:
        wa, wb = int(h2h_row["wins_2"]), int(h2h_row["wins_1"])
    d = int(h2h_row["draws"])
    segs, colors, vals = [], [C_TEAM_A, C_DRAW, C_TEAM_B], [wa, d, wb]
    labels = [f"{team_a} ganó", "Empates", f"{team_b} ganó"]
    for i, (v, c) in enumerate(zip(vals, colors)):
        pct = v / n * 100
        cls = "prob-seg" + (" first" if i == 0 else "") + (" last" if i == 2 else "")
        segs.append(f'<div class="{cls}" style="width:{max(pct, 0.5):.1f}%;background:{c}" '
                    f'title="{labels[i]}: {v}">{v if pct >= 12 else ""}</div>')
    legend = "".join(
        f'<span><span class="chip" style="background:{c}"></span>{l} <strong>{v}</strong></span>'
        for l, c, v in zip(labels, colors, vals))
    return (f'<p class="fname">Se han enfrentado <strong>{n}</strong> veces:</p>'
            f'<div class="prob-bar" style="height:22px">{"".join(segs)}</div>'
            f'<div class="legend">{legend}</div>')


def importance_html():
    path = Path(__file__).parent / "training" / "feature_importance.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())["features"][:8]
    mx = max(f["importance"] for f in data) or 1
    friendly = {
        "elo_a": "Elo equipo A", "elo_b": "Elo equipo B", "elo_diff": "Diferencia de Elo",
        "form5_ppg_a": "Forma (pts) A", "form5_ppg_b": "Forma (pts) B",
        "form5_ppg_diff": "Dif. forma (pts)", "form5_gf_diff": "Dif. goles a favor",
        "form5_ga_diff": "Dif. goles en contra", "form5_xg_diff": "Dif. xG",
        "form5_rating_diff": "Dif. rating", "form5_shots_diff": "Dif. tiros",
        "winrate_a": "% victorias A", "winrate_b": "% victorias B",
        "winrate_diff": "Dif. % victorias", "rest_days_a": "Descanso A",
        "rest_days_b": "Descanso B", "rest_diff": "Dif. descanso",
        "matches_played_a": "Partidos jugados A", "matches_played_b": "Partidos jugados B",
        "h2h_matches": "Enfrentamientos previos", "h2h_winrate_a": "% victorias H2H",
        "h2h_gd_a": "Dif. goles H2H", "squad_value_diff": "Dif. valor plantilla",
        "squad_age_diff": "Dif. edad plantilla", "squad_height_diff": "Dif. altura plantilla",
        "stage_knockout": "Fase eliminatoria",
    }
    rows = "".join(
        f'<div class="imp-row"><div class="imp-name">{friendly.get(f["feature"], f["feature"])}</div>'
        f'<div><div class="imp-bar" style="width:{f["importance"] / mx * 100:.1f}%"></div></div>'
        f'<div class="imp-val">{f["importance"]:.0f}</div></div>'
        for f in data)
    return rows


# ================================ UI ================================

st.title("⚽ Predictor Mundial 2026")
st.markdown('<p class="subtitle">Probabilidades de victoria estimadas con un modelo '
            'a nivel partido: Elo dinámico, forma reciente, historial directo y '
            'calidad de plantilla.</p>', unsafe_allow_html=True)

try:
    with st.spinner("Cargando el estado de los equipos..."):
        teams_df = load_team_state()
        h2h_df = load_h2h()
except Exception as exc:
    st.error("😕 " + friendly_error(exc))
    st.stop()

for col in teams_df.columns:
    if col not in ("team", "last_match_date"):
        teams_df[col] = pd.to_numeric(teams_df[col])
teams = sorted(teams_df["team"].tolist())

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Equipo A", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
with col2:
    team_b = st.selectbox("Equipo B", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)
knockout = st.toggle("Partido de eliminación directa", value=True,
                     help="En fase de grupos el empate es un resultado más frecuente.")

if st.button("Comparar equipos", type="primary", use_container_width=True):
    if team_a == team_b:
        st.warning("Elige dos equipos distintos para comparar.")
        st.stop()

    row_a = teams_df[teams_df["team"] == team_a].iloc[0]
    row_b = teams_df[teams_df["team"] == team_b].iloc[0]
    sa, sb = state_from_table_row(row_a), state_from_table_row(row_b)
    t1, t2 = sorted([team_a, team_b])
    h2h_match = h2h_df[(h2h_df["team_1"] == t1) & (h2h_df["team_2"] == t2)]
    h2h_row = h2h_match.iloc[0] if len(h2h_match) else None

    stage = 1 if knockout else 0
    row_ab = build_feature_row(sa, sb, h2h_stats_for(team_a, team_b, h2h_row), stage)
    row_ba = build_feature_row(sb, sa, h2h_stats_for(team_b, team_a, h2h_row), stage)

    try:
        with st.spinner(f"Calculando probabilidades de {team_a} vs {team_b}..."):
            p = predict_match(row_ab, row_ba)
    except Exception as exc:
        st.error("😕 " + friendly_error(exc))
        st.stop()

    verdict, detail = verdict_for(team_a, team_b, p)
    st.markdown(f'<div class="hero">{verdict}<small>{detail}</small></div>',
                unsafe_allow_html=True)

    st.markdown(prob_bar_html(team_a, team_b, p), unsafe_allow_html=True)
    with st.expander("Ver como tabla"):
        st.dataframe(pd.DataFrame({
            "Resultado": [f"Gana {team_a}", "Empate", f"Gana {team_b}"],
            "Probabilidad": [f"{v * 100:.1f}%" for v in p],
        }), hide_index=True, use_container_width=True)

    st.markdown(tiles_html(team_a, sa, C_TEAM_A), unsafe_allow_html=True)
    st.markdown(tiles_html(team_b, sb, C_TEAM_B), unsafe_allow_html=True)

    st.markdown('<div class="sec">Forma reciente (últimos 5 partidos)</div>',
                unsafe_allow_html=True)
    st.markdown(form_compare_html(team_a, team_b, sa, sb), unsafe_allow_html=True)

    st.markdown('<div class="sec">Historial directo</div>', unsafe_allow_html=True)
    st.markdown(h2h_html(team_a, team_b, h2h_row), unsafe_allow_html=True)

    imp = importance_html()
    if imp:
        with st.expander("¿Qué variables pesan más en el modelo?"):
            st.markdown(imp, unsafe_allow_html=True)
            st.caption("Importancia (número de divisiones) de cada variable en el "
                       "LightGBM entrenado con validación temporal.")

st.markdown('<p class="note">Modelo v5 · workspace.fifa_wc_gold.fifa_team_win_predictor · '
            'Proyecto educativo con datos sintéticos de Kaggle: las probabilidades son '
            'honestas respecto a esos datos, no pronósticos reales del Mundial 2026.</p>',
            unsafe_allow_html=True)
