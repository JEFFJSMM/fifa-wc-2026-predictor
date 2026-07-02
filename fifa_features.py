"""Construcción de features del modelo de predicción a nivel partido.

Módulo compartido entre el entrenamiento (training/) y la app (app.py):
la fila de features que recibe el modelo debe construirse EXACTAMENTE igual
en ambos lados. El orden de FEATURE_COLUMNS es el del signature del modelo.
"""

FEATURE_COLUMNS = [
    "elo_a", "elo_b", "elo_diff",
    "form5_ppg_a", "form5_ppg_b", "form5_ppg_diff",
    "form5_gf_diff", "form5_ga_diff", "form5_xg_diff",
    "form5_rating_diff", "form5_shots_diff",
    "winrate_a", "winrate_b", "winrate_diff",
    "rest_days_a", "rest_days_b", "rest_diff",
    "matches_played_a", "matches_played_b",
    "h2h_matches", "h2h_winrate_a", "h2h_gd_a",
    "squad_value_diff", "squad_age_diff", "squad_height_diff",
    "stage_knockout",
]

# Clases del modelo (índices de predict_proba)
CLASS_TEAM_A_WINS, CLASS_DRAW, CLASS_TEAM_B_WINS = 0, 1, 2


def build_feature_row(sa, sb, h2h_stats, stage_knockout):
    """Construye la fila de features de un partido A vs B.

    sa/sb: dicts de estado pre-partido de cada equipo con claves
        elo, matches_played, winrate, rest_days, form5_ppg, form5_gf,
        form5_ga, form5_xg, form5_rating, form5_shots,
        squad_value, squad_age, squad_height
    h2h_stats: dict con n, winrate_a, gd_a (desde la perspectiva del equipo A)
    stage_knockout: 1 si es eliminación directa, 0 si es fase de grupos
    """
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


def state_from_table_row(row, rest_days=4.0):
    """Convierte una fila de workspace.fifa_wc_gold.team_state_current en el
    dict de estado que espera build_feature_row.

    Para un enfrentamiento hipotético el descanso real es desconocido; se usa
    un valor neutro igual para ambos equipos (rest_diff = 0).
    """
    return {
        "elo": float(row["elo"]),
        "matches_played": int(row["matches_played"]),
        "winrate": float(row["winrate"]),
        "rest_days": float(rest_days),
        "form5_ppg": float(row["form5_ppg"]),
        "form5_gf": float(row["form5_gf"]),
        "form5_ga": float(row["form5_ga"]),
        "form5_xg": float(row["form5_xg"]),
        "form5_rating": float(row["form5_rating"]),
        "form5_shots": float(row["form5_shots"]),
        "squad_value": float(row["squad_value"]),
        "squad_age": float(row["squad_age"]),
        "squad_height": float(row["squad_height"]),
    }


def h2h_stats_for(team_a, team_b, h2h_row):
    """Orienta una fila de workspace.fifa_wc_gold.h2h_state (team_1 < team_2
    alfabéticamente) hacia la perspectiva de team_a. h2h_row puede ser None
    si los equipos nunca se han enfrentado."""
    if h2h_row is None or int(h2h_row["matches"]) == 0:
        return {"n": 0, "winrate_a": 0.5, "gd_a": 0.0}
    n = int(h2h_row["matches"])
    if team_a == h2h_row["team_1"]:
        wins_a, gd_a = int(h2h_row["wins_1"]), float(h2h_row["gd_1"])
    else:
        wins_a, gd_a = int(h2h_row["wins_2"]), -float(h2h_row["gd_1"])
    return {"n": n, "winrate_a": wins_a / n, "gd_a": gd_a / n}
