"""Pruebas del módulo compartido de features (crítico: entrenamiento y app
deben construir exactamente la misma fila)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fifa_features import (FEATURE_COLUMNS, build_feature_row, h2h_stats_for,
                           state_from_table_row)


def make_state(elo=1500.0, ppg=1.5, winrate=0.5, value=5e7, **overrides):
    s = {
        "elo": elo, "matches_played": 10, "winrate": winrate, "rest_days": 4.0,
        "form5_ppg": ppg, "form5_gf": 1.4, "form5_ga": 1.2, "form5_xg": 1.3,
        "form5_rating": 7.1, "form5_shots": 12.5,
        "squad_value": value, "squad_age": 27.0, "squad_height": 181.0,
    }
    s.update(overrides)
    return s


H2H_NEUTRAL = {"n": 0, "winrate_a": 0.5, "gd_a": 0.0}


def test_feature_row_keys_match_feature_columns_in_order():
    row = build_feature_row(make_state(), make_state(), H2H_NEUTRAL, 1)
    assert list(row.keys()) == FEATURE_COLUMNS


def test_diffs_are_a_minus_b():
    row = build_feature_row(make_state(elo=1600), make_state(elo=1500), H2H_NEUTRAL, 0)
    assert row["elo_diff"] == pytest.approx(100)
    assert row["elo_a"] == 1600 and row["elo_b"] == 1500


def test_mirror_symmetry():
    """Invertir A y B debe negar todas las diffs y intercambiar los niveles."""
    sa = make_state(elo=1650, ppg=2.1, winrate=0.7, value=8e7)
    sb = make_state(elo=1480, ppg=1.1, winrate=0.4, value=3e7)
    ab = build_feature_row(sa, sb, H2H_NEUTRAL, 1)
    ba = build_feature_row(sb, sa, H2H_NEUTRAL, 1)
    for col in FEATURE_COLUMNS:
        if col.startswith("h2h_"):
            continue  # las H2H son de perspectiva A, sin contraparte _b
        if col.endswith("_diff"):
            assert ab[col] == pytest.approx(-ba[col]), col
        elif col.endswith("_a"):
            assert ab[col] == pytest.approx(ba[col.replace("_a", "_b")]), col


def test_state_from_table_row_parses_strings():
    """El SQL warehouse devuelve strings; deben convertirse a numéricos."""
    row = {"elo": "1523.5", "matches_played": "12", "winrate": "0.58",
           "form5_ppg": "1.8", "form5_gf": "1.5", "form5_ga": "0.9",
           "form5_xg": "1.4", "form5_rating": "7.3", "form5_shots": "13.1",
           "squad_value": "45000000.0", "squad_age": "26.8", "squad_height": "180.2"}
    s = state_from_table_row(row)
    assert s["elo"] == pytest.approx(1523.5)
    assert s["matches_played"] == 12
    assert s["rest_days"] == pytest.approx(4.0)  # neutro por defecto


def test_h2h_orientation_team_1_perspective():
    h2h_row = {"team_1": "Argentina", "team_2": "Brazil", "matches": 4,
               "wins_1": 3, "wins_2": 1, "draws": 0, "gd_1": 4.0}
    stats = h2h_stats_for("Argentina", "Brazil", h2h_row)
    assert stats == {"n": 4, "winrate_a": pytest.approx(0.75), "gd_a": pytest.approx(1.0)}


def test_h2h_orientation_flipped():
    h2h_row = {"team_1": "Argentina", "team_2": "Brazil", "matches": 4,
               "wins_1": 3, "wins_2": 1, "draws": 0, "gd_1": 4.0}
    stats = h2h_stats_for("Brazil", "Argentina", h2h_row)
    assert stats == {"n": 4, "winrate_a": pytest.approx(0.25), "gd_a": pytest.approx(-1.0)}


def test_h2h_missing_is_neutral():
    assert h2h_stats_for("Canada", "Egypt", None) == H2H_NEUTRAL
    assert h2h_stats_for("Canada", "Egypt",
                         {"team_1": "Canada", "team_2": "Egypt", "matches": 0,
                          "wins_1": 0, "wins_2": 0, "draws": 0, "gd_1": 0.0}) == H2H_NEUTRAL
