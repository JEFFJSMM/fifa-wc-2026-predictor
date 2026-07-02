"""Pruebas de los helpers de la app: entrada al modelo, combinación de
predicciones y manejo de errores."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app_helpers import (INT_COLS, build_input_frame, combine_mirrored_proba,
                         friendly_error, verdict_for)
from fifa_features import FEATURE_COLUMNS, build_feature_row


def make_state(elo=1500.0):
    return {
        "elo": elo, "matches_played": 10, "winrate": 0.5, "rest_days": 4.0,
        "form5_ppg": 1.5, "form5_gf": 1.4, "form5_ga": 1.2, "form5_xg": 1.3,
        "form5_rating": 7.1, "form5_shots": 12.5,
        "squad_value": 5e7, "squad_age": 27.0, "squad_height": 181.0,
    }


H2H = {"n": 0, "winrate_a": 0.5, "gd_a": 0.0}


def test_build_input_frame_dtypes_match_signature():
    row = build_feature_row(make_state(1600), make_state(1500), H2H, 1)
    row_m = build_feature_row(make_state(1500), make_state(1600), H2H, 1)
    X = build_input_frame(row, row_m)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == 2
    for c in INT_COLS:
        assert X[c].dtype == "int64", c
    for c in set(FEATURE_COLUMNS) - set(INT_COLS):
        assert X[c].dtype == "float64", c


def test_combine_mirrored_proba_is_symmetric_and_sums_to_one():
    proba = [[0.5, 0.3, 0.2], [0.1, 0.3, 0.6]]
    p = combine_mirrored_proba(proba)
    assert p.sum() == pytest.approx(1.0)
    assert p[0] == pytest.approx((0.5 + 0.6) / 2)
    assert p[1] == pytest.approx(0.3)
    assert p[2] == pytest.approx((0.2 + 0.1) / 2)
    # invertir el orden de las filas espeja el resultado
    p_flip = combine_mirrored_proba([proba[1], proba[0]])
    assert p_flip[0] == pytest.approx(p[2])
    assert p_flip[2] == pytest.approx(p[0])


def test_friendly_error_never_leaks_technical_detail():
    cases = [
        RuntimeError("RESOURCE_DOES_NOT_EXIST: endpoint xyz not found"),
        RuntimeError("PERMISSION_DENIED: 403 forbidden"),
        RuntimeError("request timed out after 300s"),
        RuntimeError("KeyError: 'elo_zzz' traceback blah"),
    ]
    for exc in cases:
        msg = friendly_error(exc)
        assert "traceback" not in msg.lower()
        assert "RESOURCE_DOES_NOT_EXIST" not in msg
        assert "403" not in msg
        assert len(msg) > 20  # mensaje real, no vacío


def test_verdict_close_match():
    verdict, _ = verdict_for("Brazil", "Argentina", np.array([0.36, 0.30, 0.34]))
    assert "parejo" in verdict.lower()


def test_verdict_favorite_either_side():
    v_a, d_a = verdict_for("Brazil", "Argentina", np.array([0.55, 0.20, 0.25]))
    assert "Brazil" in v_a and "Argentina" in d_a
    v_b, d_b = verdict_for("Brazil", "Argentina", np.array([0.25, 0.20, 0.55]))
    assert "Argentina" in v_b and "Brazil" in d_b
