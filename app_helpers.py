"""Lógica pura de la app, separada de Streamlit para poder testearla."""
import traceback

import numpy as np
import pandas as pd

from fifa_features import FEATURE_COLUMNS

# Columnas long en el signature del modelo; el resto son double
INT_COLS = ["matches_played_a", "matches_played_b", "h2h_matches", "stage_knockout"]


def build_input_frame(row_ab, row_ba):
    """DataFrame de 2 filas (A-vs-B y B-vs-A) con los dtypes del signature."""
    X = pd.DataFrame([row_ab, row_ba], columns=FEATURE_COLUMNS).astype(float)
    return X.astype({c: "int64" for c in INT_COLS})


def combine_mirrored_proba(proba):
    """Promedia la predicción A-vs-B con la B-vs-A espejada.

    proba: array (2, 3) con [P(gana A), P(empate), P(gana B)] de cada fila.
    Devuelve un array (3,) simétrico que suma 1.
    """
    proba = np.asarray(proba, dtype=float)
    p_ab, p_ba = proba[0], proba[1]
    return np.array([(p_ab[0] + p_ba[2]) / 2,
                     (p_ab[1] + p_ba[1]) / 2,
                     (p_ab[2] + p_ba[0]) / 2])


def friendly_error(exc):
    """Mensaje apto para el usuario final; el detalle técnico va al log."""
    msg = str(exc)
    print(f"[predictor] error técnico: {msg}\n{traceback.format_exc()}")
    if "RESOURCE_DOES_NOT_EXIST" in msg or "does not exist" in msg.lower():
        return ("No encontramos el servicio de predicción. Es posible que esté en "
                "mantenimiento — intenta de nuevo en unos minutos.")
    if "PERMISSION" in msg.upper() or "403" in msg:
        return ("La app no tiene permiso para consultar los datos o el modelo. "
                "Contacta al administrador del workspace.")
    if "timed out" in msg.lower() or "timeout" in msg.lower():
        return ("El servicio está tardando más de lo normal en despertar. "
                "Espera un minuto y vuelve a intentarlo.")
    return ("Algo salió mal al calcular la predicción. Intenta de nuevo; si el "
            "problema persiste, revisa los registros de la app.")


def verdict_for(team_a, team_b, p, close_margin=5.0):
    """Texto del veredicto según el margen entre P(gana A) y P(gana B)."""
    margin = abs(p[0] - p[2]) * 100
    if margin < close_margin:
        return ("⚖️ Partido muy parejo",
                f"Ningún equipo supera al otro por más de {margin:.1f} puntos.")
    favorite, rival = (team_a, team_b) if p[0] > p[2] else (team_b, team_a)
    return (f"🏆 {favorite} parte como favorito",
            f"Ventaja de {margin:.1f} puntos de probabilidad sobre {rival}.")
