# ⚽ FIFA World Cup 2026 Predictor

Aplicación de Machine Learning que estima probabilidades de victoria, empate y derrota entre dos selecciones, con un modelo a nivel partido entrenado y servido en Databricks.

> **Nota de honestidad**: el dataset (Kaggle) es sintético — los resultados de los partidos son esencialmente aleatorios, así que las probabilidades del modelo son honestas respecto a esos datos pero no pronósticos reales del Mundial. El valor del proyecto está en la metodología: pipeline sin data leakage, validación temporal y despliegue completo end-to-end.

## 🚀 Demo en Vivo

🔗 [Ver App en Vivo](https://fifa-wc-2026-predictor-7474659930274404.aws.databricksapps.com)

## 🎯 Qué hace

1. Eliges dos selecciones (48 disponibles) y el tipo de partido (grupos o eliminatoria).
2. La app construye 26 features pre-partido desde tablas de estado en Unity Catalog: **Elo dinámico**, **forma reciente** (últimos 5 partidos), **historial directo (H2H)**, descanso y calidad de plantilla.
3. El modelo (LightGBM de 3 clases, servido en Model Serving) devuelve P(gana A), P(empate) y P(gana B). La predicción se **simetriza**: se promedia A-vs-B con B-vs-A espejado, así el orden de selección no afecta el resultado.
4. La interfaz explica la predicción: barra de probabilidades, tiles de estado por equipo, comparación de forma, historial directo e importancia de variables del modelo.

## 🏗️ Arquitectura

```
Kaggle dataset ─► Medallion (bronze/silver/gold)          [training/fifa_wc_medallion_pipeline.py]
                        │
                        ▼
        Agregación equipo-partido + features pre-partido
        (Elo, forma, H2H, descanso, plantilla)             [training/train_local.py]
                        │
          ├─► Validación temporal 70/10/20 + tuning
          │   (LogReg vs XGBoost vs LightGBM, MLflow)
          ├─► Modelo v5 en Unity Catalog Model Registry
          │   (workspace.fifa_wc_gold.fifa_team_win_predictor)
          └─► Tablas de estado para servir:
              team_state_current · h2h_state
                        │
                        ▼
   Model Serving endpoint (fifa-team-win-predictor)
                        │
                        ▼
        Databricks App (Streamlit)                          [app.py]
        features construidas con el módulo compartido       [fifa_features.py]
```

**Contrato clave**: `fifa_features.py` define `FEATURE_COLUMNS` y `build_feature_row()`, usados *idénticos* por el entrenamiento y por la app. Cualquier cambio de features exige reentrenar y redesplegar juntos.

## 📁 Estructura del proyecto

```
fifa-wc-2026-predictor/
├── app.py                     # App Streamlit (UI + llamada al endpoint)
├── app_helpers.py             # Lógica pura de la app (testeable)
├── fifa_features.py           # Contrato de features entrenamiento ↔ serving
├── app.yaml                   # Config Databricks App (env desde resources)
├── requirements.txt           # Dependencias pinneadas de la app
├── .streamlit/config.toml     # Tema visual
├── training/
│   ├── fifa_wc_medallion_pipeline.py  # Pipeline de datos bronze/silver/gold
│   ├── train_local.py         # Pipeline de entrenamiento completo (local)
│   ├── register_v5.py         # Registro del modelo de producción
│   ├── train_match_predictor.py       # Versión notebook del pipeline
│   ├── phase1_results.json    # Métricas modelo viejo vs nuevo
│   └── feature_importance.json
├── tests/                     # 12 tests del contrato de features y helpers
└── deploy/                    # Resources de la app + grants SQL
```

## 📊 Modelo: viejo vs nuevo

| | Modelo original (v1) | Modelo actual (v5) |
|---|---|---|
| Unidad | jugador-partido | **partido** (A vs B) |
| Target | victoria binaria (sin empates) | **3 clases** (victoria/empate/derrota) |
| Features | 79 (30 hardcodeadas, incluía el marcador del propio partido → **leakage**) | 26 pre-partido (Elo, forma, H2H) |
| Validación | split aleatorio | **temporal** (entrena pasado, valida futuro) |
| Algoritmo | LightGBM fijo | LightGBM vs XGBoost vs LogReg con tuning (MLflow) |

En el test temporal honesto (210 partidos), ambos modelos rondan el azar — es la naturaleza del dataset sintético. La diferencia es que el nuevo lo reporta honestamente, mientras el AUC alto del original era un artefacto del leakage. Detalle completo en `training/phase1_results.json`.

## 🛠️ Desarrollo

```bash
# Tests (lógica pura, sin red)
.venv/Scripts/python.exe -m pytest tests -q

# App local contra el modelo de UC (sin tocar el endpoint)
PYTHONUTF8=1 MODEL_LOCAL_URI="models:/workspace.fifa_wc_gold.fifa_team_win_predictor/5" \
  .venv/Scripts/python.exe -m streamlit run app.py

# Reentrenar y registrar una nueva versión del modelo
PYTHONUTF8=1 .venv/Scripts/python.exe training/train_local.py
```

⚠️ El entrenamiento corre **localmente**: los jobs serverless de este workspace (Free Edition) no tienen acceso al almacenamiento de modelos de Unity Catalog. Ver `CLAUDE.md` para los detalles de esta restricción.

## 🚢 Despliegue

```bash
databricks apps update fifa-wc-2026-predictor --json @deploy/app_resources.json  # resources (endpoint CAN_QUERY + warehouse CAN_USE)
# ejecutar deploy/grants.sql en el SQL warehouse                                 # permisos de tablas para el service principal
databricks sync --full . /Users/jsierram96@gmail.com/fifa-wc-2026-predictor
databricks apps deploy fifa-wc-2026-predictor \
  --source-code-path /Workspace/Users/jsierram96@gmail.com/fifa-wc-2026-predictor
```

Las credenciales de Kaggle del pipeline de datos viven en Databricks Secrets (scope `kaggle`), nunca en el código.

## 🛠️ Tecnologías

* **Python** · **Streamlit** · **LightGBM / XGBoost / scikit-learn**
* **Databricks**: Apps, Model Serving, Unity Catalog, MLflow, SQL Warehouse
* Arquitectura de datos **Medallion** (bronze → silver → gold)

## 📄 Licencia

Proyecto educativo y demostrativo.
