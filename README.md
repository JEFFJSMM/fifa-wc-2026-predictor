# ⚽ FIFA World Cup 2026 Predictor

Aplicación de Machine Learning para predecir resultados de partidos de la Copa del Mundo FIFA 2026.

## 🎯 Características

* **Comparación de Equipos**: Compara dos selecciones nacionales y obtén predicciones de probabilidad de victoria
* **Análisis Basado en Datos**: Utiliza datos históricos de rendimiento de jugadores y equipos
* **Modelo de ML**: Predicciones generadas por un modelo de machine learning entrenado con datos FIFA
* **Interfaz Intuitiva**: Aplicación web interactiva construida con Streamlit

## 🚀 Demo en Vivo

🔗 [Ver App en Vivo](https://fifa-wc-2026-predictor-7474659930274404.aws.databricksapps.com)

## 📊 Cómo Funciona

1. Ingresa el nombre de dos equipos (ej: Brazil, Argentina)
2. La app consulta datos históricos de rendimiento
3. Extrae 79 características por equipo (estadísticas de jugadores, resultados históricos, etc.)
4. El modelo de ML genera probabilidades de victoria para cada equipo
5. Muestra los resultados normalizados con indicadores visuales

## 🛠️ Tecnologías

* **Python 3.x**
* **Streamlit** - Framework de aplicación web
* **Databricks** - Plataforma de datos y ML
* **Unity Catalog** - Gestión de datos
* **Machine Learning** - Modelo predictivo entrenado

## 📁 Estructura del Proyecto

```
fifa-wc-2026-predictor/
├── app.py              # Aplicación principal Streamlit
├── app.yaml            # Configuración de Databricks App
├── requirements.txt    # Dependencias Python
└── README.md          # Este archivo
```

## 🎮 Ejemplo de Uso

```python
# La app consulta automáticamente estos datos:
# - Estadísticas de jugadores (edad, altura, rating, etc.)
# - Rendimiento histórico (goles, asistencias, victorias)
# - Métricas avanzadas (xG, xA, pases completados)
# - Resultados de partidos previos

# Ejemplo de comparación:
Equipo A: Brazil
Equipo B: Argentina

Resultado:
🇧🇷 Brazil: 70.4% - 🟢 Favorito
🇦🇷 Argentina: 29.6% - 🔴 Underdog
```

## 📈 Datos

Los datos provienen de:
* Tabla Unity Catalog: `workspace.fifa_wc_gold.player_performance_ml`
* Métricas de jugadores de competiciones FIFA
* Resultados históricos de partidos internacionales

## 🔧 Configuración

El app requiere:
1. Acceso a Databricks workspace
2. Permisos de lectura en Unity Catalog
3. Warehouse SQL configurado
4. Endpoint de modelo ML

## 📝 Notas

* Las predicciones son probabilísticas, no garantizan resultados reales
* El modelo se basa en datos históricos y puede no reflejar cambios recientes
* Para mejores resultados, usa nombres de equipos en inglés

## 👨‍💻 Autor

Proyecto desarrollado con Databricks Machine Learning y Databricks Apps V2

## 📄 Licencia

Este proyecto está disponible para uso educativo y demostrativo.

---

⚽ **¡Disfruta prediciendo la Copa del Mundo 2026!** 🏆