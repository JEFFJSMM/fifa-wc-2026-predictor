import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState, StatementParameterListItem
import time

# Configuración de la página
st.set_page_config(
    page_title="FIFA World Cup 2026 - Equipo vs Equipo",
    page_icon="⚽",
    layout="wide"
)

# Título principal
st.title("⚽ FIFA World Cup 2026 - Predicción Equipo vs Equipo")
st.markdown("""Compara dos equipos y descubre cuál tiene mayor probabilidad de ganar usando Machine Learning.""")

# Inicializar cliente de Databricks
@st.cache_resource
def get_workspace_client():
    return WorkspaceClient()

w = get_workspace_client()

# Función para convertir features a tipos numéricos
def convert_features_to_numeric(features_dict):
    numeric_features = {}
    position_map = {"Forward": 1, "Midfielder": 3, "Defender": 0, "Goalkeeper": 2}
    foot_map = {"Left": 0, "Right": 1, "Both": 1}
    
    for key, value in features_dict.items():
        if key == 'position':
            if isinstance(value, str):
                numeric_features[key] = position_map.get(value, 3)
            else:
                numeric_features[key] = int(value) if value is not None else 3
        elif key == 'preferred_foot':
            if isinstance(value, str):
                numeric_features[key] = foot_map.get(value, 1)
            else:
                numeric_features[key] = int(value) if value is not None else 1
        else:
            if value is None or value == '':
                numeric_features[key] = 0.0
            else:
                try:
                    numeric_features[key] = float(value)
                except (ValueError, TypeError):
                    numeric_features[key] = 0.0
    
    return numeric_features

# Función para hacer predicciones
def predict_endpoint(endpoint_name, features_dict):
    try:
        numeric_features = convert_features_to_numeric(features_dict)
        response = w.serving_endpoints.query(
            name=endpoint_name,
            dataframe_records=[numeric_features]
        )
        
        if hasattr(response, 'predictions'):
            return {"success": True, "prediction": response.predictions[0]}
        else:
            return {"success": False, "error": "No predictions in response"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Función para obtener estadísticas de un equipo
def get_team_features(team_name):
    try:
        query = """
        SELECT
            AVG(CAST(age AS DOUBLE)) as avg_age,
            AVG(CAST(height_cm AS DOUBLE)) as avg_height,
            AVG(CAST(weight_kg AS DOUBLE)) as avg_weight,
            AVG(CAST(market_value_eur AS DOUBLE)) as avg_market_value,
            AVG(CAST(minutes_played AS DOUBLE)) as avg_minutes,
            AVG(CAST(shots AS DOUBLE)) as avg_shots,
            AVG(CAST(shots_on_target AS DOUBLE)) as avg_shots_on_target,
            AVG(CAST(expected_goals_xg AS DOUBLE)) as avg_xg,
            AVG(CAST(expected_assists_xa AS DOUBLE)) as avg_xa,
            AVG(CAST(successful_passes AS DOUBLE)) as avg_passes,
            AVG(CAST(pass_accuracy AS DOUBLE)) as avg_pass_acc,
            AVG(CAST(dribbles_attempted AS DOUBLE)) as avg_dribbles,
            AVG(CAST(successful_dribbles AS DOUBLE)) as avg_succ_dribbles,
            AVG(CAST(tackles AS DOUBLE)) as avg_tackles,
            AVG(CAST(interceptions AS DOUBLE)) as avg_interceptions,
            AVG(CAST(player_rating AS DOUBLE)) as avg_rating,
            AVG(CAST(performance_score AS DOUBLE)) as avg_performance,
            COUNT(*) as games_played,
            SUM(CAST(goals AS INT)) as total_goals,
            SUM(CAST(assists AS INT)) as total_assists,
            SUM(CASE WHEN match_result = 'W' THEN 1 ELSE 0 END) as total_wins,
            AVG(CAST(goals_team AS DOUBLE)) as avg_goals_per_game
        FROM workspace.fifa_wc_gold.player_performance_ml
        WHERE LOWER(team) = LOWER(:team_name)
        """

        statement = w.statement_execution.execute_statement(
            warehouse_id="f9bb0b517b9fc8ba",
            statement=query,
            catalog="workspace",
            schema="fifa_wc_gold",
            parameters=[StatementParameterListItem(name="team_name", value=team_name)]
        )
        
        max_wait = 30
        waited = 0
        while statement.status.state in [StatementState.PENDING, StatementState.RUNNING] and waited < max_wait:
            time.sleep(0.5)
            waited += 0.5
            statement = w.statement_execution.get_statement(statement.statement_id)
        
        if statement.status.state == StatementState.SUCCEEDED and statement.result:
            if statement.result.data_array and len(statement.result.data_array) > 0:
                row = statement.result.data_array[0]
                
                # Verificar que hay datos (games_played > 0)
                games_played = int(float(row[17])) if row[17] and float(row[17]) > 0 else 0
                
                if games_played == 0:
                    return {"success": False, "error": f"No se encontraron datos para {team_name}"}
                
                # Extraer estadísticas con valores por defecto
                avg_age = float(row[0]) if row[0] else 28.0
                avg_height = int(float(row[1])) if row[1] else 180
                avg_weight = int(float(row[2])) if row[2] else 75
                avg_market_value = float(row[3]) if row[3] else 50000000
                avg_minutes = float(row[4]) if row[4] else 85.0
                avg_shots = float(row[5]) if row[5] else 15.0
                avg_shots_target = float(row[6]) if row[6] else 9.0
                avg_xg = float(row[7]) if row[7] else 1.5
                avg_xa = float(row[8]) if row[8] else 0.5
                avg_passes = float(row[9]) if row[9] else 250.0
                avg_pass_acc = float(row[10]) if row[10] else 75.0
                avg_dribbles = float(row[11]) if row[11] else 12.0
                avg_succ_dribbles = float(row[12]) if row[12] else 8.0
                avg_tackles = float(row[13]) if row[13] else 18.0
                avg_interceptions = float(row[14]) if row[14] else 10.0
                avg_rating = float(row[15]) if row[15] else 7.5
                avg_performance = float(row[16]) if row[16] else 75.0
                total_goals = int(float(row[18])) if row[18] else 0
                total_assists = int(float(row[19])) if row[19] else 0
                total_wins = int(float(row[20])) if row[20] else 0
                avg_goals_game = float(row[21]) if row[21] else 2.0
                
                win_rate = total_wins / games_played if games_played > 0 else 0.5
                
                # Crear diccionario de features (79 features requeridas por el modelo)
                features = {
                    "position": 3,  # Midfielder promedio
                    "age": avg_age,
                    "jersey_number": 10,
                    "height_cm": avg_height,
                    "weight_kg": avg_weight,
                    "preferred_foot": 1,  # Right
                    "market_value_eur": avg_market_value / 11,  # Por jugador
                    "goals_team": int(avg_goals_game),
                    "goals_opponent": 1,
                    "minutes_played": int(avg_minutes),
                    "assists": int(total_assists / games_played) if games_played > 0 else 0,
                    "shots": int(avg_shots),
                    "shots_on_target": int(avg_shots_target),
                    "expected_goals_xg": avg_xg,
                    "expected_assists_xa": avg_xa,
                    "key_passes": int(avg_passes * 0.1),
                    "successful_passes": int(avg_passes),
                    "total_passes": int(avg_passes / (avg_pass_acc/100)) if avg_pass_acc > 0 else int(avg_passes),
                    "pass_accuracy": avg_pass_acc,
                    "dribbles_attempted": int(avg_dribbles),
                    "successful_dribbles": int(avg_succ_dribbles),
                    "crosses": 15,
                    "successful_crosses": 5,
                    "tackles": int(avg_tackles),
                    "interceptions": int(avg_interceptions),
                    "clearances": 15,
                    "blocks": 8,
                    "aerial_duels_won": 20,
                    "aerial_duels_lost": 15,
                    "recoveries": 25,
                    "defensive_actions": int(avg_tackles + avg_interceptions),
                    "fouls_committed": 12,
                    "fouls_suffered": 10,
                    "yellow_cards": 2,
                    "red_cards": 0,
                    "offsides": 3,
                    "saves": 0,
                    "save_percentage": 0.0,
                    "punches": 0,
                    "clean_sheet": 0,
                    "goals_conceded": 1,
                    "penalty_saves": 0,
                    "distance_covered_km": 110.0,
                    "sprint_distance_km": 15.0,
                    "top_speed_kmh": 32.0,
                    "accelerations": 150,
                    "decelerations": 150,
                    "stamina_score": 80.0,
                    "player_rating": avg_rating,
                    "performance_score": avg_performance,
                    "offensive_contribution": avg_rating * 0.6,
                    "defensive_contribution": avg_rating * 0.4,
                    "possession_impact": avg_pass_acc / 100.0,
                    "pressure_resistance": 0.75,
                    "creativity_score": avg_xa * 10,
                    "consistency_score": 0.75,
                    "clutch_performance_score": 0.7,
                    "goal_difference": int(total_goals - games_played) if games_played > 0 else 0,
                    "goal_per_shot": avg_shots_target / avg_shots if avg_shots > 0 else 0.15,
                    "assist_per_key_pass": 0.05,
                    "high_value_player": 1 if avg_market_value > 50000000 else 0,
                    "player_games_played": games_played,
                    "player_total_goals": int(total_goals / 11) if games_played > 0 else 0,
                    "player_total_assists": int(total_assists / 11) if games_played > 0 else 0,
                    "player_avg_rating": avg_rating,
                    "player_avg_minutes": avg_minutes,
                    "player_goals_per_game": total_goals / games_played if games_played > 0 else 0.0,
                    "team_total_wins": total_wins,
                    "team_total_goals": total_goals,
                    "team_games_played": games_played,
                    "team_win_rate": win_rate,
                    "player_recent_goals": int(total_goals / games_played * 5) if games_played > 0 else 0,
                    "player_recent_avg_rating": avg_rating,
                    "player_goals_vs_opponent": 2,
                    "position_avg_goals": 0.2,
                    "position_avg_assists": 0.15,
                    "position_avg_rating": 7.3,
                    "goals_vs_position_avg": 0.0,
                    "rating_vs_position_avg": 0.2
                }
                
                return {
                    "success": True,
                    "features": features,
                    "games": games_played,
                    "wins": total_wins,
                    "win_rate": win_rate,
                    "avg_rating": avg_rating
                }
            else:
                return {"success": False, "error": f"No se encontraron datos para {team_name}"}
        else:
            error_msg = statement.status.error.message if statement.status.error else "Error desconocido"
            return {"success": False, "error": f"Error en consulta: {error_msg}"}
    except Exception as e:
        return {"success": False, "error": f"Excepción: {str(e)[:200]}"}

# ========================
# INTERFAZ PRINCIPAL
# ========================

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏆 Equipo A")
    team_a = st.text_input("Nombre del Equipo A", value="Brazil", key="team_a")

with col2:
    st.subheader("🏆 Equipo B")
    team_b = st.text_input("Nombre del Equipo B", value="Argentina", key="team_b")

st.markdown("---")

if st.button("⚔️ Comparar Equipos", use_container_width=True, type="primary"):
    if not team_a or not team_b:
        st.error("❌ Por favor ingresa los nombres de ambos equipos")
    elif team_a.lower() == team_b.lower():
        st.error("❌ Los equipos deben ser diferentes")
    else:
        with st.spinner(f"🔍 Analizando {team_a} vs {team_b}..."):
            # Obtener datos de ambos equipos
            col_status1, col_status2 = st.columns(2)
            
            with col_status1:
                st.info(f"Consultando datos de {team_a}...")
            with col_status2:
                st.info(f"Consultando datos de {team_b}...")
            
            team_a_data = get_team_features(team_a)
            team_b_data = get_team_features(team_b)
            
            # Verificar que ambos equipos se encontraron
            errors = []
            if not team_a_data["success"]:
                errors.append(f"{team_a}: {team_a_data['error']}")
            if not team_b_data["success"]:
                errors.append(f"{team_b}: {team_b_data['error']}")
            
            if errors:
                st.error("❌ " + " | ".join(errors))
            else:
                # Mostrar información de los equipos
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.success(f"✅ {team_a} encontrado")
                    st.metric("Partidos", team_a_data['games'])
                    st.metric("Victorias", team_a_data['wins'])
                    st.metric("Win Rate", f"{team_a_data['win_rate']:.1%}")
                    st.metric("Rating Promedio", f"{team_a_data['avg_rating']:.2f}")
                
                with col_info2:
                    st.success(f"✅ {team_b} encontrado")
                    st.metric("Partidos", team_b_data['games'])
                    st.metric("Victorias", team_b_data['wins'])
                    st.metric("Win Rate", f"{team_b_data['win_rate']:.1%}")
                    st.metric("Rating Promedio", f"{team_b_data['avg_rating']:.2f}")
                
                st.markdown("---")
                
                # Hacer predicciones
                with st.spinner("🔮 Calculando probabilidades..."):
                    result_a = predict_endpoint("fifa-team-win-predictor", team_a_data["features"])
                    result_b = predict_endpoint("fifa-team-win-predictor", team_b_data["features"])
                
                if result_a["success"] and result_b["success"]:
                    prob_a = result_a["prediction"]
                    prob_b = result_b["prediction"]
                    
                    # Normalizar probabilidades (suma = 1)
                    total = prob_a + prob_b
                    if total > 0:
                        prob_a_norm = prob_a / total
                        prob_b_norm = prob_b / total
                    else:
                        prob_a_norm = 0.5
                        prob_b_norm = 0.5
                    
                    st.success("✅ Predicción Completada")
                    
                    # Mostrar resultados
                    st.markdown("### 📊 Probabilidades de Victoria")
                    
                    col_result1, col_result2 = st.columns(2)
                    
                    with col_result1:
                        st.markdown(f"#### {team_a}")
                        st.metric("Probabilidad de ganar", f"{prob_a_norm * 100:.1f}%", 
                                 delta=f"{(prob_a_norm - 0.5) * 100:+.1f}% vs empate")
                        st.progress(prob_a_norm)
                        
                        if prob_a_norm >= 0.60:
                            st.success("🟢 Favorito")
                        elif prob_a_norm >= 0.45:
                            st.warning("🟡 Parejo")
                        else:
                            st.error("🔴 Underdog")
                    
                    with col_result2:
                        st.markdown(f"#### {team_b}")
                        st.metric("Probabilidad de ganar", f"{prob_b_norm * 100:.1f}%",
                                 delta=f"{(prob_b_norm - 0.5) * 100:+.1f}% vs empate")
                        st.progress(prob_b_norm)
                        
                        if prob_b_norm >= 0.60:
                            st.success("🟢 Favorito")
                        elif prob_b_norm >= 0.45:
                            st.warning("🟡 Parejo")
                        else:
                            st.error("🔴 Underdog")
                    
                    # Determinar ganador
                    st.markdown("---")
                    margin = abs(prob_a_norm - prob_b_norm) * 100
                    
                    if prob_a_norm > prob_b_norm:
                        if margin >= 15:
                            st.success(f"### 🏆 {team_a} es el CLARO FAVORITO ({margin:.1f}% de ventaja)")
                        else:
                            st.info(f"### 🎯 {team_a} tiene ligera ventaja ({margin:.1f}% de diferencia)")
                    elif prob_b_norm > prob_a_norm:
                        if margin >= 15:
                            st.success(f"### 🏆 {team_b} es el CLARO FAVORITO ({margin:.1f}% de ventaja)")
                        else:
                            st.info(f"### 🎯 {team_b} tiene ligera ventaja ({margin:.1f}% de diferencia)")
                    else:
                        st.warning("### ⚖️ EMPATE TÉCNICO - Partido muy parejo")
                else:
                    error_msgs = []
                    if not result_a["success"]:
                        error_msgs.append(f"{team_a}: {result_a['error']}")
                    if not result_b["success"]:
                        error_msgs.append(f"{team_b}: {result_b['error']}")
                    
                    st.error("❌ Error en predicción: " + " | ".join(error_msgs))

# Footer
st.markdown("---")
st.markdown("""<div style='text-align: center'>
<p>🤖 Powered by Databricks Machine Learning | ⚽ FIFA World Cup 2026</p>
<p>Modelos entrenados con datos históricos de la Copa del Mundo</p>
</div>""", unsafe_allow_html=True)
