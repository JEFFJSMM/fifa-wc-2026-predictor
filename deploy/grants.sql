-- Permisos que necesita el service principal de la app
-- (app-4ghiit fifa-wc-2026-predictor, id 13d449ab-c727-44b6-88f9-89eb887379a0)
-- sobre las tablas de estado nuevas. Ejecutar en la Fase 4 antes de desplegar.
GRANT SELECT ON TABLE workspace.fifa_wc_gold.team_state_current
  TO `13d449ab-c727-44b6-88f9-89eb887379a0`;
GRANT SELECT ON TABLE workspace.fifa_wc_gold.h2h_state
  TO `13d449ab-c727-44b6-88f9-89eb887379a0`;
