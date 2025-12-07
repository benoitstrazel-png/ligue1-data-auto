-- Fichier: update_classement.sql
CREATE OR REPLACE TABLE `ligue1-data.historic_datasets.classement_live` AS

WITH matchs_unpivoted AS (
  SELECT
    season,
    date,
    PARSE_TIMESTAMP('%Y-%m-%d %H:%M', CONCAT(CAST(date AS STRING), ' ', IFNULL(time, '00:00'))) as match_timestamp,
    home_team AS team,
    CASE WHEN full_time_result = 'H' THEN 3 WHEN full_time_result = 'D' THEN 1 ELSE 0 END AS points,
    full_time_home_goals AS but_pour,
    full_time_away_goals AS but_contre,
    (full_time_home_goals - full_time_away_goals) AS diff,
    CASE WHEN full_time_result = 'H' THEN 1 ELSE 0 END AS victoire,
    CASE WHEN full_time_result = 'D' THEN 1 ELSE 0 END AS nul,
    CASE WHEN full_time_result = 'A' THEN 1 ELSE 0 END AS defaite
  FROM `ligue1-data.historic_datasets.matchs_clean`
  WHERE full_time_result IS NOT NULL

  UNION ALL

  SELECT
    season,
    date as date_match,
    PARSE_TIMESTAMP('%Y-%m-%d %H:%M', CONCAT(CAST(date AS STRING), ' ', IFNULL(time, '00:00'))) as match_timestamp,
    away_team AS team,
    CASE WHEN full_time_result = 'A' THEN 3 WHEN full_time_result = 'D' THEN 1 ELSE 0 END AS points,
    full_time_away_goals AS but_pour,
    full_time_home_goals AS but_contre,
    (full_time_away_goals - full_time_home_goals) AS diff,
    CASE WHEN full_time_result = 'A' THEN 1 ELSE 0 END AS victoire,
    CASE WHEN full_time_result = 'D' THEN 1 ELSE 0 END AS nul,
    CASE WHEN full_time_result = 'H' THEN 1 ELSE 0 END AS defaite
  FROM `ligue1-data.historic_datasets.matchs_clean`
  WHERE full_time_result IS NOT NULL
),

classement_cumule AS (
  SELECT
    *,
    ROW_NUMBER() OVER(PARTITION BY season, team ORDER BY match_timestamp) as journee_team,
    SUM(points) OVER(PARTITION BY season, team ORDER BY match_timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS total_points,
    SUM(but_pour) OVER(PARTITION BY season, team ORDER BY match_timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS total_bp,
    SUM(but_contre) OVER(PARTITION BY season, team ORDER BY match_timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS total_bc,
    SUM(diff) OVER(PARTITION BY season, team ORDER BY match_timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS total_diff,
    SUM(victoire) OVER(PARTITION BY season, team ORDER BY match_timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS total_V,
    SUM(nul) OVER(PARTITION BY season, team ORDER BY match_timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS total_N,
    SUM(defaite) OVER(PARTITION BY season, team ORDER BY match_timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS total_D
  FROM matchs_unpivoted
)

SELECT 
  season as saison,
  date as date_match,
  match_timestamp,
  team as equipe,
  journee_team,
  total_points,
  total_bp,
  total_bc,
  total_diff,
  total_V,
  total_N,
  total_D
FROM classement_cumule
ORDER BY saison DESC, match_timestamp DESC, total_points DESC