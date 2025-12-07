from google.cloud import bigquery
import pandas as pd
import os
import json
import unicodedata
from google.oauth2 import service_account
from pandas_gbq import to_gbq

# --- CONFIGURATION ---
PROJECT_ID = os.environ["GCP_PROJECT_ID"]
DATASET_ID = "historic_datasets"
TABLE_MATCHS = f"{DATASET_ID}.matchs_clean"
TABLE_REFS = f"{DATASET_ID}.referee_details" # Nouvelle table

# --- MAPPING EXISTANT ---
COLUMN_MAPPING = {
    'Div': 'division', 'Date': 'date', 'Time': 'time',
    'HomeTeam': 'home_team', 'AwayTeam': 'away_team', 'Referee': 'referee',
    'FTHG': 'full_time_home_goals', 'FTAG': 'full_time_away_goals', 'FTR': 'full_time_result',
    'HTHG': 'half_time_home_goals', 'HTAG': 'half_time_away_goals', 'HTR': 'half_time_result',
    'HS': 'home_shots', 'AS': 'away_shots', 'HST': 'home_shots_on_target', 'AST': 'away_shots_on_target',
    'HF': 'home_fouls', 'AF': 'away_fouls', 'HC': 'home_corners', 'AC': 'away_corners',
    'HY': 'home_yellow_cards', 'AY': 'away_yellow_cards', 'HR': 'home_red_cards', 'AR': 'away_red_cards',
    'B365H': 'bet365_home_win_odds', 'B365D': 'bet365_draw_odds', 'B365A': 'bet365_away_win_odds',
    'PSH': 'pinnacle_home_win_odds', 'PSD': 'pinnacle_draw_odds', 'PSA': 'pinnacle_away_win_odds',
    'WHH': 'william_hill_home_win_odds', 'WHD': 'william_hill_draw_odds', 'WHA': 'william_hill_away_win_odds',
    'MaxH': 'max_home_win_odds', 'MaxD': 'max_draw_odds', 'MaxA': 'max_away_win_odds',
    'AvgH': 'avg_home_win_odds', 'AvgD': 'avg_draw_odds', 'AvgA': 'avg_away_win_odds',
    'B365CH': 'bet365_closing_home_win_odds', 'B365CD': 'bet365_closing_draw_odds', 'B365CA': 'bet365_closing_away_win_odds',
    'PCH': 'pinnacle_closing_home_win_odds', 'PCD': 'pinnacle_closing_draw_odds', 'PCA': 'pinnacle_closing_away_win_odds',
    'MaxCH': 'max_closing_home_win_odds', 'MaxCD': 'max_closing_draw_odds', 'MaxCA': 'max_closing_away_win_odds',
    'AvgCH': 'avg_closing_home_win_odds', 'AvgCD': 'avg_closing_draw_odds', 'AvgCA': 'avg_closing_away_win_odds',
    'B365>2.5': 'bet365_over_25_goals', 'B365<2.5': 'bet365_under_25_goals',
    'P>2.5': 'pinnacle_over_25_goals', 'P<2.5': 'pinnacle_under_25_goals',
    'Max>2.5': 'max_over_25_goals', 'Max<2.5': 'max_under_25_goals',
    'Avg>2.5': 'avg_over_25_goals', 'Avg<2.5': 'avg_under_25_goals',
    'AHh': 'asian_handicap_size', 
    'B365AHH': 'bet365_asian_handicap_home_odds', 'B365AHA': 'bet365_asian_handicap_away_odds',
    'PAHH': 'pinnacle_asian_handicap_home_odds', 'PAHA': 'pinnacle_asian_handicap_away_odds',
    'MaxAHH': 'max_asian_handicap_home_odds', 'MaxAHA': 'max_asian_handicap_away_odds',
    'AvgAHH': 'avg_asian_handicap_home_odds', 'AvgAHA': 'avg_asian_handicap_away_odds',
}

def normalize_name(name):
    """Nettoie le nom des arbitres (enlève accents, initiales prénoms, majuscules)"""
    if not isinstance(name, str): return "INCONNU"
    # Enlève les accents
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    # Majuscules
    name = name.upper().strip()
    # Gestion des initiales (ex: "C. TURPIN" -> "TURPIN")
    if ". " in name:
        name = name.split(". ")[1]
    return name

def get_ligue1_data(start_year=1993, end_year=2025):
    all_seasons_dfs = []
    valid_cols = list(set(COLUMN_MAPPING.values())) + ['season', 'division']
    
    for year in range(start_year, end_year + 1):
        code_start = str(year)[-2:]
        code_end = str(year + 1)[-2:]
        season_code = f"{code_start}{code_end}"
        season_label = f"{year}-{year+1}"
        url = f"https://www.football-data.co.uk/mmz4281/{season_code}/F1.csv"
        
        try:
            print(f"Chargement {season_label}...")
            df = pd.read_csv(url, encoding='latin-1', on_bad_lines='skip')
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.assign(season=season_label, division='Ligue 1')
            df = df.rename(columns=COLUMN_MAPPING)
            df = df.loc[:, ~df.columns.duplicated()]
            cols_to_keep = [c for c in df.columns if c in valid_cols]
            df = df[cols_to_keep]
            
            if not df.empty:
                all_seasons_dfs.append(df)
        except Exception as e:
            print(f"⚠️ Ignoré {season_label} : {e}")

    if all_seasons_dfs:
        full_df = pd.concat(all_seasons_dfs, ignore_index=True)
        # Fix Dates
        full_df['date'] = full_df['date'].astype(str)
        dates_v1 = pd.to_datetime(full_df['date'], format='%d/%m/%y', errors='coerce')
        mask_nan = dates_v1.isna()
        dates_v2 = pd.to_datetime(full_df.loc[mask_nan, 'date'], format='%d/%m/%Y', errors='coerce')
        full_df['date'] = dates_v1.fillna(dates_v2)
        full_df = full_df.dropna(subset=['date'])
        full_df['date'] = full_df['date'].dt.date 
        return full_df
    else:
        return pd.DataFrame()

def process_referee_table(df):
    """Crée la table référentiel des arbitres avec dates et stats"""
    print("🔨 Création de la table Arbitres...")
    
    # On s'assure d'avoir les colonnes nécessaires
    if 'referee' not in df.columns:
        print("⚠️ Pas de colonne 'referee' trouvée.")
        return pd.DataFrame()

    # Sélection des colonnes utiles
    cols = ['season', 'date', 'time', 'home_team', 'away_team', 'referee']
    # On garde aussi les stats disciplinaires si dispos pour l'historique
    stats_cols = ['home_yellow_cards', 'away_yellow_cards', 'home_red_cards', 'away_red_cards']
    
    available_cols = [c for c in cols + stats_cols if c in df.columns]
    df_refs = df[available_cols].copy()
    
    # Normalisation
    df_refs['referee_clean'] = df_refs['referee'].apply(normalize_name)
    
    # Calcul de la journée (approximatif basé sur l'ordre chronologique par équipe)
    # On trie par date
    df_refs = df_refs.sort_values('date')
    # On numérote les matchs par saison et par équipe
    df_refs['match_rank_home'] = df_refs.groupby(['season', 'home_team']).cumcount() + 1
    # On prend une moyenne pour estimer la "Journée" globale du championnat
    df_refs['journee'] = df_refs.groupby('season')['date'].rank(method='dense').astype(int)

    # Création du timestamp précis
    # Si 'time' est vide, on met 20:00 par défaut
    df_refs['time'] = df_refs['time'].fillna('20:00')
    # On convertit en string pour BigQuery (DATETIME ou TIMESTAMP)
    df_refs['full_date'] = pd.to_datetime(df_refs['date'].astype(str) + ' ' + df_refs['time'].astype(str), errors='coerce')
    
    return df_refs

def update_standings_table(credentials, project_id):
    """Exécute le SQL pour la table classement (code existant)"""
    from google.cloud import bigquery
    print("🔄 Mise à jour classement SQL...")
    client = bigquery.Client(credentials=credentials, project=project_id)
    try:
        with open("update_classement.sql", "r") as file:
            sql_query = file.read()
        query_job = client.query(sql_query)
        query_job.result()
        print("✅ Classement mis à jour.")
    except Exception as e:
        print(f"❌ Erreur SQL : {e}")

# --- FONCTION PRINCIPALE ---
def main():
    print("🚀 Démarrage ETL...")
    
    # 1. Get Data
    df = get_ligue1_data(start_year=1993, end_year=2025)
    if df.empty: return

    # 2. Auth
    service_account_info = json.loads(os.environ["GCP_SA_KEY"])
    credentials = service_account.Credentials.from_service_account_info(service_account_info)

    # 3. Export MATCHS CLEAN (Table Principale)
    try:
        to_gbq(df, TABLE_MATCHS, project_id=PROJECT_ID, credentials=credentials, if_exists='replace', chunksize=None)
        print(f"✅ Table {TABLE_MATCHS} exportée.")
    except Exception as e:
        print(f"❌ Erreur export Matchs : {e}")

    # 4. Export REFEREES (Nouvelle Table)
    df_refs = process_referee_table(df)
    if not df_refs.empty:
        try:
            to_gbq(df_refs, TABLE_REFS, project_id=PROJECT_ID, credentials=credentials, if_exists='replace', chunksize=None)
            print(f"✅ Table {TABLE_REFS} exportée.")
        except Exception as e:
            print(f"❌ Erreur export Arbitres : {e}")

    # 5. Update SQL Derived Tables
    update_standings_table(credentials, PROJECT_ID)

if __name__ == "__main__":
    main()