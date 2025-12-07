from google.cloud import bigquery
import pandas as pd
import os
import json
from google.oauth2 import service_account
from pandas_gbq import to_gbq

# --- TA FONCTION DE RÉCUPÉRATION (Code validé précédemment) ---
# J'ai remis ton mapping et ta fonction ici pour que le script soit autonome
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
            df = df.loc[:, ~df.columns.duplicated()] # Fix 1
            df = df.assign(season=season_label, division='Ligue 1')
            df = df.rename(columns=COLUMN_MAPPING)
            df = df.loc[:, ~df.columns.duplicated()] # Fix 2
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
        # Conversion explicite pour BigQuery (Date uniquement, pas datetime)
        full_df['date'] = full_df['date'].dt.date 
        return full_df
    else:
        return pd.DataFrame()

# --- EXPORT VERS BIGQUERY ---
def main():
    # 1. Récupération des données
    print("🚀 Démarrage du scraping...")
    df = get_ligue1_data(start_year=1993, end_year=2025)
    
    if df.empty:
        print("❌ Aucune donnée récupérée.")
        return

    print(f"✅ {len(df)} lignes récupérées. Préparation de l'envoi vers BigQuery...")

    # 2. Configuration BigQuery via GitHub Secrets
    # On récupère la clé JSON depuis la variable d'environnement (configurée plus tard dans GitHub)
    service_account_info = json.loads(os.environ["GCP_SA_KEY"])
    project_id = os.environ["GCP_PROJECT_ID"]
    dataset_table = "historic_datasets.matchs_clean" # <--- ADAPTE AVEC TON DATASET.TABLE

    credentials = service_account.Credentials.from_service_account_info(service_account_info)

    # 3. Envoi (Mode 'replace' pour tout écraser et remettre à jour proprement chaque lundi)
    try:
        to_gbq(
            df,
            destination_table=dataset_table,
            project_id=project_id,
            credentials=credentials,
            if_exists='replace', # Ecrase la table existante
            chunksize=None # Automatique
        )
        print("🎉 Données exportées avec succès vers BigQuery !")
    except Exception as e:
        print(f"❌ Erreur lors de l'export BigQuery : {e}")

if __name__ == "__main__":
    main()

# ... (Ton code existant) ...

def update_standings_table(credentials, project_id):
    """
    Lit le fichier SQL et exécute la requête dans BigQuery
    pour mettre à jour la table de classement.
    """
    print("🔄 Mise à jour de la table classement_live...")
    
    # On initialise le client BigQuery
    client = bigquery.Client(credentials=credentials, project=project_id)
    
    # On lit le fichier SQL
    try:
        with open("update_classement.sql", "r") as file:
            sql_query = file.read()
            
        # On exécute la requête
        query_job = client.query(sql_query)
        query_job.result()  # On attend que la requête soit finie
        print("✅ Table classement_live mise à jour avec succès !")
        
    except FileNotFoundError:
        print("❌ Erreur : Le fichier update_classement.sql est introuvable.")
    except Exception as e:
        print(f"❌ Erreur lors de la mise à jour du classement : {e}")

# --- EXPORT VERS BIGQUERY ---
def main():
    # 1. Récupération des données
    print("🚀 Démarrage du scraping...")
    df = get_ligue1_data(start_year=1993, end_year=2025)
    
    if df.empty:
        print("❌ Aucune donnée récupérée.")
        return

    print(f"✅ {len(df)} lignes récupérées. Préparation de l'envoi vers BigQuery...")

    # 2. Configuration BigQuery
    service_account_info = json.loads(os.environ["GCP_SA_KEY"])
    project_id = os.environ["GCP_PROJECT_ID"]
    dataset_table = "historic_datasets.matchs_clean" # Adapte si besoin

    credentials = service_account.Credentials.from_service_account_info(service_account_info)

    # 3. Envoi des données brutes
    try:
        to_gbq(
            df,
            destination_table=dataset_table,
            project_id=project_id,
            credentials=credentials,
            if_exists='replace',
            chunksize=None
        )
        print("🎉 Données exportées avec succès vers BigQuery !")
        
        # 4. LANCEMENT DU CALCUL SQL (C'est ici qu'on chaîne l'étape)
        update_standings_table(credentials, project_id)
        
    except Exception as e:
        print(f"❌ Erreur critique : {e}")

if __name__ == "__main__":
    main()