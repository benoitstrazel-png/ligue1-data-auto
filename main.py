from google.cloud import bigquery
import pandas as pd
import os
import json
from pandas_gbq import to_gbq

# --- CONFIGURATION ---
PROJECT_ID = os.environ["GCP_PROJECT_ID"]
ENV_TYPE = os.environ.get("ENV_TYPE", "dev") 
DATASET_ID = "historic_datasets" if ENV_TYPE == "prod" else "historic_datasets_dev"

print(f"🔧 Environnement détecté : {ENV_TYPE.upper()} -> Dataset : {DATASET_ID}")

TABLE_MATCHS = f"{DATASET_ID}.matchs_clean"
TABLE_CALENDRIER = f"{DATASET_ID}.referentiel_calendrier"

# MAPPING ETENDU (Inclut les noms du CSV et les variantes historiques)
TEAM_MAPPING = {
    # Variantes CSV et autres vers Nom Standard BDD
    "Paris Saint-Germain": "Paris SG", "Paris S.G.": "Paris SG", "PSG": "Paris SG",
    "Olympique de Marseille": "Marseille", "Olympique Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon", "OL": "Lyon",
    "AS Monaco": "Monaco", 
    "LOSC Lille": "Lille", "Lille OSC": "Lille", "LOSC": "Lille",
    "RC Lens": "Lens",
    "Stade Rennais FC": "Rennes", "Stade Rennais": "Rennes",
    "OGC Nice": "Nice",
    "AS Saint-Étienne": "Saint Etienne", "AS Saint-Etienne": "Saint Etienne", "St Etienne": "Saint Etienne",
    "RC Strasbourg Alsace": "Strasbourg", "RC Strasbourg": "Strasbourg",
    "FC Nantes": "Nantes",
    "Stade de Reims": "Reims",
    "Montpellier HSC": "Montpellier",
    "Toulouse FC": "Toulouse",
    "Stade Brestois 29": "Brest", "Stade Brestois": "Brest",
    "FC Lorient": "Lorient",
    "Havre Athletic Club": "Le Havre", "Le Havre AC": "Le Havre",
    "FC Metz": "Metz",
    "AJ Auxerre": "Auxerre",
    "Angers SCO": "Angers",
    "Paris FC": "Paris FC"
}

# --- SOURCE 1 : HISTORIQUE (FOOTBALL-DATA) ---
def get_football_data_legacy(start_year, end_year):
    print("   Downloading Football-Data.co.uk (Legacy)...")
    # Mapping colonnes CSV Legacy
    COLUMN_MAPPING = {
        'Div': 'division', 'Date': 'date', 'Time': 'time',
        'HomeTeam': 'home_team', 'AwayTeam': 'away_team', 'Referee': 'referee',
        'FTHG': 'full_time_home_goals', 'FTAG': 'full_time_away_goals', 'FTR': 'full_time_result',
        'HTHG': 'half_time_home_goals', 'HTAG': 'half_time_away_goals', 'HTR': 'half_time_result',
        'HS': 'home_shots', 'AS': 'away_shots', 'HST': 'home_shots_on_target', 'AST': 'away_shots_on_target',
        'HF': 'home_fouls', 'AF': 'away_fouls', 'HC': 'home_corners', 'AC': 'away_corners',
        'HY': 'home_yellow_cards', 'AY': 'away_yellow_cards', 'HR': 'home_red_cards', 'AR': 'away_red_cards',
        'B365H': 'bet365_home_win_odds', 'B365D': 'bet365_draw_odds', 'B365A': 'bet365_away_win_odds',
        'PSH': 'pinnacle_home_win_odds', 'PSD': 'pinnacle_draw_odds', 'PSA': 'pinnacle_away_win_odds'
    }
    
    all_data = []
    valid_cols = list(set(COLUMN_MAPPING.values())) + ['season', 'division']
    
    for year in range(start_year, end_year + 1):
        code = f"{str(year)[-2:]}{str(year+1)[-2:]}"
        season_label = f"{year}-{year+1}"
        url = f"https://www.football-data.co.uk/mmz4281/{code}/F1.csv"
        try:
            df = pd.read_csv(url, encoding='latin-1', on_bad_lines='skip')
            df = df.rename(columns=COLUMN_MAPPING)
            df = df.assign(season=season_label, division='Ligue 1')
            
            # Harmonisation Date
            df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce').dt.date.astype(str)
            
            # Nettoyage Colonnes
            cols_to_keep = [c for c in df.columns if c in valid_cols]
            df = df[cols_to_keep]
            
            if not df.empty:
                # Application Mapping Noms Equipes
                df['home_team'] = df['home_team'].map(TEAM_MAPPING).fillna(df['home_team'])
                df['away_team'] = df['away_team'].map(TEAM_MAPPING).fillna(df['away_team'])
                all_data.append(df)
        except: pass
        
    if all_data: return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

# --- SOURCE 2 : CALENDRIER FUTUR (CSV LOCAL) ---
def get_csv_calendar(file_path="ligue1_calendrier.csv"):
    print(f"📂 Lecture du fichier local : {file_path}...")
    try:
        df = pd.read_csv(file_path)
        
        # Mapping CSV spécifique
        df = df.rename(columns={
            'Round Number': 'journee',
            'Date': 'full_date',
            'Home Team': 'home_team',
            'Away Team': 'away_team'
        })
        
        # Conversion Date
        df['datetime_match'] = pd.to_datetime(df['full_date'], format='%d/%m/%Y %H:%M', errors='coerce')
        df['date'] = df['datetime_match'].dt.date
        df['time'] = df['datetime_match'].dt.strftime('%H:%M')
        
        # Mapping Equipes
        df['home_team'] = df['home_team'].map(TEAM_MAPPING).fillna(df['home_team'])
        df['away_team'] = df['away_team'].map(TEAM_MAPPING).fillna(df['away_team'])
        
        df['saison_label'] = '2025-2026' # Saison du fichier
        
        # Sélection finale
        final_df = df[['saison_label', 'journee', 'datetime_match', 'home_team', 'away_team']]
        final_df = final_df.dropna(subset=['datetime_match'])
        
        print(f"   ✅ {len(final_df)} matchs chargés depuis le CSV.")
        return final_df
        
    except Exception as e:
        print(f"   ❌ Erreur lecture CSV : {e}")
        return pd.DataFrame()

def update_standings_table(credentials, project_id):
    from google.cloud import bigquery
    client = bigquery.Client(credentials=credentials, project=project_id)
    try:
        with open("update_classement.sql", "r") as file:
            sql_query = file.read()
        if ENV_TYPE != "prod":
            sql_query = sql_query.replace("historic_datasets", "historic_datasets_dev")
        client.query(sql_query).result()
        print(f"✅ Classement Live mis à jour.")
    except Exception as e:
        print(f"⚠️ Warning SQL Classement : {e}")

def main():
    print("🚀 Démarrage ETL...")
    service_account_info = json.loads(os.environ["GCP_SA_KEY"])
    credentials = service_account.Credentials.from_service_account_info(service_account_info)

    # 1. Export Matchs (Historique jusqu'à 2024-2025 pour training)
    df_history = get_football_data_legacy(1994, 2024)
    if not df_history.empty:
        try:
            to_gbq(df_history, TABLE_MATCHS, project_id=PROJECT_ID, credentials=credentials, if_exists='replace', chunksize=None)
            print(f"✅ {TABLE_MATCHS} exportée.")
        except Exception as e: print(f"❌ Erreur Matchs : {e}")

    # 2. Export Calendrier (Depuis CSV Local)
    df_cal = get_csv_calendar()
    if not df_cal.empty:
        try:
            to_gbq(df_cal, TABLE_CALENDRIER, project_id=PROJECT_ID, credentials=credentials, if_exists='replace', chunksize=None)
            print(f"✅ {TABLE_CALENDRIER} exportée.")
        except Exception as e: print(f"❌ Erreur Calendrier : {e}")

    # 3. Mise à jour SQL
    update_standings_table(credentials, PROJECT_ID)
    print("🎉 Terminé.")

if __name__ == "__main__":
    main()