from google.cloud import bigquery
import pandas as pd
import os
import json
import unicodedata
import requests
from io import StringIO
from google.oauth2 import service_account
from pandas_gbq import to_gbq

# --- CONFIGURATION ---
PROJECT_ID = os.environ["GCP_PROJECT_ID"]
DATASET_ID = "historic_datasets"
TABLE_MATCHS = f"{DATASET_ID}.matchs_clean"
TABLE_CALENDRIER = f"{DATASET_ID}.referentiel_calendrier"

# MAPPING NOMS EQUIPES
TEAM_MAPPING = {
    "Paris Saint Germain": "Paris SG", "Paris S.G.": "Paris SG", "PSG": "Paris SG",
    "Marseille": "Marseille", "Olympique Marseille": "Marseille",
    "Lyon": "Lyon", "Olympique Lyonnais": "Lyon", "OL": "Lyon",
    "Monaco": "Monaco", "AS Monaco": "Monaco",
    "Lille": "Lille", "Lille OSC": "Lille", "LOSC": "Lille",
    "Lens": "Lens", "RC Lens": "Lens",
    "Rennes": "Rennes", "Stade Rennais": "Rennes",
    "Nice": "Nice", "OGC Nice": "Nice",
    "Saint-Étienne": "Saint Etienne", "AS Saint-Étienne": "Saint Etienne", "St Etienne": "Saint Etienne",
    "Strasbourg": "Strasbourg", "RC Strasbourg": "Strasbourg",
    "Nantes": "Nantes", "FC Nantes": "Nantes",
    "Reims": "Reims", "Stade de Reims": "Reims",
    "Montpellier": "Montpellier", "Montpellier HSC": "Montpellier",
    "Toulouse": "Toulouse", "Toulouse FC": "Toulouse",
    "Brest": "Brest", "Stade Brestois": "Brest",
    "Lorient": "Lorient", "FC Lorient": "Lorient",
    "Le Havre": "Le Havre", "Le Havre AC": "Le Havre",
    "Metz": "Metz", "FC Metz": "Metz",
    "Auxerre": "Auxerre", "AJ Auxerre": "Auxerre",
    "Angers": "Angers", "Angers SCO": "Angers"
}

# MAPPING COLONNES CSV
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
    'MaxCH': 'max_closing_home_win_odds', 'MaxCD': 'max_closing_draw_odds', 'MaxCA': 'max_closing_away_win_odds',
    'AvgCH': 'avg_closing_home_win_odds', 'AvgCD': 'avg_closing_draw_odds', 'AvgCA': 'avg_closing_away_win_odds',
    'B365>2.5': 'bet365_over_25_goals', 'B365<2.5': 'bet365_under_25_goals',
    'P>2.5': 'pinnacle_over_25_goals', 'P<2.5': 'pinnacle_under_25_goals',
    'Max>2.5': 'max_over_25_goals', 'Max<2.5': 'max_under_25_goals',
    'Avg>2.5': 'avg_over_25_goals', 'Avg<2.5': 'avg_under_25_goals',
}

# --- SOURCE 1 : FOOTBALL-DATA (CSV) ---
def get_football_data_legacy(start_year, end_year):
    print("   Downloading Football-Data.co.uk...")
    all_data = []
    valid_cols = list(set(COLUMN_MAPPING.values())) + ['season', 'division']
    
    for year in range(start_year, end_year + 1):
        code = f"{str(year)[-2:]}{str(year+1)[-2:]}"
        season_label = f"{year}-{year+1}"
        url = f"https://www.football-data.co.uk/mmz4281/{code}/F1.csv"
        try:
            df = pd.read_csv(url, encoding='latin-1', on_bad_lines='skip')
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.assign(season=season_label, division='Ligue 1')
            df = df.rename(columns=COLUMN_MAPPING)
            df = df.loc[:, ~df.columns.duplicated()]
            
            # Harmonisation Date
            df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce').dt.date.astype(str)
            
            # Filtrage colonnes
            cols_to_keep = [c for c in df.columns if c in valid_cols]
            df = df[cols_to_keep]
            
            if not df.empty:
                all_data.append(df)
        except: pass
        
    if all_data: return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

# --- SOURCE 2 : CALENDRIER FUTUR ---
def get_future_schedule():
    print("   Downloading FixtureDownload (Future)...")
    try:
        url = "https://fixturedownload.com/feed/json/ligue-1-2024"
        data = requests.get(url).json()
        df = pd.DataFrame(data)
        df = df.rename(columns={'DateUtc': 'full_date', 'HomeTeam': 'home_team', 'AwayTeam': 'away_team'})
        df['date'] = pd.to_datetime(df['full_date']).dt.date
        df['time'] = pd.to_datetime(df['full_date']).dt.strftime('%H:%M')
        # On ne garde que le futur
        df = df[pd.to_datetime(df['full_date']) >= pd.Timestamp.now()]
        return df[['date', 'time', 'home_team', 'away_team']]
    except: return pd.DataFrame()

def build_calendrier_referentiel(df_history):
    """Consolide historique + futur pour le calendrier"""
    print("🛠️ Construction du Référentiel Calendrier...")
    
    df_future = get_future_schedule()
    
    # On prépare l'historique pour le format calendrier
    df_hist_cal = df_history[['date', 'time', 'home_team', 'away_team', 'season']].copy() if not df_history.empty else pd.DataFrame()
    
    # On concatène
    master_df = pd.concat([df_hist_cal, df_future], ignore_index=True)
    
    if master_df.empty: return pd.DataFrame()

    # Nettoyage
    master_df['date'] = pd.to_datetime(master_df['date'], errors='coerce')
    master_df = master_df.dropna(subset=['date'])
    master_df['time'] = master_df['time'].fillna('20:00').astype(str)
    
    # Création Datetime précis
    master_df['datetime_match'] = pd.to_datetime(
        master_df['date'].astype(str) + ' ' + master_df['time'].str.strip(), 
        errors='coerce'
    )
    
    # Standardisation Noms
    master_df['home_team'] = master_df['home_team'].apply(lambda x: TEAM_MAPPING.get(x, x))
    master_df['away_team'] = master_df['away_team'].apply(lambda x: TEAM_MAPPING.get(x, x))
    
    # Calcul Saison & Journée
    master_df['season_year'] = master_df['date'].dt.year
    master_df.loc[master_df['date'].dt.month < 7, 'season_year'] -= 1
    master_df['saison_label'] = master_df['season_year'].astype(str) + "-" + (master_df['season_year'] + 1).astype(str)
    
    master_df = master_df.sort_values('datetime_match')
    master_df['journee'] = master_df.groupby('saison_label')['datetime_match'].rank(method='dense').astype(int) // 9 + 1
    
    return master_df[['saison_label', 'journee', 'datetime_match', 'home_team', 'away_team']].drop_duplicates()

def update_standings_table(credentials, project_id):
    from google.cloud import bigquery
    client = bigquery.Client(credentials=credentials, project=project_id)
    try:
        with open("update_classement.sql", "r") as file:
            client.query(file.read()).result()
        print("✅ Classement Live mis à jour.")
    except Exception as e:
        print(f"⚠️ Warning SQL Classement : {e}")

def main():
    print("🚀 Démarrage ETL...")
    service_account_info = json.loads(os.environ["GCP_SA_KEY"])
    credentials = service_account.Credentials.from_service_account_info(service_account_info)

    # 1. Export Matchs (Historique)
    df_history = get_football_data_legacy(1993, 2025)
    if not df_history.empty:
        try:
            to_gbq(df_history, TABLE_MATCHS, project_id=PROJECT_ID, credentials=credentials, if_exists='replace', chunksize=None)
            print(f"✅ {TABLE_MATCHS} exportée.")
        except Exception as e: print(f"❌ Erreur Matchs : {e}")

    # 2. Export Calendrier (Historique + Futur)
    df_cal = build_calendrier_referentiel(df_history)
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