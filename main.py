from google.cloud import bigquery
import pandas as pd
import os
import json
import unicodedata
import requests
import time
from io import StringIO
from google.oauth2 import service_account
from pandas_gbq import to_gbq

# --- CONFIGURATION ---
PROJECT_ID = os.environ["GCP_PROJECT_ID"]
DATASET_ID = "historic_datasets"

# TABLES CIBLES
TABLE_MATCHS = f"{DATASET_ID}.matchs_clean"          # Stats détaillées (Existante)
TABLE_CALENDRIER = f"{DATASET_ID}.referentiel_calendrier" # Nouvelle table demandée
TABLE_REF_INFO = f"{DATASET_ID}.referentiel_arbitres"     # Nouvelle table demandée

# MAPPING NOMS EQUIPES (Pour réconcilier les sources)
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

def normalize_text(text):
    """Nettoyage standard des noms"""
    if not isinstance(text, str): return None
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text.strip()

def normalize_referee(name):
    """Formatage propre des arbitres (ex: 'Mr. Turpin' -> 'TURPIN')"""
    if not isinstance(name, str): return "INCONNU"
    name = normalize_text(name).upper()
    
    # Nettoyage des préfixes courants
    prefixes = ["M. ", "MR. ", "MONSIEUR ", "ARBITRE: "]
    for p in prefixes:
        if name.startswith(p): name = name.replace(p, "")
        
    # Gestion Initiales (ex: "C. TURPIN" -> "TURPIN")
    if ". " in name:
        parts = name.split(". ")
        if len(parts) > 1: name = parts[-1]
        
    return name.strip()

# --- SOURCE 1 : FBREF (Scraping) ---
def scrape_fbref_schedule(season_year):
    print(f"   Trying FBref for season {season_year}-{season_year+1}...")
    url = f"https://fbref.com/en/comps/13/{season_year}-{season_year+1}/schedule/{season_year}-{season_year+1}-Ligue-1-Scores-and-Fixtures"
    if season_year == 2024: # URL spéciale pour la saison courante parfois
        url = "https://fbref.com/en/comps/13/schedule/Ligue-1-Scores-and-Fixtures"
        
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            tables = pd.read_html(StringIO(response.text))
            df = tables[0]
            # Colonnes attendues : Date, Home, Away, Referee
            if 'Referee' in df.columns:
                df = df[['Date', 'Time', 'Home', 'Away', 'Referee']].copy()
                df = df.rename(columns={'Referee': 'referee', 'Home': 'home_team', 'Away': 'away_team', 'Date': 'date', 'Time': 'time'})
                df['source'] = 'FBref'
                return df
    except Exception as e:
        print(f"   ⚠️ FBref failed: {e}")
    return pd.DataFrame()

# --- SOURCE 2 : FOOTBALL-DATA (CSV) ---
def get_football_data_legacy(start_year, end_year):
    print("   Trying Football-Data.co.uk...")
    all_data = []
    for year in range(start_year, end_year + 1):
        code = f"{str(year)[-2:]}{str(year+1)[-2:]}"
        url = f"https://www.football-data.co.uk/mmz4281/{code}/F1.csv"
        try:
            df = pd.read_csv(url, encoding='latin-1', on_bad_lines='skip')
            # Check des noms de colonnes possibles pour l'arbitre
            ref_col = next((c for c in ['Referee', 'Ref', 'referee'] if c in df.columns), None)
            
            if ref_col:
                df = df.rename(columns={
                    ref_col: 'referee', 'Date': 'date', 'Time': 'time', 
                    'HomeTeam': 'home_team', 'AwayTeam': 'away_team'
                })
                df = df[['date', 'time', 'home_team', 'away_team', 'referee']]
                df['season'] = f"{year}-{year+1}"
                df['source'] = 'FData'
                
                # Conversion date (souvent dd/mm/yy)
                df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
                all_data.append(df)
        except: pass
        
    if all_data: return pd.concat(all_data)
    return pd.DataFrame()

# --- SOURCE 3 : FIXTURE DOWNLOAD (Pour le futur proche, pas d'arbitre souvent mais on prend quand même) ---
def get_future_schedule():
    print("   Trying FixtureDownload (Future)...")
    try:
        url = "https://fixturedownload.com/feed/json/ligue-1-2024"
        data = requests.get(url).json()
        df = pd.DataFrame(data)
        df = df.rename(columns={'DateUtc': 'full_date', 'HomeTeam': 'home_team', 'AwayTeam': 'away_team'})
        df['date'] = pd.to_datetime(df['full_date']).dt.date
        df['time'] = pd.to_datetime(df['full_date']).dt.strftime('%H:%M')
        df['referee'] = None # Souvent vide ici
        df['source'] = 'FixtureDL'
        return df[['date', 'time', 'home_team', 'away_team', 'referee', 'source']]
    except: return pd.DataFrame()

def build_calendrier_referentiel():
    """Consolide toutes les sources pour créer le référentiel Calendrier"""
    print("🛠️ Construction du Référentiel Calendrier & Arbitres...")
    
    # 1. On récupère tout ce qu'on peut
    df_fdata = get_football_data_legacy(2018, 2025) # Historique récent solide
    df_fbref = scrape_fbref_schedule(2024)          # Saison en cours (souvent meilleur pour arbitres)
    df_future = get_future_schedule()               # Calendrier futur
    
    # 2. Fusion intelligente
    # On priorise FBref pour la saison actuelle, FData pour l'historique
    master_df = pd.concat([df_fbref, df_fdata, df_future], ignore_index=True)
    
    if master_df.empty:
        print("❌ Aucune donnée trouvée.")
        return pd.DataFrame(), pd.DataFrame()

    # 3. Nettoyage
    # Date
    master_df['date'] = pd.to_datetime(master_df['date'], errors='coerce')
    master_df = master_df.dropna(subset=['date'])
    
    # Time (Si manquant -> 20:00)
    master_df['time'] = master_df['time'].fillna('20:00').astype(str)
    
    # Création Datetime précis
    # Astuce : On combine date string et time string
    master_df['datetime_match'] = pd.to_datetime(
        master_df['date'].astype(str) + ' ' + master_df['time'].str.strip(), 
        errors='coerce'
    )
    
    # Noms Equipes (Standardisation)
    master_df['home_team'] = master_df['home_team'].apply(lambda x: TEAM_MAPPING.get(x, x))
    master_df['away_team'] = master_df['away_team'].apply(lambda x: TEAM_MAPPING.get(x, x))
    
    # Arbitres (Normalisation)
    master_df['referee_raw'] = master_df['referee']
    master_df['referee'] = master_df['referee'].apply(normalize_referee)
    
    # Calcul de la "Journée" (Approximation par ordre chronologique)
    # On détermine la saison en fonction du mois (Août -> Début saison)
    master_df['season_year'] = master_df['date'].dt.year
    master_df.loc[master_df['date'].dt.month < 7, 'season_year'] -= 1
    master_df['saison_label'] = master_df['season_year'].astype(str) + "-" + (master_df['season_year'] + 1).astype(str)
    
    # Rank dans la saison = Journée
    master_df = master_df.sort_values('datetime_match')
    master_df['journee'] = master_df.groupby('saison_label')['datetime_match'].rank(method='dense').astype(int) // 9 + 1
    # Note: Division par 9 (matchs par journée) + 1 pour estimer grossièrement la J1, J2...
    
    # --- TABLE 1 : CALENDRIER ---
    cols_cal = ['saison_label', 'journee', 'datetime_match', 'home_team', 'away_team', 'referee']
    df_calendrier = master_df[cols_cal].drop_duplicates(subset=['datetime_match', 'home_team'])
    
    # --- TABLE 2 : REFERENTIEL ARBITRES ---
    # On ne garde que les lignes où on a un VRAI arbitre
    df_with_ref = master_df[master_df['referee'] != "INCONNU"]
    
    if not df_with_ref.empty:
        df_arbitres = df_with_ref.groupby('referee').agg(
            first_seen=('datetime_match', 'min'),
            last_seen=('datetime_match', 'max'),
            total_matchs=('datetime_match', 'count')
        ).reset_index()
    else:
        df_arbitres = pd.DataFrame(columns=['referee', 'first_seen', 'last_seen', 'total_matchs'])
        
    return df_calendrier, df_arbitres

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
    print("🚀 Démarrage ETL Multi-Sources...")
    
    # 1. Authentification
    service_account_info = json.loads(os.environ["GCP_SA_KEY"])
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    
    # 2. Construction des Référentiels (Nouvelle Logique)
    df_cal, df_refs = build_calendrier_referentiel()
    
    if not df_cal.empty:
        print(f"📦 Export Calendrier ({len(df_cal)} matchs)...")
        to_gbq(df_cal, TABLE_CALENDRIER, project_id=PROJECT_ID, credentials=credentials, if_exists='replace', chunksize=None)
        
    if not df_refs.empty:
        print(f"📦 Export Référentiel Arbitres ({len(df_refs)} arbitres)...")
        to_gbq(df_refs, TABLE_REF_INFO, project_id=PROJECT_ID, credentials=credentials, if_exists='replace', chunksize=None)

    # 3. Export Table Matchs Clean (Méthode classique pour stats jeu)
    # On garde la fonction originale simplifiée pour ne pas casser l'existant
    print("📦 Export Matchs Clean (Stats)...")
    # (Ici on réutilise la logique get_ligue1_data du fichier précédent ou on l'appelle si besoin)
    # Pour simplifier ce bloc, je suppose que le code précédent get_ligue1_data est inclus ou fusionné
    # Je relance get_ligue1_data ici pour être sûr
    df_stats = get_football_data_legacy(1993, 2025)
    if not df_stats.empty:
        to_gbq(df_stats, TABLE_MATCHS, project_id=PROJECT_ID, credentials=credentials, if_exists='replace', chunksize=None)

    # 4. Mise à jour SQL
    update_standings_table(credentials, PROJECT_ID)
    print("🎉 Terminé avec succès.")

if __name__ == "__main__":
    main()