import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
import plotly.express as px
import numpy as np
from scipy.stats import poisson

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Ligue 1 Data Center", layout="wide", page_icon="⚽")

# --- STYLE CSS AVANCÉ (THEME SOMBRE) ---
st.markdown("""
    <style>
    /* 1. Fond général */
    .stApp { background-color: #1A1C23; }
    
    /* 2. Textes en Blanc (Zone principale) */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6, .main p, .main span, .main div, .main label {
        color: #FFFFFF !important; 
    }
    
    /* Exception : Cartes métriques */
    .metric-card h3, .metric-card div, .metric-card span { color: #DAE025 !important; }
    .metric-card .metric-label { color: #E0E0E0 !important; }

    /* 3. Sidebar (Texte Blanc) */
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    
    /* 4. Cartes KPI */
    .metric-card {
        background-color: #091C3E;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 10px;
        border: 1px solid #DAE025;
    }
    .metric-value { font-size: 2rem; font-weight: 800; margin: 0; color: #DAE025 !important; }
    
    /* 5. Score Predictor Card */
    .score-card {
        background: linear-gradient(135deg, #091C3E 0%, #1A1C23 100%);
        border: 2px solid #DAE025;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }
    .score-display {
        font-size: 3.5rem;
        font-weight: bold;
        color: #FFFFFF;
        font-family: 'Courier New', monospace;
    }
    
    /* Pastilles forme */
    .form-badge {
        display: inline-block; width: 30px; height: 30px; line-height: 30px;
        border-radius: 50%; text-align: center; font-weight: bold;
        color: white !important; margin-right: 5px; font-size: 0.8rem;
    }
    .win { background-color: #2ECC71; }
    .draw { background-color: #95A5A6; }
    .loss { background-color: #E74C3C; }
    </style>
""", unsafe_allow_html=True)

# --- CONNEXION BIGQUERY ---
@st.cache_resource
def get_db_client():
    key_dict = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(key_dict)
    return bigquery.Client(credentials=creds, project=key_dict["project_id"])

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data(ttl=3600)
def get_seasons_list():
    client = get_db_client()
    query = "SELECT DISTINCT season FROM `ligue1-data.historic_datasets.matchs_clean` ORDER BY season DESC"
    return client.query(query).to_dataframe()['season'].tolist()

@st.cache_data(ttl=600)
def load_focus_season(season_name):
    client = get_db_client()
    # Classement
    q_class = f"SELECT * FROM `ligue1-data.historic_datasets.classement_live` WHERE saison = '{season_name}' ORDER BY journee_team ASC"
    # Matchs
    q_matchs = f"SELECT * FROM `ligue1-data.historic_datasets.matchs_clean` WHERE season = '{season_name}' ORDER BY date ASC"
    
    df_class = client.query(q_class).to_dataframe()
    df_matchs = client.query(q_matchs).to_dataframe()
    
    if not df_class.empty:
        df_class['match_timestamp'] = pd.to_datetime(df_class['match_timestamp'], utc=True).dt.tz_localize(None)
    if not df_matchs.empty:
        df_matchs['date'] = pd.to_datetime(df_matchs['date'], utc=True).dt.tz_localize(None)
        
    return df_class, df_matchs

@st.cache_data(ttl=600)
def load_multi_season_stats(seasons_list):
    client = get_db_client()
    seasons_str = "', '".join(seasons_list)
    query = f"SELECT * FROM `ligue1-data.historic_datasets.matchs_clean` WHERE season IN ('{seasons_str}')"
    df = client.query(query).to_dataframe()
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    return df

# --- LOGIQUE MÉTIER ---

def calculate_h2h_detailed(df, team, opponent):
    """Calcule les stats moyennes détaillées (Tirs, Cartons) entre 2 équipes"""
    # Filtrer les matchs entre Team et Opponent
    mask = ((df['home_team'] == team) & (df['away_team'] == opponent)) | \
           ((df['home_team'] == opponent) & (df['away_team'] == team))
    df_h2h = df[mask]
    
    if df_h2h.empty:
        return None, 0
    
    stats = {
        'goals_for': [], 'goals_against': [],
        'shots_for
