import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import poisson

# --- CONFIGURATION ---
st.set_page_config(page_title="Ligue 1 Data Center", layout="wide", page_icon="⚽")
st.markdown("""
    <style>
    .stApp { background-color: #1A1C23; }
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6, .main p, .main span, .main div, .main label { color: #FFFFFF !important; }
    .metric-card { background-color: #091C3E; padding: 15px; border-radius: 12px; text-align: center; border: 1px solid #DAE025; margin-bottom: 10px; }
    .metric-value { font-size: 1.8rem; font-weight: 800; margin: 0; color: #DAE025 !important; }
    .metric-label { color: #E0E0E0 !important; font-size: 0.9rem; }
    .ref-card { background-color: #2C3E50; padding: 15px; border-radius: 10px; border-left: 5px solid #E74C3C; margin-top: 10px; }
    .score-card { background: linear-gradient(135deg, #091C3E 0%, #1A1C23 100%); border: 2px solid #DAE025; border-radius: 15px; padding: 20px; text-align: center; margin-top: 20px; }
    .score-display { font-size: 3.5rem; font-weight: bold; color: #FFFFFF; font-family: monospace; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    </style>
""", unsafe_allow_html=True)

# --- BQ LOAD ---
@st.cache_resource
def get_db_client():
    key_dict = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(key_dict)
    return bigquery.Client(credentials=creds, project=key_dict["project_id"])

@st.cache_data(ttl=3600)
def get_seasons_list():
    client = get_db_client()
    query = f"SELECT DISTINCT season FROM `{client.project}.historic_datasets.matchs_clean` ORDER BY season DESC"
    return client.query(query).to_dataframe()['season'].tolist()

@st.cache_data(ttl=3600)
def load_referee_referential():
    """Charge le référentiel arbitres (Info disponibilité)"""
    client = get_db_client()
    try:
        q = f"SELECT * FROM `{client.project}.historic_datasets.referentiel_arbitres`"
        return client.query(q).to_dataframe()
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_calendar_referential(season_label):
    """Charge le calendrier officiel avec arbitres pour la saison"""
    client = get_db_client()
    try:
        q = f"""
            SELECT * FROM `{client.project}.historic_datasets.referentiel_calendrier` 
            WHERE saison_label = '{season_label}'
            ORDER BY datetime_match
        """
        df = client.query(q).to_dataframe()
        df['datetime_match'] = pd.to_datetime(df['datetime_match']).dt.tz_localize(None)
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=600)
def load_stats_data(season_name):
    client = get_db_client()
    project_id = client.project
    # On utilise toujours les matchs_clean pour les stats de jeu (tirs, buts...)
    q_matchs = f"SELECT * FROM `{project_id}.historic_datasets.matchs_clean` WHERE season = '{season_name}' ORDER BY date ASC"
    q_class = f"SELECT * FROM `{project_id}.historic_datasets.classement_live` WHERE saison = '{season_name}' ORDER BY journee_team ASC"
    
    df_m = client.query(q_matchs).to_dataframe()
    df_c = client.query(q_class).to_dataframe()
    
    if not df_m.empty: df_m['date'] = pd.to_datetime(df_m['date']).dt.tz_localize(None)
    if not df_c.empty: df_c['match_timestamp'] = pd.to_datetime(df_c['match_timestamp']).dt.tz_localize(None)
    
    return df_c, df_m

@st.cache_data(ttl=600)
def load_multi_season_stats(seasons_list):
    client = get_db_client()
    seasons_str = "', '".join(seasons_list)
    query = f"SELECT * FROM `{client.project}.historic_datasets.matchs_clean` WHERE season IN ('{seasons_str}')"
    df = client.query(query).to_dataframe()
    if not df.empty: df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    return df

# --- LOGIQUE ANALYSE ---
def analyze_referee_risk(df_history, referee_name, team1, team2):
    """Calcule le risque basé sur l'historique des matchs nettoyés"""
    if not referee_name or referee_name == "INCONNU": return None
    
    # On filtre sur l'arbitre dans la table historique
    # Note: On suppose que la table matchs_clean a aussi une colonne 'referee' (venant de FData)
    # Si elle est vide, on ne pourra pas calculer grand chose, d'où l'importance du merge dans main.py
    matches = df_history[df_history['referee'].str.upper().str.contains(referee_name, na=False)]
    
    if matches.empty: return None
    
    # Stats globales
    avg_y = (matches['home_yellow_cards'].sum() + matches['away_yellow_cards'].sum()) / len(matches)
    avg_r = (matches['home_red_cards'].sum() + matches['away_red_cards'].sum()) / len(matches)
    
    # Risque
    risk_lvl = "FAIBLE"
    if avg_r > 0.35: risk_lvl = "CRITIQUE 🟥"
    elif avg_r > 0.20: risk_lvl = "ÉLEVÉ 🔥"
    elif avg_r > 0.10: risk_lvl = "MODÉRÉ"
    
    return {
        "matches": len(matches),
        "avg_yellow": avg_y,
        "avg_red": avg_r,
        "risk": risk_lvl
    }

# --- INTERFACE ---
st.sidebar.title("🔍 Filtres")
all_seasons = get_seasons_list()
selected_seasons = st.sidebar.multiselect("Historique", all_seasons, default=[all_seasons[0]])
focus_season = sorted(selected_seasons, reverse=True)[0]

# Chargement
df_class, df_matchs_stats = load_stats_data(focus_season)
df_history_multi = load_multi_season_stats(selected_seasons)
df_ref_info = load_referee_referential()
df_calendar = load_calendar_referential(focus_season)

teams = sorted(df_class['equipe'].unique())
my_team = st.sidebar.selectbox("Mon Équipe", teams)

# DASHBOARD
st.title(f"📊 {my_team} - Analyse & Arbitrage")

# --- SECTION 1 : CALENDRIER & ARBITRES ---
st.subheader("📅 Prochains Matchs & Arbitres")

# On cherche les prochains matchs de l'équipe dans le Référentiel Calendrier
now = pd.Timestamp.now()
future_games = df_calendar[
    ((df_calendar['home_team'] == my_team) | (df_calendar['away_team'] == my_team)) &
    (df_calendar['datetime_match'] > now)
].head(3)

if not future_games.empty:
    cols = st.columns(3)
    for i, (idx, row) in enumerate(future_games.iterrows()):
        with cols[i]:
            adv = row['away_team'] if row['home_team'] == my_team else row['home_team']
            loc = "🏠" if row['home_team'] == my_team else "✈️"
            ref = row['referee'] if row['referee'] and row['referee'] != "INCONNU" else "En attente"
            
            st.markdown(f"""
            <div class="ref-card">
                <div style="font-size:0.8rem">{row['datetime_match'].strftime('%d/%m %H:%M')}</div>
                <div style="font-weight:bold; font-size:1.1rem">{loc} vs {adv}</div>
                <hr style="margin:5px 0; border-color:#555">
                <div style="font-size:0.9rem; color:#DAE025">👮 {ref}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Analyse Rapide Arbitre si connu
            if ref != "En attente":
                ref_stats = analyze_referee_risk(df_history_multi, ref, my_team, adv)
                if ref_stats:
                    st.caption(f"Sévérité: {ref_stats['avg_yellow']:.1f} 🟨/m")
                    if "ÉLEVÉ" in ref_stats['risk'] or "CRITIQUE" in ref_stats['risk']:
                        st.warning(f"Attention : Risque exclusion {ref_stats['risk']}")
else:
    st.info("Aucun match futur trouvé dans le calendrier.")

st.markdown("---")

# --- SECTION 2 : SIMULATEUR AVANCÉ ---
st.subheader("⚔️ Simulation de Match")

c1, c2 = st.columns([1, 2])
with c1:
    opp = st.selectbox("Adversaire", [t for t in teams if t != my_team])
    loc = st.radio("Lieu", ["Domicile", "Extérieur"])
    
    # Sélecteur d'arbitre intelligent
    ref_options = ["Inconnu"]
    if not df_ref_info.empty:
        # On trie par expérience (nb matchs)
        ref_options += df_ref_info.sort_values('total_matchs', ascending=False)['referee'].tolist()
    
    ref_selected = st.selectbox("Simuler avec l'arbitre :", ref_options)

with c2:
    # On réutilise la logique de prédiction (simplifiée ici pour l'affichage)
    # (Insérez ici votre logique predict_match_score du code précédent)
    st.info("Sélectionnez les paramètres à gauche pour simuler le contexte du match.")
    
    if ref_selected != "Inconnu":
        st.markdown(f"#### 👮 Focus : {ref_selected}")
        stats = analyze_referee_risk(df_history_multi, ref_selected, my_team, opp)
        
        if stats:
            k1, k2, k3 = st.columns(3)
            k1.metric("Expérience (Dataset)", f"{stats['matches']} matchs")
            k2.metric("Moyenne Cartons", f"{stats['avg_yellow']:.1f} 🟨")
            k3.metric("Risque Rouge", stats['risk'], f"{stats['avg_red']:.2f}/m")
            
            # Info Dispo
            if not df_ref_info.empty:
                info = df_ref_info[df_ref_info['referee'] == ref_selected].iloc[0]
                st.caption(f"Données disponibles du {info['first_seen'].date()} au {info['last_seen'].date()}")
        else:
            st.warning("Pas assez de données historiques pour établir le profil de cet arbitre.")