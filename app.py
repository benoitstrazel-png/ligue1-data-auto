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
        'shots_for': [], 'shots_against': [],
        'target_for': [], 'target_against': [],
        'yellow_for': [], 'red_for': []
    }
    
    for _, row in df_h2h.iterrows():
        is_home = row['home_team'] == team
        
        # Gestion des buts
        stats['goals_for'].append(row['full_time_home_goals'] if is_home else row['full_time_away_goals'])
        stats['goals_against'].append(row['full_time_away_goals'] if is_home else row['full_time_home_goals'])
        
        # Gestion des tirs (si dispo)
        if pd.notna(row.get('home_shots')):
            stats['shots_for'].append(row['home_shots'] if is_home else row['away_shots'])
            stats['shots_against'].append(row['away_shots'] if is_home else row['home_shots'])
            stats['target_for'].append(row['home_shots_on_target'] if is_home else row['away_shots_on_target'])
            stats['target_against'].append(row['away_shots_on_target'] if is_home else row['home_shots_on_target'])
        
        # Cartons
        if pd.notna(row.get('home_yellow_cards')):
            stats['yellow_for'].append(row['home_yellow_cards'] if is_home else row['away_yellow_cards'])
            stats['red_for'].append(row['home_red_cards'] if is_home else row['away_red_cards'])

    # Calcul des moyennes
    avg_stats = {k: (np.mean(v) if v else 0) for k, v in stats.items()}
    return avg_stats, len(df_h2h)

def predict_match_score(df_season, team_home, team_away):
    """
    Prédit le score basé sur la Loi de Poisson
    en utilisant les forces d'attaque et de défense de la saison choisie.
    """
    # 1. Moyennes de la ligue
    avg_home_goals = df_season['full_time_home_goals'].mean()
    avg_away_goals = df_season['full_time_away_goals'].mean()
    
    # 2. Stats Domicile (Team Home)
    home_matches = df_season[df_season['home_team'] == team_home]
    if home_matches.empty: return None # Pas de données
    
    attack_home = home_matches['full_time_home_goals'].mean() / avg_home_goals
    defense_home = home_matches['full_time_away_goals'].mean() / avg_away_goals
    
    # 3. Stats Extérieur (Team Away)
    away_matches = df_season[df_season['away_team'] == team_away]
    if away_matches.empty: return None
    
    attack_away = away_matches['full_time_away_goals'].mean() / avg_away_goals
    defense_away = away_matches['full_time_home_goals'].mean() / avg_home_goals
    
    # 4. Calcul des "Expected Goals" (xG)
    xg_home = attack_home * defense_away * avg_home_goals
    xg_away = attack_away * defense_home * avg_away_goals
    
    return xg_home, xg_away

# --- SIDEBAR ---
st.sidebar.title("🔍 Filtres")
all_seasons = get_seasons_list()
selected_seasons = st.sidebar.multiselect("Périmètre d'analyse", all_seasons, default=[all_seasons[0]])

if not selected_seasons:
    st.warning("Veuillez sélectionner au moins une saison.")
    st.stop()

focus_season = sorted(selected_seasons, reverse=True)[0]
st.sidebar.markdown(f"**Saison Focus :** {focus_season}")

# Load Data
df_class_focus, df_matchs_focus = load_focus_season(focus_season)
df_history_multi = load_multi_season_stats(selected_seasons)

teams = sorted(df_class_focus['equipe'].unique())
selected_team = st.sidebar.selectbox("Mon Équipe", teams)

max_j = int(df_class_focus['journee_team'].max()) if not df_class_focus.empty else 1
selected_journee = st.sidebar.slider(f"Simuler à la journée :", 1, max_j, max_j)

# Snapshot Classement
df_snap = df_class_focus[df_class_focus['journee_team'] <= selected_journee] \
    .sort_values('match_timestamp').groupby('equipe').last().reset_index()
df_snap = df_snap.sort_values(['total_points', 'total_diff'], ascending=False)
df_snap['rang'] = range(1, len(df_snap) + 1)
team_stats = df_snap[df_snap['equipe'] == selected_team].iloc[0]

# --- DASHBOARD ---
st.title(f"📊 {selected_team} - {focus_season}")

# KPIS GÉNÉRAUX
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="metric-card"><div class="metric-label">Classement</div><div class="metric-value">{int(team_stats["rang"])}e</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="metric-card"><div class="metric-label">Points</div><div class="metric-value">{int(team_stats["total_points"])}</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="metric-card"><div class="metric-label">Buts Pour</div><div class="metric-value">{int(team_stats["total_bp"])}</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="metric-card"><div class="metric-label">Diff.</div><div class="metric-value">{int(team_stats["total_diff"]):+d}</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ==============================================================================
# SECTION DUEL & PRÉDICTION (NOUVEAU)
# ==============================================================================
st.subheader("⚔️ Duel & Prédiction")

# 1. Sélecteur Adversaire
opponents = [t for t in teams if t != selected_team]
col_sel_adv, col_context = st.columns([1, 2])

with col_sel_adv:
    opponent = st.selectbox("Choisir un adversaire pour l'analyse :", opponents)
    match_location = st.radio("Lieu du match :", [f"Domicile ({selected_team})", f"Extérieur ({selected_team})"])
    is_home_game = "Domicile" in match_location

# Calculs
h2h_avg, nb_games = calculate_h2h_detailed(df_history_multi, selected_team, opponent)

# Prédiction (Basée sur la saison FOCUS uniquement pour être pertinente sur la forme actuelle)
p_home = selected_team if is_home_game else opponent
p_away = opponent if is_home_game else selected_team
pred_xg_home, pred_xg_away = predict_match_score(df_matchs_focus, p_home, p_away)

with col_context:
    if pred_xg_home is not None:
        # On arrondit pour le score affiché
        s_home = round(pred_xg_home)
        s_away = round(pred_xg_away)
        
        # Qui est qui pour l'affichage ?
        score_txt = f"{s_home} - {s_away}"
        
        st.markdown(f"""
            <div class="score-card">
                <div style="color: #DAE025; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px;">Prédiction IA (Poisson)</div>
                <div class="score-display">{p_home} {score_txt} {p_away}</div>
                <div style="color: #AAAAAA; font-size: 0.8rem; margin-top: 10px;">
                    Basé sur la forme offensive/défensive de la saison {focus_season}
                    (xG: {pred_xg_home:.2f} - {pred_xg_away:.2f})
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Pas assez de données cette saison pour prédire.")

# STATS DÉTAILLÉES H2H
if h2h_avg:
    st.markdown(f"#### 📊 Historique vs {opponent} (Moyennes sur {nb_games} matchs)")
    
    k1, k2, k3, k4, k5 = st.columns(5)
    
    def h2h_card(col, label, val_for, val_against, unit=""):
        col.markdown(f"""
            <div class="metric-card" style="padding: 10px;">
                <div class="metric-label">{label}</div>
                <div style="font-size: 1.2rem; font-weight: bold; color: white;">
                    <span style="color: #2ECC71;">{val_for:.1f}</span> 
                    <span style="color: #888;">/</span> 
                    <span style="color: #E74C3C;">{val_against:.1f}</span>
                </div>
                <div style="font-size: 0.7rem; color: #888;">Pour / Contre</div>
            </div>
        """, unsafe_allow_html=True)
    
    h2h_card(k1, "Buts", h2h_avg['goals_for'], h2h_avg['goals_against'])
    h2h_card(k2, "Tirs Tentés", h2h_avg['shots_for'], h2h_avg['shots_against'])
    h2h_card(k3, "Tirs Cadrés", h2h_avg['target_for'], h2h_avg['target_against'])
    h2h_card(k4, "Cartons Jaunes", h2h_avg['yellow_for'], 0) # On n'affiche que subis ici souvent
    h2h_card(k5, "Cartons Rouges", h2h_avg['red_for'], 0)

else:
    st.info(f"Aucun historique disponible entre {selected_team} et {opponent} sur les saisons sélectionnées.")

st.markdown("---")

# --- GRAPHIQUE TRAJECTOIRE ---
st.subheader("📈 Trajectoire Saison")
history_team = df_class_focus[(df_class_focus['equipe'] == selected_team) & (df_class_focus['journee_team'] <= selected_journee)]
fig = px.line(history_team, x='journee_team', y='total_points', markers=True, labels={'journee_team': 'Journée', 'total_points': 'Points'})
fig.update_traces(line_color='#DAE025', line_width=4)
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
st.plotly_chart(fig, use_container_width=True)

# --- CLASSEMENT LIVE ---
st.subheader("🏆 Classement")
st.dataframe(df_snap[['rang', 'equipe', 'total_points', 'total_diff', 'total_V', 'total_N', 'total_D']].set_index('rang'), use_container_width=True)
