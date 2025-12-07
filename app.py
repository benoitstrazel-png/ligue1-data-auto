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

def calculate_streak_probability(df, streak_length=3):
    """
    Calcule la probabilité de gagner le match SUIVANT après une série de 'streak_length' victoires.
    Basé sur l'historique global chargé (toutes les équipes).
    """
    # On travaille sur une copie pour ne pas casser l'original
    df_calc = df.copy()
    
    # On s'assure que c'est trié par date
    df_calc = df_calc.sort_values('date')
    
    # On crée une liste unique de tous les matchs (Home + Away) du point de vue de chaque équipe
    # C'est un peu technique : on dédouble le dataframe pour avoir une ligne par équipe/match
    matches_home = df_calc[['date', 'home_team', 'full_time_result']].rename(columns={'home_team': 'team'})
    matches_home['is_win'] = matches_home['full_time_result'] == 'H'
    
    matches_away = df_calc[['date', 'away_team', 'full_time_result']].rename(columns={'away_team': 'team'})
    matches_away['is_win'] = matches_away['full_time_result'] == 'A'
    
    all_matches = pd.concat([matches_home, matches_away]).sort_values(['team', 'date'])
    
    # On calcule la série actuelle pour chaque ligne
    # Astuce Pandas : On groupe par équipe, et on compare avec un décalage (shift)
    
    stats = []
    
    # Pour chaque équipe...
    for team, group in all_matches.groupby('team'):
        # On repère les séries de victoires (rolling window)
        # Si la somme des 3 derniers matchs = 3, c'est une série
        was_streak = group['is_win'].rolling(streak_length).sum().shift(1) == streak_length
        
        # On regarde les matchs qui ont SUIVI cette série
        after_streak_matches = group[was_streak]
        
        if not after_streak_matches.empty:
            stats.append(after_streak_matches['is_win'])
            
    if not stats:
        return None
        
    # On concatène tous les résultats trouvés
    all_results_after_streak = pd.concat(stats)
    
    # Calcul du % de victoire
    win_rate = all_results_after_streak.mean() * 100
    sample_size = len(all_results_after_streak)
    
    return win_rate, sample_size


# ... (Dans la section DUEL & PRÉDICTION) ...

# 1. On calcule la série actuelle de l'équipe sélectionnée
# On regarde ses 3 derniers matchs
last_3_matches = df_matchs_focus[
    ((df_matchs_focus['home_team'] == selected_team) | (df_matchs_focus['away_team'] == selected_team))
    & (df_matchs_focus['date'] < pd.Timestamp.now()) # On s'assure de ne pas prendre de matchs futurs
].sort_values('date', ascending=False).head(3)

current_streak = 0
for _, row in last_3_matches.iterrows():
    is_home = row['home_team'] == selected_team
    if (is_home and row['full_time_result'] == 'H') or (not is_home and row['full_time_result'] == 'A'):
        current_streak += 1
    else:
        break # La série est brisée

# 2. Si l'équipe est sur une série de 3 victoires (ou plus), on affiche la stat historique
if current_streak >= 3:
    win_rate, count = calculate_streak_probability(df_history_multi, streak_length=3)
    
    if win_rate is not None:
        st.info(f"""
        🔥 **L'équipe est en feu !** {selected_team} reste sur {current_streak} victoires consécutives.
        
        Historiquement (sur les saisons analysées), une équipe qui a gagné 3 matchs de suite 
        **remporte le 4ème match dans {win_rate:.1f}% des cas** (basé sur {count} occurrences similaires).
        """)

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
# SECTION DUEL & PRÉDICTION
# ==============================================================================
st.subheader("⚔️ Duel & Prédiction")

# 1. Sélecteur Adversaire
opponents = [t for t in teams if t != selected_team]
col_sel_adv, col_context = st.columns([1, 2])

with col_sel_adv:
    opponent = st.selectbox("Choisir un adversaire pour l'analyse :", opponents)
    match_location = st.radio("Lieu du match :", [f"Domicile ({selected_team})", f"Extérieur ({selected_team})"])
    is_home_game = "Domicile" in match_location

# Calculs H2H
h2h_avg, nb_games = calculate_h2h_detailed(df_history_multi, selected_team, opponent)

# Prédiction
p_home = selected_team if is_home_game else opponent
p_away = opponent if is_home_game else selected_team
pred_xg_home, pred_xg_away = predict_match_score(df_history_multi, p_home, p_away)

with col_context:
    if pred_xg_home is not None:
        s_home = round(pred_xg_home)
        s_away = round(pred_xg_away)
        score_txt = f"{s_home} - {s_away}"
        
        st.markdown(f"""
            <div class="score-card">
                <div style="color: #DAE025; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px;">Prédiction IA (Poisson)</div>
                <div class="score-display">{p_home} {score_txt} {p_away}</div>
                <div style="color: #AAAAAA; font-size: 0.8rem; margin-top: 10px;">
                    Basé sur la performance moyenne analysée sur {len(selected_seasons)} saison(s)
                    <br>(xG: {pred_xg_home:.2f} - {pred_xg_away:.2f})
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Pas assez de données sur la période sélectionnée pour prédire ce match.")

# Stats H2H
if h2h_avg:
    st.markdown(f"#### 📊 Historique vs {opponent} (Moyennes sur {nb_games} matchs)")
    k1, k2, k3, k4, k5 = st.columns(5)
    
    def h2h_card(col, label, val_for, val_against):
        col.markdown(f"""
            <div class="metric-card" style="padding: 10px;">
                <div class="metric-label">{label}</div>
                <div style="font-size: 1.2rem; font-weight: bold; color: white;">
                    <span style="color: #2ECC71;">{val_for:.1f}</span> <span style="color: #888;">/</span> <span style="color: #E74C3C;">{val_against:.1f}</span>
                </div>
                <div style="font-size: 0.7rem; color: #888;">Pour / Contre</div>
            </div>
        """, unsafe_allow_html=True)
    
    h2h_card(k1, "Buts", h2h_avg['goals_for'], h2h_avg['goals_against'])
    h2h_card(k2, "Tirs Tentés", h2h_avg['shots_for'], h2h_avg['shots_against'])
    h2h_card(k3, "Tirs Cadrés", h2h_avg['target_for'], h2h_avg['target_against'])
    h2h_card(k4, "Cartons Jaunes", h2h_avg['yellow_for'], 0)
    h2h_card(k5, "Rouges", h2h_avg['red_for'], 0)
else:
    st.info(f"Aucun historique disponible entre {selected_team} et {opponent}.")

st.markdown("---")

# ==============================================================================
# SECTION : PRÉDICTIONS JOURNÉE SUIVANTE (HYBRIDE LIVE/HISTORIQUE)
# ==============================================================================
import requests

# --- MAPPING DES NOMS D'ÉQUIPES ---
# L'API utilise des noms longs, notre dataset des noms courts. Il faut traduire.
NAME_MAPPING = {
    "Paris Saint Germain": "Paris SG",
    "Olympique de Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "AS Monaco": "Monaco",
    "Lille OSC": "Lille",
    "Stade Rennais": "Rennes",
    "OGC Nice": "Nice",
    "RC Lens": "Lens",
    "Stade de Reims": "Reims",
    "Strasbourg Alsace": "Strasbourg",
    "Montpellier HSC": "Montpellier",
    "FC Nantes": "Nantes",
    "Toulouse FC": "Toulouse",
    "Stade Brestois 29": "Brest",
    "FC Lorient": "Lorient",
    "Clermont Foot": "Clermont",
    "Le Havre AC": "Le Havre",
    "FC Metz": "Metz",
    "AJ Auxerre": "Auxerre",
    "Angers SCO": "Angers",
    "AS Saint-Etienne": "Saint Etienne"
}

@st.cache_data(ttl=3600)
def get_live_schedule():
    """Récupère le calendrier officiel JSON gratuit"""
    try:
        # URL stable pour la Ligue 1 (source open data)
        url = "https://fixturedownload.com/feed/json/ligue-1-2024" 
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data)
        
        # Nettoyage
        df['DateUtc'] = pd.to_datetime(df['DateUtc']).dt.tz_localize(None)
        # On ne garde que les matchs futurs
        future = df[df['DateUtc'] > pd.Timestamp.now()].sort_values('DateUtc')
        return future
    except Exception:
        return pd.DataFrame()

st.subheader("🔮 Prédictions : Prochains Matchs")

# 1. Identifier si on est en mode "Replay" ou "Live"
current_cutoff = pd.to_datetime(team_stats['match_timestamp']).replace(tzinfo=None)
is_replay_mode = current_cutoff < (pd.Timestamp.now() - pd.Timedelta(days=7))

# DATAFRAME DES FUTURS MATCHS
next_round_matches = pd.DataFrame()
source_origin = ""

if is_replay_mode:
    # MODE REPLAY : On regarde dans le fichier historique
    source_origin = "Historique (Simulation)"
    future_matches = df_matchs_focus[df_matchs_focus['date'] > current_cutoff].sort_values('date')
    if not future_matches.empty:
        next_match_date = future_matches.iloc[0]['date']
        end_window = next_match_date + pd.Timedelta(days=5)
        next_round_matches = future_matches[future_matches['date'] <= end_window]
        # Standardisation des colonnes pour correspondre à la logique unique ci-dessous
        next_round_matches = next_round_matches.rename(columns={'date': 'DateUtc', 'home_team': 'HomeTeam', 'away_team': 'AwayTeam'})

else:
    # MODE LIVE : On tape dans l'API
    source_origin = "Calendrier Officiel Live"
    df_api = get_live_schedule()
    
    if not df_api.empty:
        # On prend la prochaine 'RoundNumber' disponible
        next_round_num = df_api.iloc[0]['RoundNumber']
        next_round_matches = df_api[df_api['RoundNumber'] == next_round_num]
        
        # TRADUCTION DES NOMS (CRITIQUE)
        # On applique le mapping, si pas trouvé on garde le nom original
        next_round_matches['HomeTeam'] = next_round_matches['HomeTeam'].apply(lambda x: NAME_MAPPING.get(x, x))
        next_round_matches['AwayTeam'] = next_round_matches['AwayTeam'].apply(lambda x: NAME_MAPPING.get(x, x))

# AFFICHAGE ET CALCULS
if not next_round_matches.empty:
    st.caption(f"Source des matchs : {source_origin}")
    
    predictions_data = []
    for _, match in next_round_matches.iterrows():
        dom = match['HomeTeam']
        ext = match['AwayTeam']
        date_m = match['DateUtc'].strftime('%d/%m %H:%M')
        
        # Prédiction avec nos données historiques BigQuery
        xg_h, xg_a = predict_match_score(df_history_multi, dom, ext)
        
        score_display = "N/A"
        xg_display = "Données insiffisantes"
        
        if xg_h is not None:
            s_h = round(xg_h)
            s_a = round(xg_a)
            score_display = f"{int(s_h)} - {int(s_a)}"
            xg_display = f"xG: {xg_h:.1f} - {xg_a:.1f}"
            
        predictions_data.append({
            "Date": date_m,
            "Domicile": dom,
            "Score Prédit": score_display,
            "Extérieur": ext,
            "Détails": xg_display
        })
    
    # Styliser le tableau
    st.dataframe(pd.DataFrame(predictions_data).set_index("Date"), use_container_width=True)

else:
    st.info("Aucun match futur trouvé (Fin de saison ou trêve).")

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