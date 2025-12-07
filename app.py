import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
import plotly.express as px
import numpy as np
from scipy.stats import poisson
import requests

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Ligue 1 Data Center", layout="wide", page_icon="⚽")

# --- STYLE CSS AVANCÉ (THEME SOMBRE) ---
st.markdown("""
    <style>
    .stApp { background-color: #1A1C23; }
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6, .main p, .main span, .main div, .main label {
        color: #FFFFFF !important; 
    }
    .metric-card h3, .metric-card div, .metric-card span { color: #DAE025 !important; }
    .metric-card .metric-label { color: #E0E0E0 !important; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
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
    q_class = f"SELECT * FROM `ligue1-data.historic_datasets.classement_live` WHERE saison = '{season_name}' ORDER BY journee_team ASC"
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

@st.cache_data(ttl=3600)
def get_live_schedule():
    """Récupère le calendrier officiel JSON gratuit pour le LIVE"""
    try:
        url = "https://fixturedownload.com/feed/json/ligue-1-2024" 
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data)
        df['DateUtc'] = pd.to_datetime(df['DateUtc']).dt.tz_localize(None)
        future = df[df['DateUtc'] > pd.Timestamp.now()].sort_values('DateUtc')
        return future
    except Exception:
        return pd.DataFrame()

# --- LOGIQUE MÉTIER ---

def calculate_h2h_detailed(df, team, opponent):
    mask = ((df['home_team'] == team) & (df['away_team'] == opponent)) | \
           ((df['home_team'] == opponent) & (df['away_team'] == team))
    df_h2h = df[mask]
    if df_h2h.empty: return None, 0
    
    stats = {'goals_for': [], 'goals_against': [], 'shots_for': [], 'shots_against': [], 
             'target_for': [], 'target_against': [], 'yellow_for': [], 'red_for': []}
    
    for _, row in df_h2h.iterrows():
        is_home = row['home_team'] == team
        stats['goals_for'].append(row['full_time_home_goals'] if is_home else row['full_time_away_goals'])
        stats['goals_against'].append(row['full_time_away_goals'] if is_home else row['full_time_home_goals'])
        if pd.notna(row.get('home_shots')):
            stats['shots_for'].append(row['home_shots'] if is_home else row['away_shots'])
            stats['shots_against'].append(row['away_shots'] if is_home else row['home_shots'])
            stats['target_for'].append(row['home_shots_on_target'] if is_home else row['away_shots_on_target'])
            stats['target_against'].append(row['away_shots_on_target'] if is_home else row['home_shots_on_target'])
        if pd.notna(row.get('home_yellow_cards')):
            stats['yellow_for'].append(row['home_yellow_cards'] if is_home else row['away_yellow_cards'])
            stats['red_for'].append(row['home_red_cards'] if is_home else row['away_red_cards'])

    return {k: (np.mean(v) if v else 0) for k, v in stats.items()}, len(df_h2h)

def calculate_streak_probability(df, streak_length=3):
    """Calcule la proba de gagner le match d'après une série"""
    df_calc = df.copy().sort_values('date')
    matches_home = df_calc[['date', 'home_team', 'full_time_result']].rename(columns={'home_team': 'team'})
    matches_home['is_win'] = matches_home['full_time_result'] == 'H'
    matches_away = df_calc[['date', 'away_team', 'full_time_result']].rename(columns={'away_team': 'team'})
    matches_away['is_win'] = matches_away['full_time_result'] == 'A'
    all_matches = pd.concat([matches_home, matches_away]).sort_values(['team', 'date'])
    
    stats = []
    for team, group in all_matches.groupby('team'):
        was_streak = group['is_win'].rolling(streak_length).sum().shift(1) == streak_length
        after_streak_matches = group[was_streak]
        if not after_streak_matches.empty:
            stats.append(after_streak_matches['is_win'])
            
    if not stats: return None
    all_results = pd.concat(stats)
    return all_results.mean() * 100, len(all_results)

def predict_match_score(df_history, team_home, team_away):
    if df_history.empty: return None, None
    avg_home_goals = df_history['full_time_home_goals'].mean()
    avg_away_goals = df_history['full_time_away_goals'].mean()
    
    home_matches = df_history[df_history['home_team'] == team_home]
    if home_matches.empty: return None, None
    attack_home = home_matches['full_time_home_goals'].mean() / avg_home_goals
    defense_home = home_matches['full_time_away_goals'].mean() / avg_away_goals
    
    away_matches = df_history[df_history['away_team'] == team_away]
    if away_matches.empty: return None, None
    attack_away = away_matches['full_time_away_goals'].mean() / avg_away_goals
    defense_away = away_matches['full_time_home_goals'].mean() / avg_home_goals
    
    xg_home = attack_home * defense_away * avg_home_goals
    xg_away = attack_away * defense_home * avg_away_goals
    return xg_home, xg_away

# --- SIDEBAR & CHARGEMENT ---
st.sidebar.title("🔍 Filtres")
all_seasons = get_seasons_list()
selected_seasons = st.sidebar.multiselect("Périmètre d'analyse", all_seasons, default=[all_seasons[0]])
if not selected_seasons: st.stop()
focus_season = sorted(selected_seasons, reverse=True)[0]
st.sidebar.markdown(f"**Saison Focus :** {focus_season}")

# CHARGEMENT DES DONNÉES (C'est ici que df_matchs_focus est défini !)
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

# --- DASHBOARD HEADER ---
st.title(f"📊 {selected_team} - {focus_season}")
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

# 1. Vérification Série de Victoires (L'erreur venait d'ici, maintenant c'est au bon endroit)
last_3_matches = df_matchs_focus[
    ((df_matchs_focus['home_team'] == selected_team) | (df_matchs_focus['away_team'] == selected_team))
    & (df_matchs_focus['date'] < pd.Timestamp.now())
].sort_values('date', ascending=False).head(3)

current_streak = 0
for _, row in last_3_matches.iterrows():
    is_home = row['home_team'] == selected_team
    if (is_home and row['full_time_result'] == 'H') or (not is_home and row['full_time_result'] == 'A'):
        current_streak += 1
    else: break

if current_streak >= 3:
    win_rate, count = calculate_streak_probability(df_history_multi, streak_length=3)
    if win_rate is not None:
        st.info(f"🔥 **En feu !** {selected_team} reste sur {current_streak} victoires. Historiquement, après 3 victoires, la 4ème est gagnée dans **{win_rate:.1f}%** des cas.")

# 2. Sélecteurs et Calculs
opponents = [t for t in teams if t != selected_team]
col_sel_adv, col_context = st.columns([1, 2])
with col_sel_adv:
    opponent = st.selectbox("Choisir un adversaire :", opponents)
    match_location = st.radio("Lieu :", [f"Domicile ({selected_team})", f"Extérieur ({selected_team})"])
    is_home_game = "Domicile" in match_location

h2h_avg, nb_games = calculate_h2h_detailed(df_history_multi, selected_team, opponent)
p_home = selected_team if is_home_game else opponent
p_away = opponent if is_home_game else selected_team
pred_xg_home, pred_xg_away = predict_match_score(df_history_multi, p_home, p_away)

with col_context:
    if pred_xg_home is not None:
        s_home, s_away = round(pred_xg_home), round(pred_xg_away)
        st.markdown(f"""
            <div class="score-card">
                <div style="color: #DAE025; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px;">Prédiction IA</div>
                <div class="score-display">{p_home} {int(s_home)} - {int(s_away)} {p_away}</div>
                <div style="color: #AAAAAA; font-size: 0.8rem; margin-top: 10px;">
                    Basé sur {len(selected_seasons)} saison(s) (xG: {pred_xg_home:.2f} - {pred_xg_away:.2f})
                </div>
            </div>
        """, unsafe_allow_html=True)
    else: st.warning("Données insuffisantes pour prédire.")

if h2h_avg:
    st.markdown(f"#### 📊 Moyennes vs {opponent} ({nb_games} matchs)")
    k1, k2, k3, k4, k5 = st.columns(5)
    def h2h_card(col, lbl, v1, v2):
        col.markdown(f"""<div class="metric-card" style="padding:10px;"><div class="metric-label">{lbl}</div>
        <div style="font-size:1.2rem;font-weight:bold;color:white;"><span style="color:#2ECC71;">{v1:.1f}</span> <span style="color:#888;">/</span> <span style="color:#E74C3C;">{v2:.1f}</span></div></div>""", unsafe_allow_html=True)
    h2h_card(k1, "Buts", h2h_avg['goals_for'], h2h_avg['goals_against'])
    h2h_card(k2, "Tirs", h2h_avg['shots_for'], h2h_avg['shots_against'])
    h2h_card(k3, "Cadrés", h2h_avg['target_for'], h2h_avg['target_against'])
    h2h_card(k4, "Jaunes", h2h_avg['yellow_for'], 0)
    h2h_card(k5, "Rouges", h2h_avg['red_for'], 0)

st.markdown("---")

# ==============================================================================
# SECTION : PRÉDICTIONS LIVE / FUTUR
# ==============================================================================
NAME_MAPPING = {
    "Paris Saint Germain": "Paris SG", "Olympique de Marseille": "Marseille", "Olympique Lyonnais": "Lyon",
    "AS Monaco": "Monaco", "Lille OSC": "Lille", "Stade Rennais": "Rennes", "OGC Nice": "Nice",
    "RC Lens": "Lens", "Stade de Reims": "Reims", "Strasbourg Alsace": "Strasbourg", "Montpellier HSC": "Montpellier",
    "FC Nantes": "Nantes", "Toulouse FC": "Toulouse", "Stade Brestois 29": "Brest", "FC Lorient": "Lorient",
    "Clermont Foot": "Clermont", "Le Havre AC": "Le Havre", "FC Metz": "Metz", "AJ Auxerre": "Auxerre",
    "Angers SCO": "Angers", "AS Saint-Etienne": "Saint Etienne"
}

st.subheader("🔮 Calendrier & Prédictions")
current_cutoff = pd.to_datetime(team_stats['match_timestamp']).replace(tzinfo=None)
is_replay_mode = current_cutoff < (pd.Timestamp.now() - pd.Timedelta(days=7))

next_round_matches = pd.DataFrame()
source_origin = ""

if is_replay_mode:
    source_origin = "Historique (Simulation)"
    future_matches = df_matchs_focus[df_matchs_focus['date'] > current_cutoff].sort_values('date')
    if not future_matches.empty:
        next_match_date = future_matches.iloc[0]['date']
        next_round_matches = future_matches[future_matches['date'] <= (next_match_date + pd.Timedelta(days=5))]
        next_round_matches = next_round_matches.rename(columns={'date': 'DateUtc', 'home_team': 'HomeTeam', 'away_team': 'AwayTeam'})
else:
    source_origin = "Calendrier Officiel Live"
    df_api = get_live_schedule()
    if not df_api.empty:
        next_round_num = df_api.iloc[0]['RoundNumber']
        next_round_matches = df_api[df_api['RoundNumber'] == next_round_num]
        next_round_matches['HomeTeam'] = next_round_matches['HomeTeam'].apply(lambda x: NAME_MAPPING.get(x, x))
        next_round_matches['AwayTeam'] = next_round_matches['AwayTeam'].apply(lambda x: NAME_MAPPING.get(x, x))

if not next_round_matches.empty:
    st.caption(f"Source : {source_origin}")
    predictions_data = []
    for _, match in next_round_matches.iterrows():
        dom, ext = match['HomeTeam'], match['AwayTeam']
        xg_h, xg_a = predict_match_score(df_history_multi, dom, ext)
        score_display = f"{int(round(xg_h))} - {int(round(xg_a))}" if xg_h is not None else "N/A"
        xg_display = f"xG: {xg_h:.1f}-{xg_a:.1f}" if xg_h is not None else ""
        predictions_data.append({"Date": match['DateUtc'].strftime('%d/%m %H:%M'), "Domicile": dom, "Score Prédit": score_display, "Extérieur": ext, "Détails": xg_display})
    st.dataframe(pd.DataFrame(predictions_data).set_index("Date"), use_container_width=True)
else:
    st.info("Aucun match futur trouvé.")

st.markdown("---")
# GRAPH & CLASSEMENT
col_graph, col_tab = st.columns([2, 1])
with col_graph:
    st.subheader("📈 Trajectoire")
    history_team = df_class_focus[(df_class_focus['equipe'] == selected_team) & (df_class_focus['journee_team'] <= selected_journee)]
    fig = px.line(history_team, x='journee_team', y='total_points', markers=True)
    fig.update_traces(line_color='#DAE025', line_width=4)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, use_container_width=True)
with col_tab:
    st.subheader("🏆 Classement")
    st.dataframe(df_snap[['rang', 'equipe', 'total_points', 'total_diff', 'total_V', 'total_N', 'total_D']].set_index('rang'), use_container_width=True)