import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
import plotly.express as px
import plotly.graph_objects as go
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
    
    .insight-box {
        background-color: rgba(218, 224, 37, 0.1);
        border-left: 4px solid #DAE025;
        padding: 15px;
        margin-top: 15px;
        color: #FFFFFF;
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

# --- CHARGEMENT DES DONNÉES (CACHE) ---

@st.cache_data(ttl=3600)
def get_seasons_list():
    client = get_db_client()
    # On utilise client.project pour avoir le bon ID projet automatiquement
    query = f"SELECT DISTINCT season FROM `{client.project}.historic_datasets.matchs_clean` ORDER BY season DESC"
    return client.query(query).to_dataframe()['season'].tolist()

@st.cache_data(ttl=600)
def load_focus_season(season_name):
    client = get_db_client()
    # Injection dynamique du Project ID
    q_class = f"SELECT * FROM `{client.project}.historic_datasets.classement_live` WHERE saison = '{season_name}' ORDER BY journee_team ASC"
    q_matchs = f"SELECT * FROM `{client.project}.historic_datasets.matchs_clean` WHERE season = '{season_name}' ORDER BY date ASC"
    
    df_class = client.query(q_class).to_dataframe()
    df_matchs = client.query(q_matchs).to_dataframe()
    
    if not df_class.empty:
        df_class['match_timestamp'] = pd.to_datetime(df_class['match_timestamp'], utc=True).dt.tz_localize(None)
    if not df_matchs.empty:
        df_matchs['date'] = pd.to_datetime(df_matchs['date'], utc=True).dt.tz_localize(None)
        
    return df_class, df_matchs

@st.cache_data(ttl=3600)
def load_rank_vs_rank_history():
    """
    Charge l'historique complet pour l'analyse '2e vs 17e'.
    Récupère TOUS les classements et TOUS les matchs pour construire la matrice.
    """
    client = get_db_client()
    project_id = client.project 
    
    # 1. On charge le classement (Project ID dynamique)
    # CORRECTION ICI : On renomme "saison" -> "season" et "equipe" -> "team" 
    # pour que le reste du code Python (les merges) fonctionne sans modif.
    q_all_class = f"""
        SELECT saison as season, journee_team, equipe as team, total_points 
        FROM `{project_id}.historic_datasets.classement_live`
    """
    
    try:
        df_ranks = client.query(q_all_class).to_dataframe()
    except Exception as e:
        # Si la table n'existe pas ou erreur, on retourne vide pour ne pas crasher
        return pd.DataFrame()
    
    if df_ranks.empty: return pd.DataFrame()

    # 2. On calcule le rang pour chaque (Saison, Journée)
    df_ranks['rank'] = df_ranks.groupby(['season', 'journee_team'])['total_points'].rank(ascending=False, method='min')
    
    # 3. On charge les résultats de matchs
    q_all_matchs = f"""
        SELECT season, home_team, away_team, full_time_result, full_time_home_goals, full_time_away_goals, date
        FROM `{project_id}.historic_datasets.matchs_clean`
        ORDER BY date
    """
    df_matchs = client.query(q_all_matchs).to_dataframe()
    
    # 4. Traitement Python
    df_matchs['home_journee'] = df_matchs.groupby(['season', 'home_team']).cumcount() + 1
    df_matchs['away_journee'] = df_matchs.groupby(['season', 'away_team']).cumcount() + 1
    
    df_matchs['home_prev_j'] = df_matchs['home_journee'] - 1
    df_matchs['away_prev_j'] = df_matchs['away_journee'] - 1
    
    # Merge Home Rank
    # Maintenant ça marche car df_ranks a bien une colonne 'season' (grâce à l'alias SQL)
    df_merged = pd.merge(
        df_matchs, df_ranks, 
        left_on=['season', 'home_team', 'home_prev_j'], 
        right_on=['season', 'team', 'journee_team'], 
        how='inner'
    ).rename(columns={'rank': 'home_rank'}).drop(columns=['team', 'journee_team', 'total_points'])
    
    # Merge Away Rank
    df_merged = pd.merge(
        df_merged, df_ranks, 
        left_on=['season', 'away_team', 'away_prev_j'], 
        right_on=['season', 'team', 'journee_team'], 
        how='inner'
    ).rename(columns={'rank': 'away_rank'}).drop(columns=['team', 'journee_team', 'total_points'])
    
    return df_merged

@st.cache_data(ttl=600)
def load_multi_season_stats(seasons_list):
    client = get_db_client()
    seasons_str = "', '".join(seasons_list)
    # Injection dynamique ici aussi
    query = f"SELECT * FROM `{client.project}.historic_datasets.matchs_clean` WHERE season IN ('{seasons_str}')"
    df = client.query(query).to_dataframe()
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    return df

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

# --- LOGIQUE MÉTIER AVANCÉE ---

def get_rank_vs_rank_stats(df_history_ranks, r_home, r_away, tolerance=3):
    """
    Analyse l'historique : Qui gagne quand le Xème reçoit le Yème ?
    Utilise une tolérance (ex: +/- 3 places) pour avoir assez de données.
    """
    # Filtre : On cherche les matchs où HomeRank est proche de r_home et AwayRank proche de r_away
    mask = (
        (df_history_ranks['home_rank'] >= r_home - tolerance) & 
        (df_history_ranks['home_rank'] <= r_home + tolerance) &
        (df_history_ranks['away_rank'] >= r_away - tolerance) & 
        (df_history_ranks['away_rank'] <= r_away + tolerance)
    )
    
    subset = df_history_ranks[mask]
    n = len(subset)
    
    if n < 10: return None # Pas assez de données fiables
    
    wins = len(subset[subset['full_time_result'] == 'H'])
    draws = len(subset[subset['full_time_result'] == 'D'])
    losses = len(subset[subset['full_time_result'] == 'A'])
    
    return {
        'win_pct': (wins/n)*100,
        'draw_pct': (draws/n)*100,
        'loss_pct': (losses/n)*100,
        'sample': n,
        'label': f"Top {r_home} vs Top {r_away}" # Simplifié
    }

def predict_hybrid_score(df_history, team_home, team_away, rank_stats=None):
    """
    Prédiction Hybride : Loi de Poisson (Force intrinsèque) + Historique Classement (Tendance).
    """
    # 1. Prédiction Poisson Standard (xG)
    if df_history.empty: return None, None, None
    
    avg_home = df_history['full_time_home_goals'].mean()
    avg_away = df_history['full_time_away_goals'].mean()
    
    home_m = df_history[df_history['home_team'] == team_home]
    away_m = df_history[df_history['away_team'] == team_away]
    
    if home_m.empty or away_m.empty: return None, None, None
    
    att_h = home_m['full_time_home_goals'].mean() / avg_home
    def_h = home_m['full_time_away_goals'].mean() / avg_away
    att_a = away_m['full_time_away_goals'].mean() / avg_away
    def_a = away_m['full_time_home_goals'].mean() / avg_home
    
    xg_h = att_h * def_a * avg_home
    xg_a = att_a * def_h * avg_away
    
    # 2. Ajustement par l'historique des Rangs (Si disponible)
    # Si l'historique dit que le favori gagne à 80%, on booste légèrement son xG
    confidence_msg = "Basé sur la forme (Poisson)"
    
    if rank_stats:
        # Probabilités implicites de Poisson (approximatives pour l'exemple)
        # Si rank_stats donne une proba de victoire > 60%, on bonusse
        if rank_stats['win_pct'] > 60:
            xg_h *= 1.15 # Bonus domicile
            confidence_msg = "Ajusté par l'historique du classement (Fav. Domicile)"
        elif rank_stats['loss_pct'] > 60:
            xg_a *= 1.15 # Bonus extérieur
            confidence_msg = "Ajusté par l'historique du classement (Fav. Extérieur)"
            
    return xg_h, xg_a, confidence_msg

# Fonctions utilitaires existantes...
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

def calculate_advanced_kpis(df, team):
    df_team = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
    if df_team.empty: return {}
    nb_matchs = len(df_team)
    clean_sheets = 0
    failed_to_score = 0
    balance = 0
    total_shots_target = 0; total_goals = 0
    for _, row in df_team.iterrows():
        is_home = row['home_team'] == team
        goals_for = row['full_time_home_goals'] if is_home else row['full_time_away_goals']
        goals_against = row['full_time_away_goals'] if is_home else row['full_time_home_goals']
        if goals_against == 0: clean_sheets += 1
        if goals_for == 0: failed_to_score += 1
        if is_home: total_shots_target += row['home_shots_on_target'] if pd.notna(row['home_shots_on_target']) else 0
        else: total_shots_target += row['away_shots_on_target'] if pd.notna(row['away_shots_on_target']) else 0
        total_goals += goals_for
        result = 'H' if is_home and goals_for > goals_against else ('A' if not is_home and goals_for > goals_against else 'D')
        odds = 0
        if is_home and pd.notna(row.get('bet365_home_win_odds')): odds = row['bet365_home_win_odds']
        elif not is_home and pd.notna(row.get('bet365_away_win_odds')): odds = row['bet365_away_win_odds']
        actual_res = row['full_time_result']
        won_bet = (is_home and actual_res == 'H') or (not is_home and actual_res == 'A')
        if won_bet and odds > 0: balance += (10 * odds) - 10
        else: balance -= 10
    roi = (balance / (nb_matchs*10)) * 100 if nb_matchs > 0 else 0
    conversion = (total_goals / total_shots_target * 100) if total_shots_target > 0 else 0
    return {"roi": roi, "clean_sheet_pct": (clean_sheets / nb_matchs) * 100, "no_goal_pct": (failed_to_score / nb_matchs) * 100, "conversion": conversion}

def calculate_spider_data(df, team):
    df_team = df[(df['home_team'] == team) | (df['away_team'] == team)]
    if df_team.empty: return None, None, None
    metrics = {'Buts Pour': [], 'Tirs Cadrés': [], 'Corners': [], 'Jaunes': [], 'Buts Contre': []}
    for _, row in df_team.iterrows():
        is_home = row['home_team'] == team
        metrics['Buts Pour'].append(row['full_time_home_goals'] if is_home else row['full_time_away_goals'])
        metrics['Buts Contre'].append(row['full_time_away_goals'] if is_home else row['full_time_home_goals'])
        metrics['Tirs Cadrés'].append(row['home_shots_on_target'] if is_home else row['away_shots_on_target'])
        metrics['Corners'].append(row['home_corners'] if is_home else row['away_corners'])
        metrics['Jaunes'].append(row['home_yellow_cards'] if is_home else row['away_yellow_cards'])
    team_means = [np.nanmean(v) for v in metrics.values()]
    league_metrics = {k: [] for k in metrics.keys()}
    for _, row in df.iterrows():
        league_metrics['Buts Pour'].append(row['full_time_home_goals'])
        league_metrics['Buts Contre'].append(row['full_time_away_goals'])
        league_metrics['Tirs Cadrés'].append(row['home_shots_on_target'])
        league_metrics['Corners'].append(row['home_corners'])
        league_metrics['Jaunes'].append(row['home_yellow_cards'])
    league_means = [np.nanmean(v) for v in league_metrics.values()]
    return list(metrics.keys()), team_means, league_means

def calculate_referee_impact(df, team):
    df_team = df[(df['home_team'] == team) | (df['away_team'] == team)]
    if df_team.empty or 'referee' not in df_team.columns: return None
    refs = df_team.groupby('referee').agg(Matchs=('date', 'count'), Jaunes=('home_yellow_cards', lambda x: x.sum()), Rouges=('home_red_cards', lambda x: x.sum())).reset_index()
    wins = []
    for ref in refs['referee']:
        sub = df_team[df_team['referee'] == ref]
        win_count = 0
        for _, row in sub.iterrows():
            is_home = row['home_team'] == team
            if (is_home and row['full_time_result'] == 'H') or (not is_home and row['full_time_result'] == 'A'): win_count += 1
        wins.append(round((win_count / len(sub)) * 100, 1))
    refs['% Victoire'] = wins
    return refs.sort_values('Matchs', ascending=False).head(5)

def calculate_streak_probability(df, streak_length=3):
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

# --- SIDEBAR & CHARGEMENT ---
st.sidebar.title("🔍 Filtres")
all_seasons = get_seasons_list()
selected_seasons = st.sidebar.multiselect("Périmètre d'analyse", all_seasons, default=[all_seasons[0]])
if not selected_seasons: st.stop()
focus_season = sorted(selected_seasons, reverse=True)[0]
st.sidebar.markdown(f"**Saison Focus :** {focus_season}")

# LOAD DATA
df_class_focus, df_matchs_focus = load_focus_season(focus_season)
df_history_multi = load_multi_season_stats(selected_seasons)
df_ranks_history = load_rank_vs_rank_history() # New Reference Data

teams = sorted(df_class_focus['equipe'].unique())
selected_team = st.sidebar.selectbox("Mon Équipe", teams)
max_j = int(df_class_focus['journee_team'].max()) if not df_class_focus.empty else 1
selected_journee = st.sidebar.slider(f"Simuler à la journée :", 1, max_j, max_j)

# SNAPSHOT
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

# ================= KPIS AVANCÉS =================
st.subheader("🧪 Analyse Performance & Style")
adv_stats = calculate_advanced_kpis(df_matchs_focus, selected_team)

if adv_stats:
    k1, k2, k3, k4 = st.columns(4)
    roi_val = adv_stats['roi']
    roi_color = "#2ECC71" if roi_val > 0 else "#E74C3C"
    k1.markdown(f"""<div class="metric-card"><div class="metric-label">ROI Betting (Saison)</div><div class="metric-value" style="color:{roi_color} !important;">{roi_val:+.1f}%</div></div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class="metric-card"><div class="metric-label">Clean Sheets</div><div class="metric-value">{adv_stats['clean_sheet_pct']:.0f}%</div></div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class="metric-card"><div class="metric-label">Matchs Sans Marquer</div><div class="metric-value">{adv_stats['no_goal_pct']:.0f}%</div></div>""", unsafe_allow_html=True)
    k4.markdown(f"""<div class="metric-card"><div class="metric-label">Efficacité (But/Cadré)</div><div class="metric-value">{adv_stats['conversion']:.0f}%</div></div>""", unsafe_allow_html=True)

c_radar, c_ref = st.columns([1, 1])
with c_radar:
    st.markdown("##### 🕸️ Profil vs Moyenne Ligue")
    cats, t_vals, l_vals = calculate_spider_data(df_matchs_focus, selected_team)
    if cats:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=t_vals, theta=cats, fill='toself', name=selected_team, line_color='#DAE025'))
        fig.add_trace(go.Scatterpolar(r=l_vals, theta=cats, fill='toself', name='Moyenne', line_color='#95A5A6'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, color='white')), paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), margin=dict(t=20, b=20, l=20, r=20), legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)
with c_ref:
    st.markdown("##### 👮 Top Arbitres")
    df_refs = calculate_referee_impact(df_matchs_focus, selected_team)
    if df_refs is not None: st.dataframe(df_refs.set_index('referee'), use_container_width=True)

st.markdown("---")

# ================= DUEL & PRÉDICTION HYBRIDE =================
st.subheader("⚔️ Duel & Prédiction Intelligente")

# 1. Streak
last_3 = df_matchs_focus[((df_matchs_focus['home_team'] == selected_team) | (df_matchs_focus['away_team'] == selected_team)) & (df_matchs_focus['date'] < pd.Timestamp.now())].sort_values('date', ascending=False).head(3)
streak = 0
for _, r in last_3.iterrows():
    home = r['home_team'] == selected_team
    if (home and r['full_time_result']=='H') or (not home and r['full_time_result']=='A'): streak += 1
    else: break
if streak >= 3:
    rate, n = calculate_streak_probability(df_history_multi)
    if rate: st.info(f"🔥 **En feu !** {streak} victoires de suite. Historiquement, la suivante est gagnée à **{rate:.1f}%**.")

# 2. Select Adversaire
opps = [t for t in teams if t != selected_team]
c_sel, c_ctx = st.columns([1, 2])
with c_sel:
    opp = st.selectbox("Adversaire :", opps)
    loc = st.radio("Lieu :", [f"Domicile ({selected_team})", f"Extérieur ({selected_team})"])
    is_home = "Domicile" in loc

# 3. Calculs Ranks & Stats
# On a besoin du rang actuel de l'adversaire
try:
    opp_stats = df_snap[df_snap['equipe'] == opp].iloc[0]
    rank_team = int(team_stats['rang'])
    rank_opp = int(opp_stats['rang'])
    
    # QUI RECOIT QUI ?
    r_home = rank_team if is_home else rank_opp
    r_away = rank_opp if is_home else rank_team
    
    # STATS HISTORIQUES (RANK vs RANK)
    rank_stats = get_rank_vs_rank_stats(df_ranks_history, r_home, r_away)
except:
    rank_stats = None

# 4. Prédiction
p_home = selected_team if is_home else opp
p_away = opp if is_home else selected_team
xg_h, xg_a, conf_msg = predict_hybrid_score(df_history_multi, p_home, p_away, rank_stats)

with c_ctx:
    if xg_h is not None:
        s_h, s_a = round(xg_h), round(xg_a)
        st.markdown(f"""
            <div class="score-card">
                <div style="color: #DAE025; text-transform: uppercase; letter-spacing: 2px;">Prédiction Hybride</div>
                <div class="score-display">{p_home} {int(s_h)} - {int(s_a)} {p_away}</div>
                <div style="color: #AAAAAA; font-size: 0.8rem;">{conf_msg}<br>(xG Ajustés: {xg_h:.2f} - {xg_a:.2f})</div>
            </div>
        """, unsafe_allow_html=True)
        
        if rank_stats:
            st.markdown(f"""
                <div class="insight-box">
                    📊 <b>Statistique Choc :</b> Historiquement, quand le <b>{r_home}e</b> reçoit le <b>{r_away}e</b> (zone +/- 3 places), 
                    l'équipe à domicile gagne dans <b>{rank_stats['win_pct']:.1f}%</b> des cas 
                    (sur {rank_stats['sample']} matchs analysés depuis 1993).
                </div>
            """, unsafe_allow_html=True)
    else: st.warning("Données insuffisantes.")

# Stats H2H
h2h_avg, nb = calculate_h2h_detailed(df_history_multi, selected_team, opp)
if h2h_avg:
    st.markdown(f"#### 📊 Moyennes vs {opp} ({nb} matchs)")
    k1, k2, k3, k4 = st.columns(4)
    def h2h_c(c, l, v1, v2): c.markdown(f"""<div class="metric-card" style="padding:10px;"><div class="metric-label">{l}</div><div style="font-size:1.2rem;font-weight:bold;color:white;"><span style="color:#2ECC71;">{v1:.1f}</span> / <span style="color:#E74C3C;">{v2:.1f}</span></div></div>""", unsafe_allow_html=True)
    h2h_c(k1, "Buts", h2h_avg['goals_for'], h2h_avg['goals_against'])
    h2h_c(k2, "Tirs", h2h_avg['shots_for'], h2h_avg['shots_against'])
    h2h_c(k3, "Cadrés", h2h_avg['target_for'], h2h_avg['target_against'])
    h2h_c(k4, "Jaunes", h2h_avg['yellow_for'], 0)

st.markdown("---")

# ================= FUTUR =================
NAME_MAPPING = { "Paris Saint Germain": "Paris SG", "Olympique de Marseille": "Marseille", "Olympique Lyonnais": "Lyon", "AS Monaco": "Monaco", "Lille OSC": "Lille", "Stade Rennais": "Rennes", "OGC Nice": "Nice", "RC Lens": "Lens", "Stade de Reims": "Reims", "Strasbourg Alsace": "Strasbourg", "Montpellier HSC": "Montpellier", "FC Nantes": "Nantes", "Toulouse FC": "Toulouse", "Stade Brestois 29": "Brest", "FC Lorient": "Lorient", "Clermont Foot": "Clermont", "Le Havre AC": "Le Havre", "FC Metz": "Metz", "AJ Auxerre": "Auxerre", "Angers SCO": "Angers", "AS Saint-Etienne": "Saint Etienne" }

st.subheader("🔮 Calendrier")
cutoff = pd.to_datetime(team_stats['match_timestamp']).replace(tzinfo=None)
replay = cutoff < (pd.Timestamp.now() - pd.Timedelta(days=7))

nxt = pd.DataFrame()
src = ""

if replay:
    src = "Historique (Simu)"
    fut = df_matchs_focus[df_matchs_focus['date'] > cutoff].sort_values('date')
    if not fut.empty:
        date = fut.iloc[0]['date']
        nxt = fut[fut['date'] <= (date + pd.Timedelta(days=5))].rename(columns={'date': 'DateUtc', 'home_team': 'HomeTeam', 'away_team': 'AwayTeam'})
else:
    src = "Live API"
    api = get_live_schedule()
    if not api.empty:
        rn = api.iloc[0]['RoundNumber']
        nxt = api[api['RoundNumber'] == rn]
        nxt['HomeTeam'] = nxt['HomeTeam'].apply(lambda x: NAME_MAPPING.get(x, x))
        nxt['AwayTeam'] = nxt['AwayTeam'].apply(lambda x: NAME_MAPPING.get(x, x))

if not nxt.empty:
    st.caption(f"Source : {src}")
    preds = []
    for _, m in nxt.iterrows():
        d, e = m['HomeTeam'], m['AwayTeam']
        # Pour le tableau général, on ne recalcule pas les Ranks stats (trop lourd), juste Poisson
        xh, xa, _ = predict_hybrid_score(df_history_multi, d, e, None)
        sc = f"{int(round(xh))} - {int(round(xa))}" if xh else "N/A"
        preds.append({"Date": m['DateUtc'].strftime('%d/%m %H:%M'), "Dom": d, "Score": sc, "Ext": e})
    st.dataframe(pd.DataFrame(preds).set_index("Date"), use_container_width=True)
else: st.info("Aucun match futur.")

st.markdown("---")
# GRAPH
g, t = st.columns([2, 1])
with g:
    st.subheader("📈 Trajectoire")
    hist = df_class_focus[(df_class_focus['equipe'] == selected_team) & (df_class_focus['journee_team'] <= selected_journee)]
    fig = px.line(hist, x='journee_team', y='total_points', markers=True)
    fig.update_traces(line_color='#DAE025', line_width=4)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, use_container_width=True)
with t:
    st.subheader("🏆 Classement")
    st.dataframe(df_snap[['rang', 'equipe', 'total_points', 'total_diff', 'total_V']].set_index('rang'), use_container_width=True)