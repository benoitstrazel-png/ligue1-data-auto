import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
import plotly.express as px
import numpy as np
from scipy.stats import poisson

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Ligue 1 Data Center", layout="wide", page_icon="⚽")

# --- STYLE CSS AVANCÉ (THEME SOMBRE & ROBUSTE) ---
st.markdown("""
    <style>
    /* FORCE LE FOND GRIS FONCÉ */
    .stApp { background-color: #1A1C23 !important; }
    
    /* TEXTES EN BLANC */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6, .main p, .main span, .main div, .main label, .main li {
        color: #FFFFFF !important; 
    }
    
    /* EXCEPTIONS KPI */
    .metric-card h3, .metric-card div, .metric-card span, .metric-card p { color: #DAE025 !important; }
    .metric-card .metric-label { color: #E0E0E0 !important; }
    
    /* SIDEBAR */
    [data-testid="stSidebar"] { background-color: #091C3E !important; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    
    /* DESIGN CARTES */
    .metric-card {
        background-color: #091C3E;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 10px;
        border: 1px solid #DAE025;
    }
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
        color: #FFFFFF !important;
        font-family: 'Courier New', monospace;
    }
    
    /* BADGES */
    .form-badge {
        display: inline-block; width: 30px; height: 30px; line-height: 30px;
        border-radius: 50%; text-align: center; font-weight: bold;
        color: white !important; margin-right: 5px; font-size: 0.8rem;
    }
    .win { background-color: #2ECC71; }
    .draw { background-color: #95A5A6; }
    .loss { background-color: #E74C3C; }
    
    /* Tooltip Helper */
    .tooltip-icon { font-size: 0.8rem; cursor: help; color: #DAE025; vertical-align: super; }
    </style>
""", unsafe_allow_html=True)

# --- CONNEXION BIGQUERY ---
@st.cache_resource
def get_db_client():
    key_dict = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(key_dict)
    return bigquery.Client(credentials=creds, project=key_dict["project_id"])

# --- CHARGEMENT DONNÉES ---
@st.cache_data(ttl=3600)
def get_seasons_list():
    client = get_db_client()
    return client.query("SELECT DISTINCT season FROM `ligue1-data.historic_datasets.matchs_clean` ORDER BY season DESC").to_dataframe()['season'].tolist()

@st.cache_data(ttl=600)
def load_focus_season(season_name):
    client = get_db_client()
    df_c = client.query(f"SELECT * FROM `ligue1-data.historic_datasets.classement_live` WHERE saison = '{season_name}' ORDER BY journee_team ASC").to_dataframe()
    df_m = client.query(f"SELECT * FROM `ligue1-data.historic_datasets.matchs_clean` WHERE season = '{season_name}' ORDER BY date ASC").to_dataframe()
    
    if not df_c.empty: df_c['match_timestamp'] = pd.to_datetime(df_c['match_timestamp'], utc=True).dt.tz_localize(None)
    if not df_m.empty: df_m['date'] = pd.to_datetime(df_m['date'], utc=True).dt.tz_localize(None)
    return df_c, df_m

@st.cache_data(ttl=600)
def load_multi_season_stats(seasons_list):
    client = get_db_client()
    s_str = "', '".join(seasons_list)
    df = client.query(f"SELECT * FROM `ligue1-data.historic_datasets.matchs_clean` WHERE season IN ('{s_str}')").to_dataframe()
    if not df.empty: df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    return df

# --- LOGIQUE MÉTIER & PRÉDICTIONS ---

def calculate_probabilities(xg_home, xg_away):
    """Calcule la matrice de probabilités exactes selon la loi de Poisson"""
    max_goals = 6
    prob_matrix = np.zeros((max_goals, max_goals))
    
    # On génère la grille des scores possibles (ex: proba que Home marque 2 ET Away marque 1)
    for i in range(max_goals):
        for j in range(max_goals):
            prob_matrix[i, j] = poisson.pmf(i, xg_home) * poisson.pmf(j, xg_away)
            
    # Probabilités issues de la matrice
    prob_home_win = np.sum(np.tril(prob_matrix, -1)) # Triangle inférieur
    prob_draw = np.sum(np.diag(prob_matrix))         # Diagonale
    prob_away_win = np.sum(np.triu(prob_matrix, 1))  # Triangle supérieur
    
    # Over / Under 2.5
    prob_over_2_5 = 0
    for i in range(max_goals):
        for j in range(max_goals):
            if (i + j) > 2.5:
                prob_over_2_5 += prob_matrix[i, j]
                
    return prob_home_win, prob_draw, prob_away_win, prob_over_2_5

def predict_match_advanced(df_history, team_home, team_away):
    if df_history.empty: return None
    
    # 1. Calcul des xG (Expected Goals)
    avg_h = df_history['full_time_home_goals'].mean()
    avg_a = df_history['full_time_away_goals'].mean()
    
    stats_h = df_history[df_history['home_team'] == team_home]
    stats_a = df_history[df_history['away_team'] == team_away]
    
    if stats_h.empty or stats_a.empty: return None
    
    att_h = stats_h['full_time_home_goals'].mean() / avg_h
    def_h = stats_h['full_time_away_goals'].mean() / avg_a
    att_a = stats_a['full_time_away_goals'].mean() / avg_a
    def_a = stats_a['full_time_home_goals'].mean() / avg_h
    
    xg_home = att_h * def_a * avg_h
    xg_away = att_a * def_h * avg_a
    
    # 2. Calcul des Probabilités
    p_win, p_draw, p_loss, p_over = calculate_probabilities(xg_home, xg_away)
    
    return {
        'xg_home': xg_home, 'xg_away': xg_away,
        'p_win': p_win, 'p_draw': p_draw, 'p_loss': p_loss,
        'p_over_2_5': p_over
    }

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

def calculate_nemesis_stats(df, team):
    stats = []
    df_team = df[(df['home_team'] == team) | (df['away_team'] == team)]
    if df_team.empty: return None

    for _, row in df_team.iterrows():
        is_home = row['home_team'] == team
        opponent = row['away_team'] if is_home else row['home_team']
        
        if row['full_time_result'] == 'D': result = 'N'
        elif (is_home and row['full_time_result'] == 'H') or (not is_home and row['full_time_result'] == 'A'): result = 'V'
        else: result = 'D'
        
        stats.append({
            'opponent': opponent, 'result': result,
            'goals_against': row['full_time_away_goals'] if is_home else row['full_time_home_goals'],
            'reds': row['home_red_cards'] if is_home else row['away_red_cards']
        })
    
    agg = pd.DataFrame(stats).groupby('opponent').agg({'result': list, 'goals_against': 'sum', 'reds': 'sum'})
    agg['wins'] = agg['result'].apply(lambda x: x.count('V'))
    agg['losses'] = agg['result'].apply(lambda x: x.count('D'))
    return agg

def apply_standings_style(df):
    """Applique le style Ligue des Champions / Relégation"""
    def style_rows(row):
        rank = row.name # Le rang est dans l'index (1, 2, 3...)
        css = ''
        # Couleurs avec transparence (0.6 = 60%)
        if rank <= 4: css = 'background-color: rgba(66, 133, 244, 0.4);' # Bleu C1
        elif rank == 5: css = 'background-color: rgba(255, 165, 0, 0.4);' # Orange C3
        elif rank == 6: css = 'background-color: rgba(46, 204, 113, 0.4);' # Vert C4
        elif rank == 16: css = 'background-color: rgba(231, 76, 60, 0.2);' # Rouge léger
        elif rank == 17: css = 'background-color: rgba(231, 76, 60, 0.35);' # Rouge moyen
        elif rank >= 18: css = 'background-color: rgba(231, 76, 60, 0.5);' # Rouge fort
        return css
    
    # Ajout Couronne
    df_styled = df.copy()
    if 1 in df_styled.index:
        df_styled.loc[1, 'equipe'] = "👑 " + df_styled.loc[1, 'equipe']
        
    return df_styled.style.apply(style_rows, axis=1)

# --- SIDEBAR ---
st.sidebar.title("🔍 Filtres")
all_seasons = get_seasons_list()
selected_seasons = st.sidebar.multiselect("Périmètre d'analyse", all_seasons, default=[all_seasons[0]])
if not selected_seasons: st.stop()
focus_season = sorted(selected_seasons, reverse=True)[0]
st.sidebar.markdown(f"**Saison Focus :** {focus_season}")

# Data Loading
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

# Helper Tooltip
help_kpi = "💡 **Explications :**\n- **Classement**: Position à la journée sélectionnée.\n- **Points**: Total cumulé (Victoire=3, Nul=1).\n- **Diff**: Buts marqués moins buts encaissés."
help_form = "💡 **Forme :**\nIndique les résultats des 5 derniers matchs.\nV = Victoire (Vert)\nN = Nul (Gris)\nD = Défaite (Rouge)"

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="metric-card" title="{help_kpi}"><div class="metric-label">Classement 💡</div><div class="metric-value">{int(team_stats["rang"])}e</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="metric-card"><div class="metric-label">Points</div><div class="metric-value">{int(team_stats["total_points"])}</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="metric-card"><div class="metric-label">Buts Pour</div><div class="metric-value">{int(team_stats["total_bp"])}</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="metric-card"><div class="metric-label">Diff.</div><div class="metric-value">{int(team_stats["total_diff"]):+d}</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ==============================================================================
# SECTION ANALYSE BÊTE NOIRE
# ==============================================================================
st.subheader("⚔️ Analyse des Confrontations", help="💡 **Analyse:** Basée sur toutes les saisons sélectionnées. Identifie les adversaires qui réussissent le mieux ou le moins bien à votre équipe.")
agg_stats = calculate_nemesis_stats(df_history_multi, selected_team)

if agg_stats is not None and not agg_stats.empty:
    def get_top(col, asc=False):
        srt = agg_stats.sort_values(col, ascending=asc)
        return (srt.index[0], srt.iloc[0][col]) if not srt.empty else ("N/A", 0)

    best, n_wins = get_top('wins')
    worst, n_loss = get_top('losses')
    leak, n_goals = get_top('goals_against')
    butcher, n_reds = get_top('reds')

    k1, k2, k3, k4 = st.columns(4)
    k1.success(f"**Proie Favorite**\n\n### {best}\n({n_wins} victoires)")
    k2.error(f"**Bête Noire**\n\n### {worst}\n({n_loss} défaites)")
    k3.warning(f"**Passoire contre**\n\n### {leak}\n({int(n_goals)} buts pris)")
    k4.info(f"**Tensions contre**\n\n### {butcher}\n({int(n_reds)} rouges)")
else:
    st.info("Pas assez de données pour l'analyse des confrontations.")

st.markdown("---")

# ==============================================================================
# SECTION DUEL & PRÉDICTION (COHÉRENT & AMÉLIORÉ)
# ==============================================================================
st.subheader("🔮 Prédiction & Paris", help="💡 **Modèle Mathématique :** Utilise la Loi de Poisson. Nous croisons la force offensive de l'équipe A avec la défense de l'équipe B sur la période choisie pour simuler 100% des scores possibles.")

col_sel, col_pred = st.columns([1, 2])

with col_sel:
    opponents = [t for t in teams if t != selected_team]
    opponent = st.selectbox("Adversaire :", opponents)
    match_location = st.radio("Lieu :", [f"Domicile ({selected_team})", f"Extérieur ({selected_team})"])
    is_home_game = "Domicile" in match_location

# Calculs & Prédictions
h2h_avg, nb_games = calculate_h2h_detailed(df_history_multi, selected_team, opponent)
p_home = selected_team if is_home_game else opponent
p_away = opponent if is_home_game else selected_team

# Prédiction
pred_data = predict_match_advanced(df_history_multi, p_home, p_away)

with col_pred:
    if pred_data:
        # Score le plus probable (basé sur xG arrondis)
        score_h, score_a = int(round(pred_data['xg_home'])), int(round(pred_data['xg_away']))
        
        # Détermination du Pari recommandé (Basé sur les PROBAS, pas le score exact)
        probs = {'Victoire ' + p_home: pred_data['p_win'], 'Match Nul': pred_data['p_draw'], 'Victoire ' + p_away: pred_data['p_loss']}
        best_bet = max(probs, key=probs.get)
        best_prob = probs[best_bet] * 100
        
        # Recommandation Over/Under
        ou_txt = "Plus de 2.5 Buts" if pred_data['p_over_2_5'] > 0.55 else "Moins de 2.5 Buts"
        ou_prob = pred_data['p_over_2_5'] if pred_data['p_over_2_5'] > 0.55 else (1 - pred_data['p_over_2_5'])
        
        st.markdown(f"""
            <div class="score-card">
                <div style="color: #DAE025; font-size: 0.9rem;">SCORE LE PLUS PROBABLE</div>
                <div class="score-display">{p_home} {score_h} - {score_a} {p_away}</div>
                <div style="display: flex; justify-content: space-around; margin-top: 15px;">
                    <div style="text-align:center;">
                        <span style="color: #2ECC71; font-weight: bold;">🎯 Recommandation 1N2</span><br>
                        <span style="font-size: 1.2rem; color: white;">{best_bet}</span><br>
                        <span style="color: #AAA; font-size: 0.8rem;">Fiabilité: {best_prob:.1f}%</span>
                    </div>
                    <div style="border-left: 1px solid #555;"></div>
                    <div style="text-align:center;">
                        <span style="color: #F1C40F; font-weight: bold;">⚽ Buts (Over/Under)</span><br>
                        <span style="font-size: 1.2rem; color: white;">{ou_txt}</span><br>
                        <span style="color: #AAA; font-size: 0.8rem;">Probabilité: {ou_prob*100:.1f}%</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Données insuffisantes pour la prédiction.")

st.markdown("---")

# ==============================================================================
# SECTION CLASSEMENT FINAL PROJETÉ (NOUVEAU)
# ==============================================================================
st.subheader("🏁 Projection Classement Final", help="💡 **Projection :** Estimation linéaire des points en fin de saison (34 journées) basée sur la moyenne de points actuelle.")

if not df_snap.empty:
    # Projection simple : (Points Actuels / Matchs Joués) * 34
    df_proj = df_snap.copy()
    # On estime le nombre de matchs joués par la journée sélectionnée
    nb_matchs_played = selected_journee 
    
    if nb_matchs_played > 0:
        df_proj['pts_proj'] = (df_proj['total_points'] / nb_matchs_played) * 34
        df_proj['pts_proj'] = df_proj['pts_proj'].round(0).astype(int)
        df_proj = df_proj.sort_values(['pts_proj', 'total_diff'], ascending=False).reset_index(drop=True)
        df_proj.index = df_proj.index + 1 # Rang commence à 1
        
        # Affichage avec style
        st.dataframe(
            apply_standings_style(df_proj[['equipe', 'pts_proj', 'total_points']]),
            column_config={
                "equipe": "Équipe",
                "pts_proj": st.column_config.NumberColumn("Points Finaux (Est.)", help="Estimation mathématique fin de saison"),
                "total_points": "Points Actuels"
            },
            use_container_width=True,
            height=400
        )
    else:
        st.info("En attente de matchs joués pour la projection.")

st.markdown("---")

# ==============================================================================
# CLASSEMENT LIVE (AVEC STYLE)
# ==============================================================================
st.subheader(f"🏆 Classement Live (J{selected_journee})", help="💡 **Légende Couleurs :**\n🔵 Top 4 : Ligue des Champions\n🟠 5ème : Europa League\n🟢 6ème : Conférence League\n🔴 16-18ème : Zone Relégation")

# Préparation dataframe clean
df_display = df_snap[['rang', 'equipe', 'total_points', 'total_diff', 'total_V', 'total_N', 'total_D']].set_index('rang')

# Application du style visuel demandé
st.dataframe(
    apply_standings_style(df_display),
    use_container_width=True,
    height=600
)