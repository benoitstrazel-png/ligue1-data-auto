import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
import plotly.express as px

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Ligue 1 Data Center", layout="wide", page_icon="⚽")

# --- STYLE CSS AVANCÉ (CORRIGÉ POUR LISIBILITÉ) ---
st.markdown("""
    <style>
    /* 1. Fond général de l'application */
    .stApp {
        background-color: #F4F4F4;
    }
    
    /* 2. FORCE LA COULEUR DES TEXTES ET TITRES EN BLEU FONCÉ */
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #091C3E; 
    }
    
    /* 3. Exception : Les textes à l'intérieur des cartes métriques personnalisées restent jaunes/blancs */
    .metric-card h3, .metric-card div, .metric-card span {
        color: #DAE025 !important; /* Jaune pour les valeurs */
    }
    .metric-card .metric-label {
        color: #E0E0E0 !important; /* Blanc cassé pour les étiquettes dans le bleu */
    }

    /* 4. Style des cartes KPI (Bleu foncé) */
    .metric-card {
        background-color: #091C3E;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        color: #DAE025 !important;
    }
    
    /* 5. Style des titres de section natifs Streamlit */
    .stHeadingContainer {
        color: #091C3E !important;
    }

    /* 6. Pastilles de forme (V-N-D) */
    .form-badge {
        display: inline-block;
        width: 30px;
        height: 30px;
        line-height: 30px;
        border-radius: 50%;
        text-align: center;
        font-weight: bold;
        color: white !important; /* Force le blanc dans les pastilles */
        margin-right: 5px;
        font-size: 0.8rem;
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
    """Récupère la liste des saisons disponibles"""
    client = get_db_client()
    query = "SELECT DISTINCT season FROM `ligue1-data.historic_datasets.matchs_clean` ORDER BY season DESC"
    return client.query(query).to_dataframe()['season'].tolist()

@st.cache_data(ttl=600)
def load_focus_season(season_name):
    """Charge les données pour la saison 'FOCUS' (celle du classement)"""
    client = get_db_client()
    
    # Pour le classement, on ne veut QUE cette saison
    q_class = f"""
        SELECT * FROM `ligue1-data.historic_datasets.classement_live`
        WHERE saison = '{season_name}'
        ORDER BY journee_team ASC
    """
    
    # Pour le calendrier de cette saison
    q_matchs = f"""
        SELECT * FROM `ligue1-data.historic_datasets.matchs_clean`
        WHERE season = '{season_name}'
        ORDER BY date ASC
    """
    
    df_class = client.query(q_class).to_dataframe()
    df_matchs = client.query(q_matchs).to_dataframe()
    
    # Nettoyage Timezone
    if not df_class.empty:
        df_class['match_timestamp'] = pd.to_datetime(df_class['match_timestamp'], utc=True).dt.tz_localize(None)
    if not df_matchs.empty:
        df_matchs['date'] = pd.to_datetime(df_matchs['date'], utc=True).dt.tz_localize(None)
        
    return df_class, df_matchs

@st.cache_data(ttl=600)
def load_multi_season_stats(seasons_list):
    """Charge l'historique des matchs pour TOUTES les saisons sélectionnées"""
    client = get_db_client()
    
    # On formate la liste pour le SQL: ('2024-2025', '2023-2024')
    seasons_str = "', '".join(seasons_list)
    
    query = f"""
        SELECT * FROM `ligue1-data.historic_datasets.matchs_clean`
        WHERE season IN ('{seasons_str}')
    """
    df = client.query(query).to_dataframe()
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
        
    return df

# --- LOGIQUE MÉTIER ---

def get_form_badges(last_matches, team):
    html = ""
    for _, row in last_matches.iterrows():
        val = 'N'
        res = 'draw'
        
        if row['full_time_result'] == 'D':
            res = 'draw'; val = 'N'
        elif (row['home_team'] == team and row['full_time_result'] == 'H') or \
             (row['away_team'] == team and row['full_time_result'] == 'A'):
            res = 'win'; val = 'V'
        else:
            res = 'loss'; val = 'D'
            
        adversaire = row['away_team'] if row['home_team'] == team else row['home_team']
        score = f"{int(row['full_time_home_goals'])}-{int(row['full_time_away_goals'])}"
        
        html += f'<div class="form-badge {res}" title="{adversaire} ({score})">{val}</div>'
    return html

def calculate_nemesis_stats(df, team):
    """Calcule les stats agrégées sur le dataset fourni (multi-saisons ou global)"""
    stats = []
    
    # On filtre d'abord pour ne garder que les matchs de l'équipe
    # C'est plus rapide que d'itérer sur tout le dataframe
    df_team = df[(df['home_team'] == team) | (df['away_team'] == team)]
    
    if df_team.empty:
        return None

    for _, row in df_team.iterrows():
        is_home = row['home_team'] == team
        opponent = row['away_team'] if is_home else row['home_team']
        
        goals_for = row['full_time_home_goals'] if is_home else row['full_time_away_goals']
        goals_against = row['full_time_away_goals'] if is_home else row['full_time_home_goals']
        
        if row['full_time_result'] == 'D': result = 'N'
        elif (is_home and row['full_time_result'] == 'H') or (not is_home and row['full_time_result'] == 'A'): result = 'V'
        else: result = 'D'
        
        # Remontada logic
        is_remontada = False
        if is_home and row['half_time_result'] == 'A' and row['full_time_result'] == 'H': is_remontada = True
        elif not is_home and row['half_time_result'] == 'H' and row['full_time_result'] == 'A': is_remontada = True
            
        stats.append({
            'opponent': opponent, 'result': result,
            'goals_for': goals_for, 'goals_against': goals_against,
            'reds': row['home_red_cards'] if is_home else row['away_red_cards'],
            'remontada': 1 if is_remontada else 0
        })
    
    df_stats = pd.DataFrame(stats)
    
    agg = df_stats.groupby('opponent').agg({
        'result': list, 'goals_for': 'sum', 'goals_against': 'sum',
        'reds': 'sum', 'remontada': 'sum'
    })
    
    agg['wins'] = agg['result'].apply(lambda x: x.count('V'))
    agg['losses'] = agg['result'].apply(lambda x: x.count('D'))
    agg['draws'] = agg['result'].apply(lambda x: x.count('N'))
    agg['total_games'] = agg['result'].apply(len)
    
    return agg

# --- SIDEBAR FILTRES ---
st.sidebar.title("🔍 Filtres d'Analyse")

all_seasons = get_seasons_list()

# 1. Sélection MULTIPLE des saisons
selected_seasons = st.sidebar.multiselect(
    "Périmètre d'analyse (Historique)", 
    all_seasons, 
    default=[all_seasons[0]] # Par défaut la plus récente
)

if not selected_seasons:
    st.warning("Veuillez sélectionner au moins une saison.")
    st.stop()

# 2. Définition de la saison "FOCUS" (pour le classement)
# On prend automatiquement la saison la plus récente parmi celles sélectionnées
focus_season = sorted(selected_seasons, reverse=True)[0]

st.sidebar.markdown(f"**Saison de référence :** {focus_season}")

# 3. Chargement des données
try:
    # Données pour le classement (Saison Focus uniquement)
    df_class_focus, df_matchs_focus = load_focus_season(focus_season)
    
    # Données pour l'analyse (Toutes saisons sélectionnées)
    df_history_multi = load_multi_season_stats(selected_seasons)
    
except Exception as e:
    st.error(f"Erreur chargement : {e}")
    st.stop()

# 4. Sélection Équipe et Journée
teams_in_season = sorted(df_class_focus['equipe'].unique())
selected_team = st.sidebar.selectbox("Choisir une équipe", teams_in_season)

max_journee = int(df_class_focus['journee_team'].max()) if not df_class_focus.empty else 1
selected_journee = st.sidebar.slider(f"Simuler le classement à la journée (Saison {focus_season}):", 1, max_journee, max_journee)

st.sidebar.markdown("---")
st.sidebar.info("Le classement s'affiche pour la saison la plus récente sélectionnée. Les statistiques (Bête noire, etc.) prennent en compte TOUTES les saisons cochées.")

# --- TRAITEMENT TEMPOREL (SNAPSHOT) ---
df_class_filtered = df_class_focus[df_class_focus['journee_team'] <= selected_journee]
df_snapshot = df_class_filtered.sort_values('match_timestamp').groupby('equipe').last().reset_index()

# Recalcul du classement
df_snapshot = df_snapshot.sort_values(['total_points', 'total_diff', 'total_bp'], ascending=[False, False, False])
df_snapshot['rang'] = range(1, len(df_snapshot) + 1)

try:
    team_stats_t = df_snapshot[df_snapshot['equipe'] == selected_team].iloc[0]
except IndexError:
    st.error(f"L'équipe {selected_team} n'a pas encore joué à la journée {selected_journee}.")
    st.stop()

# --- DISPLAY HEADER ---
st.title(f"📊 Rapport : {selected_team}")
st.markdown(f"**Saison {focus_season}** | Arrêté à la **Journée {selected_journee}**")

# --- LIGNE 1 : KPI ---
c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 2])
c1.metric("Classement", f"{int(team_stats_t['rang'])}e")
c2.metric("Points", int(team_stats_t['total_points']))
c3.metric("Diff.", f"{int(team_stats_t['total_diff']):+d}")
c4.metric("Buts Pour", int(team_stats_t['total_bp']))

with c5:
    st.markdown("##### 📅 Forme (5 derniers matchs)")
    cutoff_date = pd.to_datetime(team_stats_t['match_timestamp']).replace(tzinfo=None)
    
    past_matches = df_matchs_focus[
        ((df_matchs_focus['home_team'] == selected_team) | (df_matchs_focus['away_team'] == selected_team)) &
        (df_matchs_focus['date'] <= cutoff_date) &
        (df_matchs_focus['full_time_result'].notna())
    ].sort_values('date', ascending=False).head(5)
    
    if not past_matches.empty:
        st.markdown(get_form_badges(past_matches.sort_values('date'), selected_team), unsafe_allow_html=True)
    else:
        st.write("Pas de matchs joués")

st.markdown("---")

# --- LIGNE 2 : ANALYSE PROFONDEUR ---
st.subheader(f"⚔️ Analyse des Confrontations ({len(selected_seasons)} saisons analysées)")

# Calcul des stats sur le dataset MULTI-SAISONS
agg_stats = calculate_nemesis_stats(df_history_multi, selected_team)

if agg_stats is not None and not agg_stats.empty:
    c1, c2, c3, c4 = st.columns(4)
    
    def get_top(col, asc=False):
        srt = agg_stats.sort_values(col, ascending=asc)
        return (srt.index[0], srt.iloc[0][col]) if not srt.empty else ("N/A", 0)

    best_opp, n_wins = get_top('wins')
    c1.success(f"**Proie Favorite**\n\n### {best_opp}\n({n_wins} victoires)")
    
    worst_opp, n_loss = get_top('losses')
    c2.error(f"**Bête Noire**\n\n### {worst_opp}\n({n_loss} défaites)")
    
    leak_opp, n_goals = get_top('goals_against')
    c3.warning(f"**Passoire contre**\n\n### {leak_opp}\n({int(n_goals)} buts pris)")
    
    red_opp, n_reds = get_top('reds')
    if n_reds > 0:
        c4.error(f"**Boucherie contre**\n\n### {red_opp}\n({int(n_reds)} rouges)")
    else:
        rem_opp, n_rem = get_top('remontada')
        c4.info(f"**Roi du Comeback vs**\n\n### {rem_opp}\n({int(n_rem)} fois)")

    with st.expander("Voir le tableau détaillé complet"):
        # Affichage sans style gradient si matplotlib manque, ou simple dataframe
        try:
            st.dataframe(
                agg_stats.style.background_gradient(cmap="RdYlGn", subset=['wins', 'losses']), 
                use_container_width=True
            )
        except ImportError:
            # Fallback si matplotlib n'est pas installé malgré tout
            st.dataframe(agg_stats, use_container_width=True)
else:
    st.info("Pas de données suffisantes sur la période sélectionnée.")

st.markdown("---")

# --- LIGNE 3 : GRAPHIQUE & TABLEAU ---
col_graph, col_tab = st.columns([2, 1])

with col_graph:
    st.subheader(f"📈 Trajectoire (Saison {focus_season})")
    history_team = df_class_focus[
        (df_class_focus['equipe'] == selected_team) & 
        (df_class_focus['journee_team'] <= selected_journee)
    ]
    fig = px.line(history_team, x='journee_team', y='total_points', markers=True, 
                  labels={'journee_team': 'Journée', 'total_points': 'Points'})
    fig.update_traces(line_color='#DAE025', line_width=4)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

with col_tab:
    st.subheader("🏆 Classement Live")
    st.dataframe(
        df_snapshot[['rang', 'equipe', 'total_points', 'total_diff', 'total_V', 'total_N', 'total_D']].set_index('rang'),
        height=400,
        use_container_width=True
    )