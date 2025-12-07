import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Ligue 1 Data Center", layout="wide", page_icon="⚽")

# --- STYLE CSS AVANCÉ ---
st.markdown("""
    <style>
    /* Fond général */
    .stApp { background-color: #F8F9FA; }
    
    /* Cartes KPI */
    .metric-card {
        background-color: #091C3E;
        color: #DAE025;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .metric-value { font-size: 2rem; font-weight: 800; margin: 0; }
    .metric-label { font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9; }
    
    /* Pastilles de forme (V-N-D) */
    .form-badge {
        display: inline-block;
        width: 30px;
        height: 30px;
        line-height: 30px;
        border-radius: 50%;
        text-align: center;
        font-weight: bold;
        color: white;
        margin-right: 5px;
        font-size: 0.8rem;
    }
    .win { background-color: #2ECC71; }
    .draw { background-color: #95A5A6; }
    .loss { background-color: #E74C3C; }
    
    /* Sections */
    h3 { color: #091C3E; border-bottom: 2px solid #DAE025; padding-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- CONNEXION BIGQUERY ---
@st.cache_resource
def get_db_client():
    key_dict = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(key_dict)
    return bigquery.Client(credentials=creds, project=key_dict["project_id"])

# --- CHARGEMENT DES DONNÉES (OPTIMISÉ) ---
@st.cache_data(ttl=3600)
def get_seasons():
    """Récupère la liste des saisons disponibles"""
    client = get_db_client()
    query = "SELECT DISTINCT season FROM `ligue1-data.historic_datasets.matchs_clean` ORDER BY season DESC"
    return client.query(query).to_dataframe()['season'].tolist()

@st.cache_data(ttl=600)
def load_season_data(selected_season):
    """Charge les données pour une saison spécifique"""
    client = get_db_client()
    
    # 1. Classement (Tout l'historique de la saison pour pouvoir filtrer par journée ensuite)
    q_class = f"""
        SELECT * FROM `ligue1-data.historic_datasets.classement_live`
        WHERE saison = '{selected_season}'
        ORDER BY journee_team ASC
    """
    
    # 2. Matchs (Pour les KPIs détaillés et le calendrier)
    q_matchs = f"""
        SELECT * FROM `ligue1-data.historic_datasets.matchs_clean`
        WHERE season = '{selected_season}'
        ORDER BY date ASC
    """
    
    df_class = client.query(q_class).to_dataframe()
    df_matchs = client.query(q_matchs).to_dataframe()
    
    # Conversion dates
    df_class['match_timestamp'] = pd.to_datetime(df_class['match_timestamp'])
    df_matchs['date'] = pd.to_datetime(df_matchs['date'])
    
    return df_class, df_matchs

@st.cache_data(ttl=3600)
def load_historical_stats(team_name):
    """Charge TOUT l'historique d'une équipe (toutes saisons) pour l'analyse 'Bête Noire'"""
    client = get_db_client()
    query = f"""
        SELECT * FROM `ligue1-data.historic_datasets.matchs_clean`
        WHERE home_team = "{team_name}" OR away_team = "{team_name}"
    """
    return client.query(query).to_dataframe()

# --- LOGIQUE DE CALCUL KPI AVANCÉS ---
def get_form_badges(last_matches, team):
    html = ""
    for _, row in last_matches.iterrows():
        res = 'draw'
        val = 'N'
        
        # Déterminer si Victoire/Nul/Défaite pour l'équipe sélectionnée
        if row['full_time_result'] == 'D':
            res = 'draw'
            val = 'N'
        elif (row['home_team'] == team and row['full_time_result'] == 'H') or \
             (row['away_team'] == team and row['full_time_result'] == 'A'):
            res = 'win'
            val = 'V'
        else:
            res = 'loss'
            val = 'D'
            
        # Tooltip avec l'adversaire
        adversaire = row['away_team'] if row['home_team'] == team else row['home_team']
        score = f"{int(row['full_time_home_goals'])}-{int(row['full_time_away_goals'])}"
        
        html += f'<div class="form-badge {res}" title="{adversaire} ({score})">{val}</div>'
    return html

def calculate_nemesis_stats(df, team):
    """Calcule les stats 'Bête Noire' / 'Favori' sur un dataframe donné"""
    # On prépare un dataset orienté "Equipe vs Adversaire"
    stats = []
    
    for _, row in df.iterrows():
        is_home = row['home_team'] == team
        opponent = row['away_team'] if is_home else row['home_team']
        
        # Score
        goals_for = row['full_time_home_goals'] if is_home else row['full_time_away_goals']
        goals_against = row['full_time_away_goals'] if is_home else row['full_time_home_goals']
        
        # Résultat
        result = 'N'
        if row['full_time_result'] == 'D': result = 'N'
        elif (is_home and row['full_time_result'] == 'H') or (not is_home and row['full_time_result'] == 'A'): result = 'V'
        else: result = 'D'
        
        # Cartons (si dispo)
        yellows = row['home_yellow_cards'] if is_home else row['away_yellow_cards']
        reds = row['home_red_cards'] if is_home else row['away_red_cards']
        
        # Remontada (Perdait à la mi-temps -> Gagne à la fin)
        ht_result = row['half_time_result']
        ft_result = row['full_time_result']
        is_remontada = False
        
        if is_home:
            if ht_result == 'A' and ft_result == 'H': is_remontada = True
        else:
            if ht_result == 'H' and ft_result == 'A': is_remontada = True
            
        stats.append({
            'opponent': opponent,
            'result': result,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'yellows': yellows,
            'reds': reds,
            'remontada': 1 if is_remontada else 0
        })
    
    if not stats:
        return None

    df_stats = pd.DataFrame(stats)
    
    # Agrégation par adversaire
    agg = df_stats.groupby('opponent').agg({
        'result': list,
        'goals_for': 'sum',
        'goals_against': 'sum',
        'yellows': 'sum',
        'reds': 'sum',
        'remontada': 'sum'
    })
    
    agg['wins'] = agg['result'].apply(lambda x: x.count('V'))
    agg['losses'] = agg['result'].apply(lambda x: x.count('D'))
    agg['draws'] = agg['result'].apply(lambda x: x.count('N'))
    agg['total_games'] = agg['result'].apply(len)
    
    return agg

# --- INTERFACE UTILISATEUR ---

# 1. Sidebar - Filtres Globaux
st.sidebar.title("🔍 Filtres d'Analyse")

seasons_list = get_seasons()
selected_season = st.sidebar.selectbox("Saison", seasons_list)

# Chargement initial pour la saison choisie
try:
    df_classement_full, df_matchs_full = load_season_data(selected_season)
except Exception as e:
    st.error(f"Erreur chargement données : {e}")
    st.stop()

# Sélecteur d'équipe (basé sur la saison choisie)
teams_in_season = sorted(df_classement_full['equipe'].unique())
selected_team = st.sidebar.selectbox("Choisir une équipe", teams_in_season)

# Slider Journée (Dynamique selon la saison)
max_journee = int(df_classement_full['journee_team'].max())
selected_journee = st.sidebar.slider("Arrêter l'analyse à la journée :", 1, max_journee, max_journee)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Info :** En changeant la journée, tout le dashboard (classement, stats, forme) se recalcule comme si nous étions à cette date précise.")

# --- FILTRAGE TEMPOREL (LA MACHINE A VOYAGER DANS LE TEMPS) ---
# On filtre le classement pour ne garder que ce qui s'est passé <= journée choisie
# On prend la ligne la plus récente pour chaque équipe JUSQU'A la journée X
df_classement_filtered = df_classement_full[df_classement_full['journee_team'] <= selected_journee]
df_classement_snapshot = df_classement_filtered.sort_values('match_timestamp').groupby('equipe').last().reset_index()

# On recalcule le rang dynamiquement sur ce snapshot
df_classement_snapshot = df_classement_snapshot.sort_values(['total_points', 'total_diff', 'total_bp'], ascending=[False, False, False])
df_classement_snapshot['rang'] = range(1, len(df_classement_snapshot) + 1)

# Stats de l'équipe sélectionnée au moment T
team_stats_t = df_classement_snapshot[df_classement_snapshot['equipe'] == selected_team].iloc[0]

# --- DASHBOARD HEADER ---
st.title(f"📊 Rapport : {selected_team}")
st.markdown(f"**Saison {selected_season}** | Arrêté à la **Journée {selected_journee}**")

# --- LIGNE 1 : KPI CLÉS & FORME ---
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])

def kpi(col, label, val):
    col.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{val}</div></div>""", unsafe_allow_html=True)

kpi(col1, "Classement", f"{int(team_stats_t['rang'])}e")
kpi(col2, "Points", int(team_stats_t['total_points']))
kpi(col3, "Diff.", f"{int(team_stats_t['total_diff']):+d}")
kpi(col4, "Buts Pour", int(team_stats_t['total_bp']))

# Forme (5 derniers matchs avant la date de la journée sélectionnée)
with col5:
    st.markdown("##### 📅 Forme (5 derniers matchs)")
    # Date butoir de la journée sélectionnée
    cutoff_date = team_stats_t['match_timestamp']
    
    # On prend les matchs de l'équipe AVANT cette date
    past_matches = df_matchs_full[
        ((df_matchs_full['home_team'] == selected_team) | (df_matchs_full['away_team'] == selected_team)) &
        (df_matchs_full['date'] <= cutoff_date) &
        (df_matchs_full['full_time_result'].notna())
    ].sort_values('date', ascending=False).head(5) # Les 5 plus récents
    
    # Inverser pour avoir chronologique (Plus vieux -> Plus récent) pour l'affichage visuel
    form_html = get_form_badges(past_matches.sort_values('date'), selected_team)
    st.markdown(form_html, unsafe_allow_html=True)
    st.caption("Survolez les pastilles pour voir l'adversaire")

st.markdown("---")

# --- LIGNE 2 : ANALYSE STATISTIQUE AVANCÉE ---
# On charge l'historique complet pour l'analyse "Bête noire"
# Mais on peut choisir de le faire soit sur la Saison en cours (filtrée), soit sur l'Histoire
analysis_mode = st.radio("Base d'analyse pour les statistiques adverses :", 
                         ["Saison en cours (jusqu'à J" + str(selected_journee) + ")", "Historique Global (Toutes saisons)"], 
                         horizontal=True)

if "Saison" in analysis_mode:
    # On utilise les matchs de la saison coupés à la date
    df_analysis = df_matchs_full[
        ((df_matchs_full['home_team'] == selected_team) | (df_matchs_full['away_team'] == selected_team)) &
        (df_matchs_full['date'] <= cutoff_date)
    ]
else:
    # On charge tout
    df_analysis = load_historical_stats(selected_team)

agg_stats = calculate_nemesis_stats(df_analysis, selected_team)

if agg_stats is not None and not agg_stats.empty:
    st.subheader("⚔️ Confrontations & Bêtes Noires")
    
    c1, c2, c3, c4 = st.columns(4)
    
    # Helper pour trouver le max en gérant les égalités
    def get_top_stat(col, asc=False):
        if df_analysis.empty: return "N/A", 0
        sorted_df = agg_stats.sort_values(col, ascending=asc)
        if sorted_df.empty: return "N/A", 0
        top_team = sorted_df.index[0]
        val = sorted_df.iloc[0][col]
        return top_team, val

    # 1. Victoires
    best_opponent, nb_wins = get_top_stat('wins')
    c1.success(f"**Proie Favorite**\n\n### {best_opponent}\n({nb_wins} victoires)")
    
    # 2. Défaites
    worst_opponent, nb_loss = get_top_stat('losses')
    c2.error(f"**Bête Noire**\n\n### {worst_opponent}\n({nb_loss} défaites)")
    
    # 3. Buts Encaissés
    leak_opponent, nb_goals = get_top_stat('goals_against')
    c3.warning(f"**Passoire contre**\n\n### {leak_opponent}\n({int(nb_goals)} buts pris)")
    
    # 4. Cartons Rouges (si dispo)
    red_opponent, nb_reds = get_top_stat('reds')
    if nb_reds > 0:
        c4.error(f"**Boucherie contre**\n\n### {red_opponent}\n({int(nb_reds)} rouges)")
    else:
        # Sinon Remontada
        remontada_opp, nb_rem = get_top_stat('remontada')
        c4.info(f"**Roi du Comeback vs**\n\n### {remontada_opp}\n({int(nb_rem)} fois)")

    with st.expander(f"Voir le détail complet contre tous les adversaires ({analysis_mode})"):
        st.dataframe(agg_stats.style.background_gradient(cmap="RdYlGn", subset=['wins', 'losses']), use_container_width=True)
else:
    st.info("Pas assez de données pour générer les statistiques avancées sur cette période.")

st.markdown("---")

# --- LIGNE 3 : GRAPHIQUE & TABLEAU ---
col_graph, col_tab = st.columns([2, 1])

with col_graph:
    st.subheader("📈 Trajectoire")
    # Historique de la saison pour l'équipe (filtré à la journée)
    history_team = df_classement_full[
        (df_classement_full['equipe'] == selected_team) & 
        (df_classement_full['journee_team'] <= selected_journee)
    ]
    
    fig = px.line(history_team, x='journee_team', y='total_points', 
                  markers=True, title=f"Points cumulés - {selected_team}")
    fig.update_traces(line_color='#DAE025', line_width=4, marker_size=8)
    fig.update_layout(xaxis_title="Journée", yaxis_title="Points", 
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

with col_tab:
    st.subheader(f"🏆 Classement J{selected_journee}")
    # On affiche un tableau propre
    st.dataframe(
        df_classement_snapshot[['rang', 'equipe', 'total_points', 'total_diff', 'total_V', 'total_N', 'total_D']]
        .set_index('rang'),
        height=400,
        use_container_width=True
    )