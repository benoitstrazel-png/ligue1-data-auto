import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import requests

# --- CONFIG ---
st.set_page_config(page_title="Ligue 1 Data Center", layout="wide", page_icon="⚽")
st.markdown("""
    <style>
    .stApp { background-color: #1A1C23; }
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6, .main p, .main span, .main div, .main label { color: #FFFFFF !important; }
    .metric-card h3, .metric-card div, .metric-card span { color: #DAE025 !important; }
    .metric-card .metric-label { color: #E0E0E0 !important; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    .metric-card { background-color: #091C3E; padding: 15px; border-radius: 12px; text-align: center; border: 1px solid #DAE025; margin-bottom: 10px; }
    .metric-value { font-size: 1.8rem; font-weight: 800; margin: 0; color: #DAE025 !important; }
    .score-card { background: linear-gradient(135deg, #091C3E 0%, #1A1C23 100%); border: 2px solid #DAE025; border-radius: 15px; padding: 20px; text-align: center; margin-top: 20px; }
    .score-display { font-size: 3.5rem; font-weight: bold; color: #FFFFFF; font-family: monospace; }
    .ref-card { background-color: #2C3E50; padding: 15px; border-radius: 10px; border-left: 5px solid #E74C3C; margin-top: 10px; }
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
def load_referee_data():
    """Charge la table référentiel arbitres"""
    client = get_db_client()
    # On charge l'historique complet des arbitres pour avoir des stats solides
    query = f"""
        SELECT season, date, home_team, away_team, referee_clean as referee, 
               home_yellow_cards, away_yellow_cards, home_red_cards, away_red_cards
        FROM `{client.project}.historic_datasets.referee_details`
        WHERE referee_clean IS NOT NULL AND referee_clean != 'INCONNU'
    """
    return client.query(query).to_dataframe()

@st.cache_data(ttl=600)
def load_focus_season(season_name):
    client = get_db_client()
    project_id = client.project
    q_class = f"SELECT * FROM `{project_id}.historic_datasets.classement_live` WHERE saison = '{season_name}' ORDER BY journee_team ASC"
    q_matchs = f"SELECT * FROM `{project_id}.historic_datasets.matchs_clean` WHERE season = '{season_name}' ORDER BY date ASC"
    
    df_c = client.query(q_class).to_dataframe()
    df_m = client.query(q_matchs).to_dataframe()
    
    if not df_c.empty: df_c['match_timestamp'] = pd.to_datetime(df_c['match_timestamp'], utc=True).dt.tz_localize(None)
    if not df_m.empty: df_m['date'] = pd.to_datetime(df_m['date'], utc=True).dt.tz_localize(None)
    return df_c, df_m

@st.cache_data(ttl=600)
def load_multi_season_stats(seasons_list):
    client = get_db_client()
    seasons_str = "', '".join(seasons_list)
    query = f"SELECT * FROM `{client.project}.historic_datasets.matchs_clean` WHERE season IN ('{seasons_str}')"
    df = client.query(query).to_dataframe()
    if not df.empty: df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    return df

@st.cache_data(ttl=3600)
def get_live_schedule():
    try:
        url = "https://fixturedownload.com/feed/json/ligue-1-2024"
        df = pd.DataFrame(requests.get(url).json())
        df['DateUtc'] = pd.to_datetime(df['DateUtc']).dt.tz_localize(None)
        return df[df['DateUtc'] > pd.Timestamp.now()].sort_values('DateUtc')
    except: return pd.DataFrame()

# --- LOGIQUE ARBITRE ---
def analyze_referee(df_refs, referee_name, team_home, team_away):
    """Calcule les stats détaillées d'un arbitre"""
    if referee_name is None: return None
    
    # 1. Stats Globales de l'arbitre (Tout historique)
    ref_matches = df_refs[df_refs['referee'] == referee_name]
    if ref_matches.empty: return None
    
    nb_matchs = len(ref_matches)
    total_yellow = ref_matches['home_yellow_cards'].sum() + ref_matches['away_yellow_cards'].sum()
    total_red = ref_matches['home_red_cards'].sum() + ref_matches['away_red_cards'].sum()
    
    avg_yellow = total_yellow / nb_matchs
    avg_red = total_red / nb_matchs
    
    # 2. Stats vs Equipe Domicile
    home_history = ref_matches[(ref_matches['home_team'] == team_home) | (ref_matches['away_team'] == team_home)]
    cards_received_home = 0
    if not home_history.empty:
        for _, r in home_history.iterrows():
            if r['home_team'] == team_home: cards_received_home += r['home_yellow_cards']
            else: cards_received_home += r['away_yellow_cards']
    avg_yellow_vs_home = cards_received_home / len(home_history) if not home_history.empty else 0
    
    # 3. Stats vs Equipe Extérieur
    away_history = ref_matches[(ref_matches['home_team'] == team_away) | (ref_matches['away_team'] == team_away)]
    cards_received_away = 0
    if not away_history.empty:
        for _, r in away_history.iterrows():
            if r['home_team'] == team_away: cards_received_away += r['home_yellow_cards']
            else: cards_received_away += r['away_yellow_cards']
    avg_yellow_vs_away = cards_received_away / len(away_history) if not away_history.empty else 0
    
    # Risque exclusion
    red_risk = "FAIBLE"
    if avg_red > 0.3: red_risk = "ÉLEVÉ 🔥"
    elif avg_red > 0.15: red_risk = "MODÉRÉ"
    
    return {
        "name": referee_name,
        "matches": nb_matchs,
        "avg_yellow_global": avg_yellow,
        "avg_red_global": avg_red,
        "risk": red_risk,
        "avg_yellow_home": avg_yellow_vs_home,
        "avg_yellow_away": avg_yellow_vs_away,
        "nb_match_home": len(home_history),
        "nb_match_away": len(away_history)
    }

# --- FONCTIONS PREDICT & H2H (Conservez les versions précédentes) ---
def calculate_streak_probability(df, streak_length=3):
    # (Copier la logique précédente ou laisser simplifié ici pour l'exemple)
    return None, 0 # Placeholder pour raccourcir

def predict_match_score(df_history, th, ta):
    # Logique Poisson simple
    if df_history.empty: return None, None
    hm = df_history[df_history['home_team']==th]
    am = df_history[df_history['away_team']==ta]
    if hm.empty or am.empty: return None, None
    # Moyennes
    avg_hg = df_history['full_time_home_goals'].mean()
    avg_ag = df_history['full_time_away_goals'].mean()
    # Forces
    att_h = hm['full_time_home_goals'].mean() / avg_hg
    def_h = hm['full_time_away_goals'].mean() / avg_ag
    att_a = am['full_time_away_goals'].mean() / avg_ag
    def_a = am['full_time_home_goals'].mean() / avg_hg
    return att_h * def_a * avg_hg, att_a * def_h * avg_ag

# --- SIDEBAR ---
st.sidebar.title("🔍 Filtres")
all_seasons = get_seasons_list()
selected_seasons = st.sidebar.multiselect("Périmètre", all_seasons, default=[all_seasons[0]])
if not selected_seasons: st.stop()
focus_season = sorted(selected_seasons, reverse=True)[0]

# LOAD
df_class, df_matchs = load_focus_season(focus_season)
df_history = load_multi_season_stats(selected_seasons)
df_referees = load_referee_data() # NOUVEAU

teams = sorted(df_class['equipe'].unique())
my_team = st.sidebar.selectbox("Mon Équipe", teams)
max_j = int(df_class['journee_team'].max()) if not df_class.empty else 1
cur_j = st.sidebar.slider("Journée", 1, max_j, max_j)

# SNAPSHOT CLASSEMENT
df_snap = df_class[df_class['journee_team'] <= cur_j].sort_values('match_timestamp').groupby('equipe').last().reset_index()
df_snap['rang'] = df_snap['total_points'].rank(ascending=False, method='min')
stats_team = df_snap[df_snap['equipe'] == my_team].iloc[0]

# --- DASHBOARD ---
st.title(f"📊 {my_team}")
c1, c2, c3 = st.columns(3)
c1.markdown(f'<div class="metric-card"><div class="metric-label">Classement</div><div class="metric-value">{int(stats_team["rang"])}e</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="metric-card"><div class="metric-label">Points</div><div class="metric-value">{int(stats_team["total_points"])}</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="metric-card"><div class="metric-label">Buts Pour</div><div class="metric-value">{int(stats_team["total_bp"])}</div></div>', unsafe_allow_html=True)

st.markdown("---")
st.subheader("⚔️ Simulation de Match")

opps = [t for t in teams if t != my_team]
c_sel, c_res = st.columns([1, 2])

with c_sel:
    opp = st.selectbox("Adversaire", opps)
    loc = st.radio("Lieu", ["Domicile", "Extérieur"])
    
    # SÉLECTEUR ARBITRE
    ref_list = sorted(df_referees['referee'].unique())
    ref_selected = st.selectbox("Arbitre (Optionnel)", ["Inconnu"] + ref_list)

th = my_team if loc == "Domicile" else opp
ta = opp if loc == "Domicile" else my_team

# Prédiction
xg_h, xg_a = predict_match_score(df_history, th, ta)

with c_res:
    if xg_h:
        s_h, s_a = int(round(xg_h)), int(round(xg_a))
        st.markdown(f"""
            <div class="score-card">
                <div style="color:#DAE025;">PRÉDICTION IA</div>
                <div class="score-display">{th} {s_h} - {s_a} {ta}</div>
                <div style="font-size:0.8rem; color:#AAA;">xG: {xg_h:.2f} - {xg_a:.2f}</div>
            </div>
        """, unsafe_allow_html=True)

# --- SECTION ARBITRAGE ---
if ref_selected != "Inconnu":
    st.markdown(f"#### 👮 Analyse Arbitre : {ref_selected}")
    
    ref_stats = analyze_referee(df_referees, ref_selected, th, ta)
    
    if ref_stats:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Expérience", f"{ref_stats['matches']} matchs")
        k2.metric("Sévérité (Jaunes/M)", f"{ref_stats['avg_yellow_global']:.2f}")
        k3.metric("Risque Rouge", ref_stats['risk'], f"{ref_stats['avg_red_global']:.2f}/m")
        
        # Comparaison Graphique
        fig = go.Figure(data=[
            go.Bar(name='Moyenne Globale', x=['Jaunes'], y=[ref_stats['avg_yellow_global']], marker_color='#95A5A6'),
            go.Bar(name=f'vs {th}', x=['Jaunes'], y=[ref_stats['avg_yellow_home']], marker_color='#2ECC71'),
            go.Bar(name=f'vs {ta}', x=['Jaunes'], y=[ref_stats['avg_yellow_away']], marker_color='#E74C3C')
        ])
        fig.update_layout(title="Distribution des Cartons Jaunes", barmode='group', 
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption(f"Note : {ref_selected} a arbitré {th} {ref_stats['nb_match_home']} fois et {ta} {ref_stats['nb_match_away']} fois dans l'historique.")
    else:
        st.warning("Pas assez de données pour cet arbitre.")
else:
    st.info("💡 Sélectionnez un arbitre pour voir ses statistiques et ses tendances avec les équipes.")

st.markdown("---")
# Graph Trajectoire
hist = df_class[df_class['equipe'] == my_team]
fig = px.line(hist, x='journee_team', y='total_points', title=f"Parcours {my_team}")
fig.update_traces(line_color='#DAE025')
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
st.plotly_chart(fig, use_container_width=True)