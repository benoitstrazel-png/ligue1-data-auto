import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import poisson
import requests

# --- CONFIGURATION ---
st.set_page_config(page_title="Ligue 1 Data Center", layout="wide", page_icon="⚽")
st.markdown("""
    <style>
    .stApp { background-color: #1A1C23; }
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6, .main p, .main span, .main div, .main label { color: #FFFFFF !important; }
    .metric-card h3, .metric-card div, .metric-card span { color: #DAE025 !important; }
    .metric-card .metric-label { color: #E0E0E0 !important; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    .metric-card { background-color: #091C3E; padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #DAE025; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .metric-value { font-size: 2rem; font-weight: 800; margin: 0; color: #DAE025 !important; }
    .score-card { background: linear-gradient(135deg, #091C3E 0%, #1A1C23 100%); border: 2px solid #DAE025; border-radius: 15px; padding: 20px; text-align: center; margin-top: 20px; }
    .score-display { font-size: 3.5rem; font-weight: bold; color: #FFFFFF; font-family: monospace; }
    .insight-box { background-color: rgba(218, 224, 37, 0.1); border-left: 4px solid #DAE025; padding: 15px; margin-top: 15px; color: #FFFFFF; }
    .form-badge { display: inline-block; width: 30px; height: 30px; line-height: 30px; border-radius: 50%; text-align: center; font-weight: bold; color: white !important; margin-right: 5px; font-size: 0.8rem; }
    .win { background-color: #2ECC71; } .draw { background-color: #95A5A6; } .loss { background-color: #E74C3C; }
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
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

@st.cache_data(ttl=600)
def load_focus_season(season_name):
    client = get_db_client()
    q_class = f"SELECT * FROM `{client.project}.historic_datasets.classement_live` WHERE saison = '{season_name}' ORDER BY journee_team ASC"
    q_matchs = f"SELECT * FROM `{client.project}.historic_datasets.matchs_clean` WHERE season = '{season_name}' ORDER BY date ASC"
    df_c = client.query(q_class).to_dataframe()
    df_m = client.query(q_matchs).to_dataframe()
    if not df_c.empty: df_c['match_timestamp'] = pd.to_datetime(df_c['match_timestamp'], utc=True).dt.tz_localize(None)
    if not df_m.empty: df_m['date'] = pd.to_datetime(df_m['date'], utc=True).dt.tz_localize(None)
    return df_c, df_m

@st.cache_data(ttl=600)
def load_rank_history():
    client = get_db_client()
    q = f"SELECT saison as season, journee_team, equipe as team, total_points FROM `{client.project}.historic_datasets.classement_live`"
    try: df = client.query(q).to_dataframe()
    except: return pd.DataFrame()
    if df.empty: return pd.DataFrame()
    df['rank'] = df.groupby(['season', 'journee_team'])['total_points'].rank(ascending=False, method='min')
    
    q_m = f"SELECT season, home_team, away_team, full_time_result, date FROM `{client.project}.historic_datasets.matchs_clean` ORDER BY date"
    df_m = client.query(q_m).to_dataframe()
    
    df_m['home_journee'] = df_m.groupby(['season', 'home_team']).cumcount() + 1
    df_m['away_journee'] = df_m.groupby(['season', 'away_team']).cumcount() + 1
    df_m['h_prev'] = df_m['home_journee'] - 1
    df_m['a_prev'] = df_m['away_journee'] - 1
    
    m = pd.merge(df_m, df, left_on=['season', 'home_team', 'h_prev'], right_on=['season', 'team', 'journee_team'], how='inner').rename(columns={'rank': 'home_rank'}).drop(columns=['team', 'journee_team', 'total_points'])
    m = pd.merge(m, df, left_on=['season', 'away_team', 'a_prev'], right_on=['season', 'team', 'journee_team'], how='inner').rename(columns={'rank': 'away_rank'}).drop(columns=['team', 'journee_team', 'total_points'])
    return m

@st.cache_data(ttl=600)
def load_multi_season(seasons_list):
    client = get_db_client()
    s_str = "', '".join(seasons_list)
    q = f"SELECT * FROM `{client.project}.historic_datasets.matchs_clean` WHERE season IN ('{s_str}')"
    df = client.query(q).to_dataframe()
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

# --- LOGIQUE ---
def get_spider_data_normalized(df, team1, team2=None):
    """Calcule les stats normalisées (0-100) basées sur les Max de la ligue"""
    # 1. Calcul des moyennes pour TOUTES les équipes (pour trouver le Max)
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    metrics = {
        'Buts Pour': lambda x, is_home: x['full_time_home_goals'] if is_home else x['full_time_away_goals'],
        'Tirs Cadrés': lambda x, is_home: x['home_shots_on_target'] if is_home else x['away_shots_on_target'],
        'Corners': lambda x, is_home: x['home_corners'] if is_home else x['away_corners'],
        'Jaunes (Fairplay)': lambda x, is_home: x['home_yellow_cards'] if is_home else x['away_yellow_cards'],
        'Défense (Buts encaissés)': lambda x, is_home: x['full_time_away_goals'] if is_home else x['full_time_home_goals']
    }
    
    # Calcul des moyennes par équipe
    team_stats = {}
    for t in all_teams:
        sub = df[(df['home_team'] == t) | (df['away_team'] == t)]
        if sub.empty: continue
        
        t_vals = {}
        for m_name, func in metrics.items():
            vals = []
            for _, r in sub.iterrows():
                vals.append(func(r, r['home_team'] == t))
            t_vals[m_name] = np.nanmean(vals)
        team_stats[t] = t_vals
        
    if not team_stats: return None, None, None, None

    # Trouver le MAX de la ligue pour chaque métrique
    df_stats = pd.DataFrame(team_stats).T
    max_vals = df_stats.max()
    
    # Fonction de normalisation (Min-Max simplifiée : Val / Max * 100)
    def get_norm_vals(t_name):
        if t_name not in team_stats: return [0]*len(metrics)
        raw = team_stats[t_name]
        norm = []
        for k in metrics.keys():
            # Inversion pour Jaunes et Buts Encaissés (Moins c'est mieux)
            if k in ['Jaunes (Fairplay)', 'Défense (Buts encaissés)']:
                # Score = (1 - (val / max)) * 100 roughly
                v = 100 - (raw[k] / max_vals[k] * 100)
            else:
                v = (raw[k] / max_vals[k]) * 100
            norm.append(v)
        return norm

    vals_1 = get_norm_vals(team1)
    vals_2 = get_norm_vals(team2) if team2 else [df_stats[k].mean()/max_vals[k]*100 for k in metrics] # Moyenne si pas de team2
    
    return list(metrics.keys()), vals_1, vals_2, df_stats # df_stats for raw values if needed

# --- INTERFACE ---
st.sidebar.title("🔍 Filtres")
all_seasons = get_seasons_list()
selected_seasons = st.sidebar.multiselect("Historique", all_seasons, default=[all_seasons[0]])
focus_season = sorted(selected_seasons, reverse=True)[0]

df_class, df_matchs = load_focus_season(focus_season)
df_hist = load_multi_season(selected_seasons)
df_ranks = load_rank_history()

teams = sorted(df_class['equipe'].unique())
my_team = st.sidebar.selectbox("Mon Équipe", teams)
max_j = int(df_class['journee_team'].max()) if not df_class.empty else 1
cur_j = st.sidebar.slider("Journée", 1, max_j, max_j)

# Snapshot
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

# ================= DUEL RADAR & COMPARATEUR =================
st.subheader("🕸️ Comparateur de Style (Radar)")

col_radar, col_comp = st.columns([2, 1])

with col_comp:
    st.markdown("##### Comparer avec :")
    comp_team_list = ["Moyenne Ligue"] + [t for t in teams if t != my_team]
    comp_target = st.selectbox("Sélectionner un adversaire", comp_team_list)
    
    st.info("""
    **Légende du Radar :**
    - Échelle 0-100 normalisée.
    - **100** = Meilleure équipe de la ligue sur ce critère.
    - Pour *Défense* et *Jaunes*, 100 signifie "Encaisser peu" et "Peu de cartons".
    """)

with col_radar:
    # Calcul Données Normalisées
    target_team = None if comp_target == "Moyenne Ligue" else comp_target
    categories, val_1, val_2, _ = get_spider_data_normalized(df_matchs, my_team, target_team)
    
    if categories:
        fig = go.Figure()
        
        # Trace Equipe Principale
        fig.add_trace(go.Scatterpolar(
            r=val_1, theta=categories, fill='toself', 
            name=my_team, line_color='#DAE025', marker=dict(size=8)
        ))
        
        # Trace Comparaison
        color_2 = '#95A5A6' if comp_target == "Moyenne Ligue" else '#E74C3C' # Gris ou Rouge
        fig.add_trace(go.Scatterpolar(
            r=val_2, theta=categories, fill='toself', 
            name=comp_target, line_color=color_2, opacity=0.6
        ))
        
        # Layout propre et lisible
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color='white', gridcolor='#444'),
                angularaxis=dict(color='white')
            ),
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14), # Police plus grande
            margin=dict(t=30, b=30, l=40, r=40),
            legend=dict(
                orientation="h", y=-0.1, 
                font=dict(color="white", size=14), # Légende bien visible
                bgcolor="rgba(0,0,0,0)"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ================= SIMULATION MATCH =================
st.subheader("⚔️ Simulation de Match")
c_sel, c_res = st.columns([1, 2])
with c_sel:
    opp = st.selectbox("Adversaire", [t for t in teams if t != my_team], key="sim_opp")
    loc = st.radio("Lieu", ["Domicile", "Extérieur"])
    is_home = "Domicile" in loc

# Calculs
try:
    opp_s = df_snap[df_snap['equipe'] == opp].iloc[0]
    r_h = int(stats_team['rang']) if is_home else int(opp_s['rang'])
    r_a = int(opp_s['rang']) if is_home else int(stats_team['rang'])
    
    # Rank Stats
    mask = (df_ranks['home_rank'].between(r_h-3, r_h+3)) & (df_ranks['away_rank'].between(r_a-3, r_a+3))
    sub = df_ranks[mask]
    rank_stats = None
    if len(sub) > 10:
        w = len(sub[sub['full_time_result']=='H'])
        rank_stats = {'win_pct': (w/len(sub))*100, 'n': len(sub)}
except: rank_stats = None

# Poisson
th, ta = (my_team, opp) if is_home else (opp, my_team)
hm = df_history[df_history['home_team']==th]
am = df_history[df_history['away_team']==ta]
xg_h, xg_a = None, None
if not hm.empty and not am.empty:
    av_h = df_history['full_time_home_goals'].mean()
    av_a = df_history['full_time_away_goals'].mean()
    att_h, def_h = hm['full_time_home_goals'].mean()/av_h, hm['full_time_away_goals'].mean()/av_a
    att_a, def_a = am['full_time_away_goals'].mean()/av_a, am['full_time_home_goals'].mean()/av_h
    xg_h, xg_a = att_h*def_a*av_h, att_a*def_h*av_a
    
    # Ajustement Rank
    if rank_stats and rank_stats['win_pct'] > 60: xg_h *= 1.15
    elif rank_stats and rank_stats['win_pct'] < 20: xg_a *= 1.15

with c_res:
    if xg_h:
        sh, sa = int(round(xg_h)), int(round(xg_a))
        st.markdown(f"""
            <div class="score-card">
                <div style="color:#DAE025;">PRÉDICTION IA</div>
                <div class="score-display">{th} {sh} - {sa} {ta}</div>
                <div style="font-size:0.8rem; color:#AAA;">xG: {xg_h:.2f} - {xg_a:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        if rank_stats:
            st.markdown(f"""<div class="insight-box">📊 <b>Stat Choc :</b> Quand le {r_h}e reçoit le {r_a}e, il gagne dans <b>{rank_stats['win_pct']:.0f}%</b> des cas.</div>""", unsafe_allow_html=True)

st.markdown("---")
# ================= CALENDRIER =================
NAME_MAPPING = { "Paris Saint Germain": "Paris SG", "Olympique de Marseille": "Marseille", "Olympique Lyonnais": "Lyon", "AS Monaco": "Monaco", "Lille OSC": "Lille", "Stade Rennais": "Rennes", "OGC Nice": "Nice", "RC Lens": "Lens", "Stade de Reims": "Reims", "Strasbourg Alsace": "Strasbourg", "Montpellier HSC": "Montpellier", "FC Nantes": "Nantes", "Toulouse FC": "Toulouse", "Stade Brestois 29": "Brest", "FC Lorient": "Lorient", "Clermont Foot": "Clermont", "Le Havre AC": "Le Havre", "FC Metz": "Metz", "AJ Auxerre": "Auxerre", "Angers SCO": "Angers", "AS Saint-Etienne": "Saint Etienne" }

cutoff = pd.to_datetime(stats_team['match_timestamp']).replace(tzinfo=None)
replay = cutoff < (pd.Timestamp.now() - pd.Timedelta(days=7))
nxt = pd.DataFrame()

if replay:
    fut = df_matchs[df_matchs['date'] > cutoff].sort_values('date')
    if not fut.empty:
        nxt = fut[fut['date'] <= (fut.iloc[0]['date'] + pd.Timedelta(days=5))].rename(columns={'date':'DateUtc', 'home_team':'HomeTeam', 'away_team':'AwayTeam'})
else:
    api = get_live_schedule()
    if not api.empty:
        nxt = api[api['DateUtc'] <= (api.iloc[0]['DateUtc'] + pd.Timedelta(days=5))]
        nxt['HomeTeam'] = nxt['HomeTeam'].apply(lambda x: NAME_MAPPING.get(x, x))
        nxt['AwayTeam'] = nxt['AwayTeam'].apply(lambda x: NAME_MAPPING.get(x, x))

st.subheader("🔮 Calendrier")
if not nxt.empty:
    preds = []
    for _, m in nxt.iterrows():
        d, e = m['HomeTeam'], m['AwayTeam']
        # Simple Poisson Recalculé pour le tableau
        h_m = df_history[df_history['home_team']==d]
        a_m = df_history[df_history['away_team']==e]
        sc = "N/A"
        if not h_m.empty and not a_m.empty:
            ah, dh = h_m['full_time_home_goals'].mean(), h_m['full_time_away_goals'].mean()
            aa, da = a_m['full_time_away_goals'].mean(), a_m['full_time_home_goals'].mean()
            # On simplifie sans moyenne ligue pour aller vite dans la boucle
            sc = f"{int(round(ah*da))} - {int(round(aa*dh))}"
        preds.append({"Date": m['DateUtc'].strftime('%d/%m %H:%M'), "Dom": d, "Score": sc, "Ext": e})
    st.dataframe(pd.DataFrame(preds).set_index("Date"), use_container_width=True)
else: st.info("Pas de matchs proches.")

st.markdown("---")
# GRAPH
hist = df_class[df_class['equipe'] == my_team]
fig = px.line(hist, x='journee_team', y='total_points', title=f"Trajectoire {my_team}", markers=True)
fig.update_traces(line_color='#DAE025', line_width=4)
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
st.plotly_chart(fig, use_container_width=True)