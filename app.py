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
    /* Texte général en blanc */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6, .main p, .main span, .main div, .main label, .stDataFrame {
        color: #FFFFFF !important; 
    }
    /* Exceptions pour les cartes KPIs */
    .metric-card h3, .metric-card div, .metric-card span { color: #DAE025 !important; }
    .metric-card .metric-label { color: #E0E0E0 !important; }
    
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    
    .metric-card {
        background-color: #091C3E;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #DAE025;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    .metric-value { font-size: 1.8rem; font-weight: 800; margin: 0; color: #DAE025 !important; }
    .metric-label { font-size: 0.9rem; color: #CCC !important; }
    
    .score-card {
        background: linear-gradient(135deg, #091C3E 0%, #1A1C23 100%);
        border: 2px solid #DAE025;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }
    .score-display { font-size: 3.5rem; font-weight: bold; color: #FFFFFF; font-family: monospace; }
    
    .betting-box {
        background-color: #2C3E50;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2ECC71;
    }
    
    /* Tableaux */
    [data-testid="stDataFrame"] { background-color: #2C3E50; }
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
def load_multi_season_stats(seasons_list):
    client = get_db_client()
    seasons_str = "', '".join(seasons_list)
    query = f"SELECT * FROM `{client.project}.historic_datasets.matchs_clean` WHERE season IN ('{seasons_str}')"
    df = client.query(query).to_dataframe()
    if not df.empty: df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    return df

@st.cache_data(ttl=600)
def load_rank_history():
    client = get_db_client()
    # Alias SQL pour éviter l'erreur de colonne
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

@st.cache_data(ttl=3600)
def get_live_schedule():
    try:
        url = "https://fixturedownload.com/feed/json/ligue-1-2024" 
        df = pd.DataFrame(requests.get(url).json())
        df['DateUtc'] = pd.to_datetime(df['DateUtc']).dt.tz_localize(None)
        return df[df['DateUtc'] > pd.Timestamp.now()].sort_values('DateUtc')
    except: return pd.DataFrame()

# --- LOGIQUE MÉTIER ---

def calculate_advanced_stats_and_betting(df, team, stake=10):
    """Calcul Stats Avancées + Simulation Paris"""
    df_team = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
    if df_team.empty: return None, None

    # Indicateurs Jeu
    total_shots = 0; total_target = 0; total_yellow = 0; total_red = 0
    
    # Indicateurs Paris (ROI)
    # Structure : {Nom Stratégie: {Profit: X, Investi: Y}}
    strats = {
        'Victoire': {'p': 0, 'i': 0}, 'Nul': {'p': 0, 'i': 0}, 'Défaite': {'p': 0, 'i': 0},
        'Over 2.5': {'p': 0, 'i': 0}, 'Under 2.5': {'p': 0, 'i': 0}
    }

    for _, row in df_team.iterrows():
        is_home = row['home_team'] == team
        
        # Stats Jeu
        if is_home:
            total_shots += row.get('home_shots', 0)
            total_target += row.get('home_shots_on_target', 0)
            total_yellow += row.get('home_yellow_cards', 0)
            total_red += row.get('home_red_cards', 0)
        else:
            total_shots += row.get('away_shots', 0)
            total_target += row.get('away_shots_on_target', 0)
            total_yellow += row.get('away_yellow_cards', 0)
            total_red += row.get('away_red_cards', 0)

        # Betting
        res = row['full_time_result']
        goals = row['full_time_home_goals'] + row['full_time_away_goals']
        
        # Cotes (Gestion des NaN)
        h_odd = row.get('bet365_home_win_odds', 0) if pd.notna(row.get('bet365_home_win_odds')) else 0
        d_odd = row.get('bet365_draw_odds', 0) if pd.notna(row.get('bet365_draw_odds')) else 0
        a_odd = row.get('bet365_away_win_odds', 0) if pd.notna(row.get('bet365_away_win_odds')) else 0
        o25 = row.get('bet365_over_25_goals', 0) if pd.notna(row.get('bet365_over_25_goals')) else 0
        u25 = row.get('bet365_under_25_goals', 0) if pd.notna(row.get('bet365_under_25_goals')) else 0

        # Stratégie Victoire
        if h_odd and a_odd:
            strats['Victoire']['i'] += stake
            w_odd = h_odd if is_home else a_odd
            if (is_home and res == 'H') or (not is_home and res == 'A'): strats['Victoire']['p'] += (w_odd * stake) - stake
            else: strats['Victoire']['p'] -= stake
            
            # Stratégie Défaite (Miser contre son équipe)
            strats['Défaite']['i'] += stake
            l_odd = a_odd if is_home else h_odd
            if (is_home and res == 'A') or (not is_home and res == 'H'): strats['Défaite']['p'] += (l_odd * stake) - stake
            else: strats['Défaite']['p'] -= stake
            
            # Stratégie Nul
            strats['Nul']['i'] += stake
            if res == 'D': strats['Nul']['p'] += (d_odd * stake) - stake
            else: strats['Nul']['p'] -= stake

        # Stratégies Buts
        if o25 and u25:
            strats['Over 2.5']['i'] += stake
            if goals > 2.5: strats['Over 2.5']['p'] += (o25 * stake) - stake
            else: strats['Over 2.5']['p'] -= stake
            
            strats['Under 2.5']['i'] += stake
            if goals < 2.5: strats['Under 2.5']['p'] += (u25 * stake) - stake
            else: strats['Under 2.5']['p'] -= stake

    nb = len(df_team)
    stats = {
        'avg_shots': total_shots/nb, 'avg_target': total_target/nb,
        'avg_yellow': total_yellow/nb, 'avg_red': total_red/nb
    }
    
    # Calcul ROI Final
    betting_res = []
    for name, data in strats.items():
        roi = (data['p'] / data['i'] * 100) if data['i'] > 0 else 0
        betting_res.append({'Type': name, 'ROI': roi, 'Profit': data['p']})
        
    return stats, pd.DataFrame(betting_res)

def get_opponent_history_table(df, team):
    """Génère le tableau détaillé des adversaires"""
    df_t = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('date', ascending=False)
    data = []
    for _, r in df_t.iterrows():
        is_home = r['home_team'] == team
        opp = r['away_team'] if is_home else r['home_team']
        venue = "Domicile" if is_home else "Extérieur"
        score = f"{int(r['full_time_home_goals'])}-{int(r['full_time_away_goals'])}"
        
        res = "Nul"
        if r['full_time_result'] == 'D': res = "Nul"
        elif (is_home and r['full_time_result'] == 'H') or (not is_home and r['full_time_result'] == 'A'): res = "Victoire"
        else: res = "Défaite"
        
        data.append({
            "Date": r['date'].strftime('%d/%m/%Y'),
            "Adversaire": opp,
            "Lieu": venue,
            "Score": score,
            "Résultat": res
        })
    return pd.DataFrame(data)

def get_spider_data_normalized(df, team1, team2=None):
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    metrics = {
        'Buts Pour': lambda x, h: x['full_time_home_goals'] if h else x['full_time_away_goals'],
        'Tirs Cadrés': lambda x, h: x['home_shots_on_target'] if h else x['away_shots_on_target'],
        'Corners': lambda x, h: x['home_corners'] if h else x['away_corners'],
        'Fairplay (Inv)': lambda x, h: x['home_yellow_cards'] if h else x['away_yellow_cards'],
        'Défense (Inv)': lambda x, h: x['full_time_away_goals'] if h else x['full_time_home_goals']
    }
    
    team_stats = {}
    for t in all_teams:
        sub = df[(df['home_team'] == t) | (df['away_team'] == t)]
        if sub.empty: continue
        vals = {k: [] for k in metrics}
        for _, r in sub.iterrows():
            is_h = r['home_team'] == t
            for k, f in metrics.items(): vals[k].append(f(r, is_h))
        team_stats[t] = {k: np.nanmean(v) for k, v in vals.items()}
        
    if not team_stats: return None, None, None, None

    df_s = pd.DataFrame(team_stats).T
    max_v = df_s.max()
    
    def get_norm(t_name):
        if t_name not in team_stats: return [0]*len(metrics)
        raw = team_stats[t_name]
        norm = []
        for k in metrics:
            if 'Inv' in k: v = 100 - (raw[k]/max_v[k]*100)
            else: v = (raw[k]/max_v[k])*100
            norm.append(v)
        return norm

    v1 = get_norm(team1)
    v2 = get_norm(team2) if team2 else [df_s[k].mean()/max_v[k]*100 if 'Inv' not in k else 100-(df_s[k].mean()/max_v[k]*100) for k in metrics]
    
    labels = [k.replace(' (Inv)', '') for k in metrics]
    return labels, v1, v2, df_s

# --- INTERFACE ---
st.sidebar.title("🔍 Filtres")
all_seasons = get_seasons_list()
selected_seasons = st.sidebar.multiselect("Historique", all_seasons, default=[all_seasons[0]])
focus_season = sorted(selected_seasons, reverse=True)[0]

df_class, df_matchs = load_focus_season(focus_season)
df_history = load_multi_season_stats(selected_seasons)
df_ranks = load_rank_history()

teams = sorted(df_class['equipe'].unique())
my_team = st.sidebar.selectbox("Mon Équipe", teams)
max_j = int(df_class['journee_team'].max()) if not df_class.empty else 1
cur_j = st.sidebar.slider("Journée", 1, max_j, max_j)

df_snap = df_class[df_class['journee_team'] <= cur_j].sort_values('match_timestamp').groupby('equipe').last().reset_index()
df_snap['rang'] = df_snap['total_points'].rank(ascending=False, method='min')
stats_team = df_snap[df_snap['equipe'] == my_team].iloc[0]

# --- DASHBOARD HEADER ---
st.title(f"📊 {my_team}")
c1, c2, c3 = st.columns(3)
c1.markdown(f'<div class="metric-card"><div class="metric-label">Classement</div><div class="metric-value">{int(stats_team["rang"])}e</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="metric-card"><div class="metric-label">Points</div><div class="metric-value">{int(stats_team["total_points"])}</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="metric-card"><div class="metric-label">Buts Pour</div><div class="metric-value">{int(stats_team["total_bp"])}</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ================= STATS & BETTING =================
st.subheader("📈 Stats Jeu & Paris Sportifs")

# Inputs Paris
stake = st.number_input("Mise par match (€)", min_value=1, value=10, step=5)

# Calculs
game_stats, bet_df = calculate_advanced_stats_and_betting(df_matchs, my_team, stake)

if game_stats:
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="metric-card"><div class="metric-label">Tirs / Match</div><div class="metric-value">{game_stats["avg_shots"]:.1f}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="metric-card"><div class="metric-label">Tirs Cadrés / M</div><div class="metric-value">{game_stats["avg_target"]:.1f}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="metric-card"><div class="metric-label">Cartons Jaunes / M</div><div class="metric-value">{game_stats["avg_yellow"]:.1f}</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="metric-card"><div class="metric-label">Cartons Rouges / M</div><div class="metric-value">{game_stats["avg_red"]:.2f}</div></div>', unsafe_allow_html=True)

col_bet, col_radar = st.columns([1, 1])

with col_bet:
    st.markdown(f"##### 💰 Rentabilité Paris (Mise {stake}€)")
    if bet_df is not None:
        # Graphique ROI Barres
        colors = ['#2ECC71' if x > 0 else '#E74C3C' for x in bet_df['Profit']]
        fig_bet = go.Figure(go.Bar(
            x=bet_df['Type'], y=bet_df['Profit'],
            text=bet_df['ROI'].apply(lambda x: f"{x:+.1f}%"),
            marker_color=colors
        ))
        # Police NOIRE et fond BLANC comme demandé
        fig_bet.update_layout(
            title="Profit Net (Pertes/Gains)",
            font=dict(color='black'), # Police noire
            paper_bgcolor='white',    # Fond blanc
            plot_bgcolor='white',
            height=300
        )
        st.plotly_chart(fig_bet, use_container_width=True)
        st.caption("Simulation basée sur les cotes historiques Bet365 de la saison.")

with col_radar:
    st.markdown("##### 🕸️ Style de Jeu")
    cats, v1, v2, _ = get_spider_data_normalized(df_matchs, my_team, None) # None = Moyenne Ligue
    if cats:
        fig_rad = go.Figure()
        fig_rad.add_trace(go.Scatterpolar(r=v1, theta=cats, fill='toself', name=my_team, line_color='#DAE025'))
        fig_rad.add_trace(go.Scatterpolar(r=v2, theta=cats, fill='toself', name='Moyenne', line_color='#95A5A6'))
        # Police NOIRE et fond BLANC comme demandé
        fig_rad.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], color='black'), angularaxis=dict(color='black')),
            font=dict(color='black'),
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(t=30, b=30, l=40, r=40),
            legend=dict(orientation="h", y=-0.1)
        )
        st.plotly_chart(fig_rad, use_container_width=True)

st.markdown("---")

# ================= SIMULATION OU VUE GLOBALE =================
st.subheader("⚔️ Analyse Adversaire")

# Menu intelligent : "Vue d'ensemble" ou Liste des équipes
opp_list = ["Vue d'ensemble (Tous)"] + [t for t in teams if t != my_team]
opp_selection = st.selectbox("Adversaire", opp_list)

if opp_selection == "Vue d'ensemble (Tous)":
    # --- VUE TABLEAU GLOBAL ---
    st.markdown(f"##### 📜 Historique complet - {my_team}")
    df_all_opp = get_opponent_history_table(df_matchs, my_team)
    if not df_all_opp.empty:
        st.dataframe(df_all_opp, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun match joué cette saison.")

else:
    # --- VUE DUEL / PREDICTION ---
    opp = opp_selection
    c_opt, c_res = st.columns([1, 2])
    with c_opt:
        loc = st.radio("Lieu", ["Domicile", "Extérieur"])
        is_home = "Domicile" in loc
    
    # Calculs Prédictions (Existants)
    try:
        opp_s = df_snap[df_snap['equipe'] == opp].iloc[0]
        r_h = int(stats_team['rang']) if is_home else int(opp_s['rang'])
        r_a = int(opp_s['rang']) if is_home else int(stats_team['rang'])
        
        mask = (df_ranks['home_rank'].between(r_h-3, r_h+3)) & (df_ranks['away_rank'].between(r_a-3, r_a+3))
        sub = df_ranks[mask]
        rank_stats = None
        if len(sub) > 10:
            w = len(sub[sub['full_time_result']=='H'])
            rank_stats = {'win_pct': (w/len(sub))*100, 'n': len(sub)}
    except: rank_stats = None

    # Poisson & Predict
    th, ta = (my_team, opp) if is_home else (opp, my_team)
    # RAPPEL FIX : on utilise df_history (multi-saison)
    hm = df_history[df_history['home_team']==th]
    am = df_history[df_history['away_team']==ta]
    xg_h, xg_a = None, None

    if not hm.empty and not am.empty:
        av_h = df_history['full_time_home_goals'].mean()
        av_a = df_history['full_time_away_goals'].mean()
        att_h, def_h = hm['full_time_home_goals'].mean()/av_h, hm['full_time_away_goals'].mean()/av_a
        att_a, def_a = am['full_time_away_goals'].mean()/av_a, am['full_time_home_goals'].mean()/av_h
        xg_h, xg_a = att_h*def_a*av_h, att_a*def_h*av_a
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
        h_m = df_history[df_history['home_team']==d]
        a_m = df_history[df_history['away_team']==e]
        sc = "N/A"
        if not h_m.empty and not a_m.empty:
            ah, dh = h_m['full_time_home_goals'].mean(), h_m['full_time_away_goals'].mean()
            aa, da = a_m['full_time_away_goals'].mean(), a_m['full_time_home_goals'].mean()
            sc = f"{int(round(ah*da))} - {int(round(aa*dh))}"
        preds.append({"Date": m['DateUtc'].strftime('%d/%m %H:%M'), "Dom": d, "Score": sc, "Ext": e})
    st.dataframe(pd.DataFrame(preds).set_index("Date"), use_container_width=True)
else: st.info("Pas de matchs proches.")

st.markdown("---")
# GRAPH
hist = df_class[df_class['equipe'] == my_team]
fig = px.line(hist, x='journee_team', y='total_points', title=f"Trajectoire {my_team}", markers=True)
fig.update_traces(line_color='#DAE025', line_width=4)
# Police NOIRE et fond BLANC comme demandé
fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'))
st.plotly_chart(fig, use_container_width=True)