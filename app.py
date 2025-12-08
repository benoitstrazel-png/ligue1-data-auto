import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import poisson
import requests
import itertools

# --- CONFIGURATION ---
st.set_page_config(page_title="Ligue 1 Data Center", layout="wide", page_icon="⚽")

# --- STYLE CSS AVANCÉ ---
st.markdown("""
    <style>
    .stApp { background-color: #1A1C23; }
    
    /* Textes */
    h1, h2, h3, h4, h5, p, span, div, label, .stDataFrame { color: #FFFFFF !important; }
    
    /* Titre Principal */
    .main-title {
        font-size: 3rem; font-weight: 900; text-transform: uppercase;
        background: -webkit-linear-gradient(left, #DAE025, #FFFFFF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    
    /* Cartes Globales (Header) */
    .global-card {
        background-color: #091C3E; border-left: 4px solid #DAE025;
        padding: 10px; border-radius: 8px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .global-val { font-size: 1.5rem; font-weight: 800; color: #DAE025; margin: 0; }
    .global-lbl { font-size: 0.8rem; color: #E0E0E0; text-transform: uppercase; }
    
    /* Cartes KPI Club */
    .metric-card {
        background-color: #2C3E50; padding: 15px; border-radius: 12px;
        text-align: center; border: 1px solid #444; margin-bottom: 10px;
    }
    .metric-value { font-size: 1.8rem; font-weight: 800; margin: 0; color: #2ECC71; }
    .metric-label { font-size: 0.9rem; color: #CCC; }
    
    /* Score & Conseils */
    .score-card {
        background: linear-gradient(135deg, #091C3E 0%, #1A1C23 100%);
        border: 2px solid #DAE025; border-radius: 15px; padding: 20px;
        text-align: center; margin-top: 10px;
    }
    .score-display { font-size: 3rem; font-weight: bold; color: #FFFFFF; font-family: monospace; }
    
    .advice-box {
        background-color: rgba(46, 204, 113, 0.1); border-left: 5px solid #2ECC71;
        padding: 15px; margin-top: 10px; border-radius: 5px;
    }
    .insight-box {
        background-color: rgba(218, 224, 37, 0.1); border-left: 5px solid #DAE025;
        padding: 15px; margin-top: 10px; color: #FFFFFF; border-radius: 5px;
    }
    
    /* Pastilles Forme */
    .form-badge {
        display: inline-block; width: 32px; height: 32px; line-height: 32px;
        border-radius: 50%; text-align: center; font-weight: bold;
        color: white !important; margin-right: 4px; font-size: 0.8rem;
        border: 2px solid #1A1C23;
    }
    .win { background-color: #2ECC71; }
    .draw { background-color: #95A5A6; }
    .loss { background-color: #E74C3C; }
    
    /* Sidebar & Tableaux */
    [data-testid="stSidebar"] { background-color: #091C3E; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
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
def load_data_complete(season_name, history_seasons):
    client = get_db_client()
    p = client.project
    
    # 1. Saison Focus
    q_curr = f"SELECT * FROM `{p}.historic_datasets.matchs_clean` WHERE season = '{season_name}' ORDER BY date ASC"
    df_curr = client.query(q_curr).to_dataframe()
    
    # 2. Historique Multi-Saisons
    s_str = "', '".join(history_seasons)
    q_hist = f"SELECT * FROM `{p}.historic_datasets.matchs_clean` WHERE season IN ('{s_str}')"
    df_hist = client.query(q_hist).to_dataframe()
    
    # 3. Classement Live
    q_class = f"SELECT * FROM `{p}.historic_datasets.classement_live` WHERE saison = '{season_name}' ORDER BY journee_team ASC"
    df_class = client.query(q_class).to_dataframe()
    
    # 4. Historique Rank vs Rank (Alias 'saison' -> 'season')
    q_ranks = f"SELECT saison as season, journee_team, equipe as team, total_points FROM `{p}.historic_datasets.classement_live`"
    try: df_ranks_Raw = client.query(q_ranks).to_dataframe()
    except: df_ranks_Raw = pd.DataFrame()
    
    # Nettoyage Dates
    for df in [df_curr, df_hist, df_class]:
        if not df.empty:
            for col in ['date', 'match_timestamp']:
                if col in df.columns: df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
                
    return df_curr, df_hist, df_class, df_ranks_Raw

@st.cache_data(ttl=3600)
def get_live_schedule():
    try:
        url = "https://fixturedownload.com/feed/json/ligue-1-2024"
        df = pd.DataFrame(requests.get(url).json())
        df['DateUtc'] = pd.to_datetime(df['DateUtc']).dt.tz_localize(None)
        return df[df['DateUtc'] > pd.Timestamp.now()].sort_values('DateUtc')
    except: return pd.DataFrame()

# --- LOGIQUE MÉTIER ---

def process_rank_history(df_ranks, df_matchs):
    """Prépare la matrice Rank vs Rank"""
    if df_ranks.empty or df_matchs.empty: return pd.DataFrame()
    
    df_ranks['rank'] = df_ranks.groupby(['season', 'journee_team'])['total_points'].rank(ascending=False, method='min')
    
    df_m = df_matchs.copy()
    df_m['home_journee'] = df_m.groupby(['season', 'home_team']).cumcount() + 1
    df_m['away_journee'] = df_m.groupby(['season', 'away_team']).cumcount() + 1
    df_m['h_prev'] = df_m['home_journee'] - 1
    df_m['a_prev'] = df_m['away_journee'] - 1
    
    m = pd.merge(df_m, df_ranks, left_on=['season', 'home_team', 'h_prev'], right_on=['season', 'team', 'journee_team'], how='inner').rename(columns={'rank': 'home_rank'}).drop(columns=['team', 'journee_team', 'total_points'])
    m = pd.merge(m, df_ranks, left_on=['season', 'away_team', 'a_prev'], right_on=['season', 'team', 'journee_team'], how='inner').rename(columns={'rank': 'away_rank'}).drop(columns=['team', 'journee_team', 'total_points'])
    return m

def get_rank_stats(df_rank_hist, r_home, r_away, tol=3):
    if df_rank_hist.empty: return None
    mask = (df_rank_hist['home_rank'].between(r_home-tol, r_home+tol)) & (df_rank_hist['away_rank'].between(r_away-tol, r_away+tol))
    sub = df_rank_hist[mask]
    if len(sub) < 5: return None
    win = len(sub[sub['full_time_result']=='H'])
    return {'win_pct': (win/len(sub))*100, 'n': len(sub)}

def get_team_form_html(df_matchs, team, limit=5):
    matches = df_matchs[((df_matchs['home_team'] == team) | (df_matchs['away_team'] == team)) & (df_matchs['date'] < pd.Timestamp.now())].sort_values('date', ascending=False).head(limit)
    matches = matches.sort_values('date', ascending=True) # Chrono
    
    html = '<div style="display:flex; align-items:center;">'
    for _, r in matches.iterrows():
        is_h = r['home_team'] == team
        res, cls, flame = 'D', 'loss', ''
        diff = abs(r['full_time_home_goals'] - r['full_time_away_goals'])
        
        if r['full_time_result'] == 'D': res, cls = 'N', 'draw'
        elif (is_h and r['full_time_result']=='H') or (not is_h and r['full_time_result']=='A'):
            res, cls = 'V', 'win'
            if diff >= 3: flame = "🔥"
            
        opp = r['away_team'] if is_h else r['home_team']
        sc = f"{int(r['full_time_home_goals'])}-{int(r['full_time_away_goals'])}"
        html += f'<div class="form-badge {cls}" title="vs {opp} ({sc})">{res}{flame}</div>'
    html += '</div>'
    return html

def calculate_global_stats(df):
    if df.empty: return {}
    nb = len(df)
    g = df['full_time_home_goals'].sum() + df['full_time_away_goals'].sum()
    s = df['home_shots'].sum() + df['away_shots'].sum()
    t = df['home_shots_on_target'].sum() + df['away_shots_on_target'].sum()
    y = df['home_yellow_cards'].sum() + df['away_yellow_cards'].sum()
    return {
        'Buts Totaux': int(g), 'Buts / Match': round(g/nb, 2),
        'Tirs / Match': round(s/nb, 1), '% Cadrés': round((t/s*100), 1) if s>0 else 0,
        'Jaunes / M': round(y/nb, 2), 'Rouges / M': round((df['home_red_cards'].sum()+df['away_red_cards'].sum())/nb, 2)
    }

def calculate_advanced_stats_and_betting(df, team, stake=10):
    df_t = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
    if df_t.empty: return None, None
    
    # Stats
    shots = 0; target = 0; yel = 0; red = 0
    # Betting
    strats = {'Victoire':0, 'Nul':0, 'Défaite':0, 'Over 2.5':0, 'Under 2.5':0} # Profits
    invest = {k:0 for k in strats} # Investissement
    
    for _, r in df_t.iterrows():
        is_h = r['home_team'] == team
        shots += r['home_shots'] if is_h else r['away_shots']
        target += r['home_shots_on_target'] if is_h else r['away_shots_on_target']
        yel += r['home_yellow_cards'] if is_h else r['away_yellow_cards']
        red += r['home_red_cards'] if is_h else r['away_red_cards']
        
        # Bet calc
        res = r['full_time_result']
        goals = r['full_time_home_goals'] + r['full_time_away_goals']
        ho, do, ao = r.get('bet365_home_win_odds', 0), r.get('bet365_draw_odds', 0), r.get('bet365_away_win_odds', 0)
        o25, u25 = r.get('bet365_over_25_goals', 0), r.get('bet365_under_25_goals', 0)
        
        if pd.notna(ho) and ho:
            invest['Victoire'] += stake; invest['Nul'] += stake; invest['Défaite'] += stake
            # Win
            if (is_h and res=='H') or (not is_h and res=='A'): strats['Victoire'] += (ho if is_h else ao)*stake - stake
            else: strats['Victoire'] -= stake
            # Draw
            if res=='D': strats['Nul'] += (do*stake) - stake
            else: strats['Nul'] -= stake
            # Loss
            if (is_h and res=='A') or (not is_h and res=='H'): strats['Défaite'] += (ao if is_h else ho)*stake - stake
            else: strats['Défaite'] -= stake
            
        if pd.notna(o25) and o25:
            invest['Over 2.5'] += stake; invest['Under 2.5'] += stake
            if goals > 2.5: strats['Over 2.5'] += (o25*stake) - stake
            else: strats['Over 2.5'] -= stake
            if goals < 2.5: strats['Under 2.5'] += (u25*stake) - stake
            else: strats['Under 2.5'] -= stake
            
    nb = len(df_t)
    stats = {'avg_shots': shots/nb, 'avg_target': target/nb, 'avg_yellow': yel/nb, 'avg_red': red/nb}
    res_df = pd.DataFrame([
        {'Type': k, 'Profit': v, 'ROI': (v/invest[k]*100) if invest[k]>0 else 0} 
        for k, v in strats.items()
    ])
    return stats, res_df

def get_spider_data_normalized(df, team1, team2=None):
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    metrics = {
        'Buts': lambda x, h: x['full_time_home_goals'] if h else x['full_time_away_goals'],
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
    return list(metrics.keys()), v1, v2, df_s

def calculate_match_probabilities_detailed(att_h, def_h, att_a, def_a, avg_h, avg_a):
    mu_h = att_h * def_a * avg_h
    mu_a = att_a * def_h * avg_a
    max_g = 8
    ph = [poisson.pmf(i, mu_h) for i in range(max_g)]
    pa = [poisson.pmf(i, mu_a) for i in range(max_g)]
    
    win, draw, loss, u25 = 0, 0, 0, 0
    max_p = 0
    exact = (0, 0)
    
    for i in range(max_g):
        for j in range(max_g):
            p = ph[i] * pa[j]
            if i > j: win += p
            elif i == j: draw += p
            else: loss += p
            
            if i+j < 2.5: u25 += p
            if p > max_p:
                max_p = p
                exact = (i, j)
                
    return {'win': win*100, 'draw': draw*100, 'loss': loss*100, 'exact': exact, 'xg': (mu_h, mu_a), 'u25': u25*100, 'o25': (1-u25)*100}

def simulate_season_end(df_played, df_history, teams_list):
    played_pairs = set(zip(df_played['home_team'], df_played['away_team']))
    all_pairs = set(itertools.permutations(teams_list, 2))
    remaining = list(all_pairs - played_pairs)
    
    avg_h = df_history['full_time_home_goals'].mean()
    avg_a = df_history['full_time_away_goals'].mean()
    
    team_forces = {}
    for t in teams_list:
        h = df_history[df_history['home_team']==t]
        a = df_history[df_history['away_team']==t]
        if h.empty or a.empty: continue
        team_forces[t] = {
            'ah': h['full_time_home_goals'].mean()/avg_h, 'dh': h['full_time_away_goals'].mean()/avg_a,
            'aa': a['full_time_away_goals'].mean()/avg_a, 'da': a['full_time_home_goals'].mean()/avg_h
        }
    
    res = []
    for h, a in remaining:
        if h not in team_forces or a not in team_forces: continue
        th, ta = team_forces[h], team_forces[a]
        mh = th['ah'] * ta['da'] * avg_h
        ma = ta['aa'] * th['dh'] * avg_a
        
        # Logique simplifiée Points Espérés
        # On ne joue pas aux dés, on prend le résultat le plus probable
        ph = 3 if mh > ma + 0.2 else (1 if abs(mh-ma) <= 0.2 else 0)
        pa = 3 if ma > mh + 0.2 else (1 if abs(mh-ma) <= 0.2 else 0)
        
        res.append({'equipe': h, 'pts': ph})
        res.append({'equipe': a, 'pts': pa})
        
    return pd.DataFrame(res)

# --- NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Menu", ["Dashboard", "Classement Prédictif"])

st.sidebar.markdown("---")
st.sidebar.title("🔍 Filtres")
all_seasons = get_seasons_list()
selected_seasons = st.sidebar.multiselect("Historique (Poisson)", all_seasons, default=all_seasons[:3])
focus_season = sorted(selected_seasons, reverse=True)[0] if selected_seasons else all_seasons[0]

# Chargement
df_curr, df_hist, df_class, df_ranks_raw = load_data_complete(focus_season, selected_seasons)
teams = sorted(df_curr['home_team'].unique())

# Preparation Rank History Matrix
df_rank_matrix = process_rank_history(df_ranks_raw, df_hist)

# =================================================================================
# PAGE 1 : DASHBOARD
# =================================================================================
if page == "Dashboard":
    st.markdown('<div class="main-title">LIGUE 1 DATA CENTER</div>', unsafe_allow_html=True)
    
    # 1. GLOBAL SCORECARDS
    st.markdown(f"### 🌍 Statistiques {focus_season}")
    g_stats = calculate_global_stats(df_curr)
    if g_stats:
        cols = st.columns(6)
        for i, (k, v) in enumerate(g_stats.items()):
            cols[i].markdown(f"""
                <div class="global-card">
                    <p class="global-val">{v}</p>
                    <p class="global-lbl">{k}</p>
                </div>
            """, unsafe_allow_html=True)
    st.markdown("---")
    
    # 2. FOCUS HISTORIQUE
    st.subheader("📈 Focus Historique")
    c_met, c_gran = st.columns(2)
    with c_met: h_met = st.selectbox("Métrique", ["Buts", "Tirs", "Tirs Cadrés", "Fautes", "Cartons Jaunes"])
    with c_gran: h_gra = st.selectbox("Granularité", ["Saison", "Équipe"])
    
    # Préparation Graphique Historique
    map_met = {'Buts': ('full_time_home_goals', 'full_time_away_goals'), 'Tirs': ('home_shots', 'away_shots'), 'Tirs Cadrés': ('home_shots_on_target', 'away_shots_on_target'), 'Fautes': ('home_fouls', 'away_fouls'), 'Cartons Jaunes': ('home_yellow_cards', 'away_yellow_cards')}
    c1, c2 = map_met.get(h_met, ('full_time_home_goals', 'full_time_away_goals'))
    
    dh = df_hist[['season', 'home_team', c1]].rename(columns={'home_team':'team', c1:'val'})
    da = df_hist[['season', 'away_team', c2]].rename(columns={'away_team':'team', c2:'val'})
    df_chart = pd.concat([dh, da])
    
    if h_gra == 'Saison': df_chart = df_chart.groupby('season')['val'].mean().reset_index()
    else: df_chart = df_chart.groupby(['team', 'season'])['val'].mean().reset_index()
    
    fig_hist = px.bar(df_chart, x=('season' if h_gra=='Saison' else 'team'), y='val', color='season' if h_gra=='Équipe' else None, title=f"Moyenne {h_met}")
    fig_hist.update_layout(paper_bgcolor='rgba(255,255,255,0.1)', plot_bgcolor='rgba(255,255,255,0.1)', font=dict(color='white'))
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")

    # 3. ANALYSE CLUB
    st.subheader("🛡️ Analyse Club")
    my_team = st.sidebar.selectbox("Sélectionner un Club", teams)
    
    # Snapshot
    max_j = int(df_class['journee_team'].max()) if not df_class.empty else 1
    cur_j = st.sidebar.slider("Journée", 1, max_j, max_j)
    
    df_snap = df_class[df_class['journee_team'] <= cur_j].sort_values('match_timestamp').groupby('equipe').last().reset_index()
    df_snap['rang'] = df_snap['total_points'].rank(ascending=False, method='min')
    stats_team = df_snap[df_snap['equipe'] == my_team].iloc[0]
    
    # Header Team
    c_head, c_form = st.columns([2, 1])
    with c_head: st.markdown(f"## {my_team}")
    with c_form: 
        st.caption("Forme (5 derniers matchs)")
        st.markdown(get_team_form_html(df_curr, my_team), unsafe_allow_html=True)
        
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="metric-card"><div class="metric-label">Classement</div><div class="metric-value">{int(stats_team["rang"])}e</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="metric-card"><div class="metric-label">Points</div><div class="metric-value">{int(stats_team["total_points"])}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="metric-card"><div class="metric-label">Buts Pour</div><div class="metric-value">{int(stats_team["total_bp"])}</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="metric-card"><div class="metric-label">Diff.</div><div class="metric-value" style="color:white !important">{int(stats_team["total_diff"]):+d}</div></div>', unsafe_allow_html=True)
    
    # Stats Avancées & Paris
    adv_stats, bet_df = calculate_advanced_stats_and_betting(df_curr, my_team)
    
    if adv_stats:
        st.markdown("##### 📊 Moyennes par Match")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Tirs Totaux", f"{adv_stats['avg_shots']:.1f}")
        s2.metric("Tirs Cadrés", f"{adv_stats['avg_target']:.1f}")
        s3.metric("Cartons Jaunes", f"{adv_stats['avg_yellow']:.1f}")
        s4.metric("Cartons Rouges", f"{adv_stats['avg_red']:.2f}")
        
        c_bet, c_rad = st.columns([1, 1])
        with c_bet:
            st.markdown("##### 💰 Simulation ROI (Mise 10€)")
            colors = ['#2ECC71' if x > 0 else '#E74C3C' for x in bet_df['Profit']]
            bet_df['Txt'] = bet_df.apply(lambda x: f"{x['Profit']:+.0f}€", axis=1)
            fig_b = go.Figure(go.Bar(x=bet_df['Type'], y=bet_df['Profit'], text=bet_df['Txt'], marker_color=colors))
            fig_b.update_layout(paper_bgcolor='rgba(255,255,255,0.1)', plot_bgcolor='rgba(255,255,255,0.1)', font=dict(color='white'), height=300)
            st.plotly_chart(fig_b, use_container_width=True)
            
        with c_rad:
            st.markdown("##### 🕸️ Style vs Moyenne")
            cats, v1, v2, _ = get_spider_data_normalized(df_curr, my_team, None)
            if cats:
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatterpolar(r=v1, theta=cats, fill='toself', name=my_team, line_color='#DAE025'))
                fig_r.add_trace(go.Scatterpolar(r=v2, theta=cats, fill='toself', name='Moyenne', line_color='#95A5A6'))
                fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], color='white'), angularaxis=dict(color='white')), paper_bgcolor='rgba(255,255,255,0.1)', font=dict(color='white'), margin=dict(t=30, b=30), height=300)
                st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("---")
    
    # 4. SIMULATION
    st.subheader("⚔️ Analyse Adversaire")
    opp_sel = st.selectbox("Adversaire", ["Vue Globale"] + [t for t in teams if t != my_team])
    
    if opp_sel == "Vue Globale":
        # Tableau Probas Global
        rows = []
        avg_h, avg_a = df_hist['full_time_home_goals'].mean(), df_hist['full_time_away_goals'].mean()
        
        # Force MyTeam
        h = df_hist[df_hist['home_team']==my_team]
        a = df_hist[df_hist['away_team']==my_team]
        if not h.empty and not a.empty:
            ath = h['full_time_home_goals'].mean()/avg_h; dfh = h['full_time_away_goals'].mean()/avg_a
            ata = a['full_time_away_goals'].mean()/avg_a; dfa = a['full_time_home_goals'].mean()/avg_h
            
            for o in teams:
                if o == my_team: continue
                oh = df_hist[df_hist['home_team']==o]; oa = df_hist[df_hist['away_team']==o]
                if oh.empty or oa.empty: continue
                # Opp forces
                oah = oh['full_time_home_goals'].mean()/avg_h; odh = oh['full_time_away_goals'].mean()/avg_a
                
                # Sim MyTeam (Home) vs Opp (Away)
                r = calculate_match_probabilities_detailed(ath, dfh, ata, dfa, avg_h, avg_a) # Simplified for bulk
                rows.append({'Adversaire': o, 'Prob. Victoire': r['win'], 'Prob. Nul': r['draw'], 'Prob. Défaite': r['loss']})
                
            st.dataframe(pd.DataFrame(rows).set_index('Adversaire').style.format("{:.1f}%"), use_container_width=True)
            
    else:
        # DUEL MODE
        c_opt, c_res = st.columns([1, 2])
        with c_opt:
            loc = st.radio("Lieu", ["Domicile", "Extérieur"])
            is_home = "Domicile" in loc
            
        th, ta = (my_team, opp_sel) if is_home else (opp_sel, my_team)
        
        # Rank Stats
        rank_stats = None
        try:
            r_h = int(stats_team['rang']) if is_home else int(df_snap[df_snap['equipe']==opp_sel].iloc[0]['rang'])
            r_a = int(df_snap[df_snap['equipe']==opp_sel].iloc[0]['rang']) if is_home else int(stats_team['rang'])
            rank_stats = get_rank_stats(df_rank_matrix, r_h, r_a)
        except: pass
        
        # Poisson
        hm = df_hist[df_hist['home_team']==th]
        am = df_hist[df_hist['away_team']==ta]
        
        if not hm.empty and not am.empty:
            av_h, av_a = df_hist['full_time_home_goals'].mean(), df_hist['full_time_away_goals'].mean()
            ath, dfh = hm['full_time_home_goals'].mean()/av_h, hm['full_time_away_goals'].mean()/av_a
            ata, dfa = am['full_time_away_goals'].mean()/av_a, am['full_time_home_goals'].mean()/av_h
            
            res = calculate_match_probabilities_detailed(ath, dfh, ata, dfa, av_h, av_a)
            
            # Rank Adjust
            if rank_stats and rank_stats['win_pct'] > 60: 
                res['win'] = min(99, res['win']*1.15)
                res['loss'] = max(1, res['loss']*0.85)
            
            with c_res:
                s1, s2 = res['exact']
                st.markdown(f"""
                    <div class="score-card">
                        <div style="color:#DAE025;">SCORE PROBABLE</div>
                        <div class="score-display">{th} {s1}-{s2} {ta}</div>
                        <div style="font-size:0.8rem; color:#AAA;">xG: {res['xg'][0]:.2f} - {res['xg'][1]:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Conseil
                probs = {'Victoire': res['win'], 'Nul': res['draw'], 'Défaite': res['loss']}
                
                # Traduction pour MyTeam
                my_win_p = res['win'] if is_home else res['loss']
                my_loss_p = res['loss'] if is_home else res['win']
                
                best_opt = "Victoire" if my_win_p > max(res['draw'], my_loss_p) else ("Défaite" if my_loss_p > max(my_win_p, res['draw']) else "Nul")
                best_prob = max(my_win_p, my_loss_p, res['draw'])
                
                if best_opt == "Victoire": txt = f"Victoire de {my_team}"
                elif best_opt == "Défaite": txt = f"Victoire de {opp_sel}"
                else: txt = "Match Nul"
                
                col_adv = "#2ECC71" if best_prob > 45 else "#F1C40F"
                st.markdown(f"""
                    <div class="advice-box" style="border-color:{col_adv}">
                        💡 <b>Option 1N2 la plus sûre :</b> <br>
                        <span style="font-size:1.2rem; font-weight:bold; color:{col_adv}">{txt}</span> 
                        <span style="color:#EEE">({best_prob:.1f}%)</span>
                    </div>
                """, unsafe_allow_html=True)
                
                g_opt = "Plus de 2.5 buts" if res['o25'] > 50 else "Moins de 2.5 buts"
                st.markdown(f"""
                    <div class="advice-box" style="border-color:#3498DB">
                        ⚽ <b>Option Buts :</b> <br>
                        <span style="font-size:1.1rem; font-weight:bold; color:#3498DB">{g_opt}</span>
                        <span style="color:#EEE">({max(res['o25'], res['u25']):.1f}%)</span>
                    </div>
                """, unsafe_allow_html=True)
                
    st.markdown("---")
    st.subheader("🏆 Classement Live")
    
    # Styling Table
    def highlight_rows(val):
        if val <= 4: return 'background-color: rgba(46, 204, 113, 0.2)' # LDC
        if val >= 16: return 'background-color: rgba(231, 76, 60, 0.2)' # Releg
        return ''
        
    show_cols = ['rang', 'equipe', 'total_points', 'total_diff', 'total_V', 'total_N', 'total_D']
    st.dataframe(df_snap[show_cols].set_index('rang'), use_container_width=True)

# =================================================================================
# PAGE 2 : CLASSEMENT PRÉDICTIF
# =================================================================================
elif page == "Classement Prédictif":
    st.markdown('<div class="main-title">🔮 CLASSEMENT FINAL PROJETÉ</div>', unsafe_allow_html=True)
    st.info("Simulation de tous les matchs restants via Loi de Poisson (Basé sur l'historique sélectionné).")
    
    if st.button("Lancer la simulation"):
        with st.spinner("Calcul en cours..."):
            sim_pts = simulate_season_end(df_curr, df_hist, teams)
            if not sim_pts.empty:
                agg = sim_pts.groupby('equipe')['pts'].sum().reset_index()
                
                curr = df_class.sort_values('match_timestamp').groupby('equipe').last().reset_index()[['equipe', 'total_points']]
                final = pd.merge(curr, agg, on='equipe', how='left').fillna(0)
                final['Total Projeté'] = final['total_points'] + final['pts']
                final = final.sort_values('Total Projeté', ascending=False).reset_index(drop=True)
                final.index += 1
                
                st.dataframe(final.rename(columns={'total_points': 'Actuel', 'pts': 'Restant (Sim)'}), use_container_width=True, height=600)
            else:
                st.warning("Données insuffisantes ou saison terminée.")