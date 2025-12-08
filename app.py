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
    /* Exceptions KPIs */
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
    .score-display { font-size: 3.5rem; font-weight: bold; color: #FFFFFF; font-family: 'Courier New', monospace; }
    
    .advice-box {
        background-color: rgba(46, 204, 113, 0.1);
        border-left: 5px solid #2ECC71;
        padding: 15px;
        margin-top: 10px;
        border-radius: 5px;
    }
    
    /* Pastilles Forme */
    .form-badge {
        display: inline-block; width: 35px; height: 35px; line-height: 35px;
        border-radius: 50%; text-align: center; font-weight: bold;
        color: white !important; margin-right: 5px; font-size: 0.9rem;
        border: 2px solid #1A1C23;
    }
    .win { background-color: #2ECC71; }
    .draw { background-color: #95A5A6; }
    .loss { background-color: #E74C3C; }
    
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

def get_team_form_html(df_matchs, team, limit=5):
    """Génère les pastilles de forme avec emoji Flamme si gros écart"""
    # Matchs passés
    matches = df_matchs[
        ((df_matchs['home_team'] == team) | (df_matchs['away_team'] == team)) & 
        (df_matchs['date'] < pd.Timestamp.now())
    ].sort_values('date', ascending=False).head(limit)
    
    # On remet dans l'ordre chronologique (gauche = plus vieux, droite = récent)
    matches = matches.sort_values('date', ascending=True)
    
    html = '<div style="display:flex; align-items:center; gap:5px;">'
    for _, r in matches.iterrows():
        is_home = r['home_team'] == team
        res = 'D'
        cls = 'loss'
        
        diff = abs(r['full_time_home_goals'] - r['full_time_away_goals'])
        flame = "🔥" if diff >= 3 else ""
        
        if r['full_time_result'] == 'D':
            res = 'N'
            cls = 'draw'
            flame = "" # Pas de flamme sur un nul généralement
        elif (is_home and r['full_time_result'] == 'H') or (not is_home and r['full_time_result'] == 'A'):
            res = 'V'
            cls = 'win'
        
        tooltip = f"vs {r['away_team'] if is_home else r['home_team']} ({int(r['full_time_home_goals'])}-{int(r['full_time_away_goals'])})"
        html += f'<div class="form-badge {cls}" title="{tooltip}">{res}{flame}</div>'
    html += '</div>'
    return html

def calculate_advanced_stats_and_betting(df, team, stake=10):
    df_team = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
    if df_team.empty: return None, None

    total_shots = 0; total_target = 0; total_yellow = 0; total_red = 0
    strats = {'Victoire': {'p': 0, 'i': 0}, 'Nul': {'p': 0, 'i': 0}, 'Défaite': {'p': 0, 'i': 0}}

    for _, row in df_team.iterrows():
        is_home = row['home_team'] == team
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

        res = row['full_time_result']
        h_odd = row.get('bet365_home_win_odds', 0) if pd.notna(row.get('bet365_home_win_odds')) else 0
        d_odd = row.get('bet365_draw_odds', 0) if pd.notna(row.get('bet365_draw_odds')) else 0
        a_odd = row.get('bet365_away_win_odds', 0) if pd.notna(row.get('bet365_away_win_odds')) else 0

        if h_odd and a_odd:
            strats['Victoire']['i'] += stake
            w_odd = h_odd if is_home else a_odd
            if (is_home and res == 'H') or (not is_home and res == 'A'): strats['Victoire']['p'] += (w_odd * stake) - stake
            else: strats['Victoire']['p'] -= stake
            
            strats['Défaite']['i'] += stake
            l_odd = a_odd if is_home else h_odd
            if (is_home and res == 'A') or (not is_home and res == 'H'): strats['Défaite']['p'] += (l_odd * stake) - stake
            else: strats['Défaite']['p'] -= stake
            
            strats['Nul']['i'] += stake
            if res == 'D': strats['Nul']['p'] += (d_odd * stake) - stake
            else: strats['Nul']['p'] -= stake

    nb = len(df_team)
    stats = {'avg_shots': total_shots/nb, 'avg_target': total_target/nb, 'avg_yellow': total_yellow/nb, 'avg_red': total_red/nb}
    betting_res = [{'Type': k, 'ROI': (v['p']/v['i']*100) if v['i']>0 else 0, 'Profit': v['p']} for k,v in strats.items()]
    return stats, pd.DataFrame(betting_res)

def calculate_match_probabilities_detailed(att_h, def_h, att_a, def_a, avg_h, avg_a):
    """
    Calcule les probabilités V/N/D ET le score le plus probable (Mode)
    ET la probabilité Over/Under 2.5
    """
    mu_h = att_h * def_a * avg_h
    mu_a = att_a * def_h * avg_a
    
    max_goals = 10
    prob_h = [poisson.pmf(i, mu_h) for i in range(max_goals)]
    prob_a = [poisson.pmf(i, mu_a) for i in range(max_goals)]
    
    win, draw, loss = 0, 0, 0
    prob_under_2_5 = 0
    
    # Matrice des scores pour trouver le score exact le plus probable
    max_prob_score = 0
    most_likely_score = (0, 0)
    
    for i in range(max_goals):
        for j in range(max_goals):
            p = prob_h[i] * prob_a[j]
            
            # 1N2
            if i > j: win += p
            elif i == j: draw += p
            else: loss += p
            
            # Score Exact
            if p > max_prob_score:
                max_prob_score = p
                most_likely_score = (i, j)
                
            # Over/Under
            if (i + j) < 2.5:
                prob_under_2_5 += p
                
    prob_over_2_5 = 1 - prob_under_2_5
    
    return {
        'win': win*100, 'draw': draw*100, 'loss': loss*100, 
        'exact_score': most_likely_score,
        'xg_h': mu_h, 'xg_a': mu_a,
        'over_2_5': prob_over_2_5 * 100, 'under_2_5': prob_under_2_5 * 100
    }

def get_probabilities_table(df_history, my_team, teams_list, mode="Global"):
    if df_history.empty: return pd.DataFrame()
    avg_h = df_history['full_time_home_goals'].mean()
    avg_a = df_history['full_time_away_goals'].mean()
    my_home = df_history[df_history['home_team'] == my_team]
    my_away = df_history[df_history['away_team'] == my_team]
    if my_home.empty or my_away.empty: return pd.DataFrame()
    
    att_h_my = my_home['full_time_home_goals'].mean() / avg_h
    def_h_my = my_home['full_time_away_goals'].mean() / avg_a
    att_a_my = my_away['full_time_away_goals'].mean() / avg_a
    def_a_my = my_away['full_time_home_goals'].mean() / avg_h
    
    rows = []
    for opp in teams_list:
        if opp == my_team: continue
        opp_home = df_history[df_history['home_team'] == opp]
        opp_away = df_history[df_history['away_team'] == opp]
        if opp_home.empty or opp_away.empty: continue
        
        att_h_opp = opp_home['full_time_home_goals'].mean() / avg_h
        def_h_opp = opp_home['full_time_away_goals'].mean() / avg_a
        att_a_opp = opp_away['full_time_away_goals'].mean() / avg_a
        def_a_opp = opp_away['full_time_home_goals'].mean() / avg_h
        
        res_h = calculate_match_probabilities_detailed(att_h_my, def_h_my, att_a_opp, def_a_opp, avg_h, avg_a)
        res_a = calculate_match_probabilities_detailed(att_h_opp, def_h_opp, att_a_my, def_a_my, avg_h, avg_a)
        
        if mode == "Domicile":
            rows.append({'Adversaire': opp, 'Victoire': res_h['win'], 'Nul': res_h['draw'], 'Défaite': res_h['loss']})
        elif mode == "Extérieur":
            # MyTeam Away vs Opp Home. Opp Win = MyTeam Loss
            rows.append({'Adversaire': opp, 'Victoire': res_a['loss'], 'Nul': res_a['draw'], 'Défaite': res_a['win']})
        else: 
            rows.append({
                'Adversaire': opp,
                'Victoire': (res_h['win'] + res_a['loss']) / 2,
                'Nul': (res_h['draw'] + res_a['draw']) / 2,
                'Défaite': (res_h['loss'] + res_a['win']) / 2
            })
            
    df_res = pd.DataFrame(rows)
    if not df_res.empty:
        df_res = df_res.set_index('Adversaire').applymap(lambda x: f"{x:.1f}%")
    return df_res

def get_spider_data_normalized(df, team1, team2=None):
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    metrics = {
        'Buts': lambda x, h: x['full_time_home_goals'] if h else x['full_time_away_goals'],
        'Tirs Cadrés': lambda x, h: x['home_shots_on_target'] if h else x['away_shots_on_target'],
        'Corners': lambda x, h: x['home_corners'] if h else x['away_corners'],
        'Fairplay': lambda x, h: x['home_yellow_cards'] if h else x['away_yellow_cards'],
        'Défense': lambda x, h: x['full_time_away_goals'] if h else x['full_time_home_goals']
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
            # Pour Fairplay et Défense, une petite valeur est mieux -> on inverse
            if k in ['Fairplay', 'Défense']: v = 100 - (raw[k]/max_v[k]*100)
            else: v = (raw[k]/max_v[k])*100
            norm.append(v)
        return norm

    v1 = get_norm(team1)
    v2 = get_norm(team2) if team2 else [df_s[k].mean()/max_v[k]*100 if k not in ['Fairplay', 'Défense'] else 100-(df_s[k].mean()/max_v[k]*100) for k in metrics]
    return list(metrics.keys()), v1, v2, df_s

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
c_title, c_form = st.columns([2, 1])
with c_title:
    st.title(f"📊 {my_team}")
with c_form:
    st.caption("Forme récente (Matchs terminés)")
    st.markdown(get_team_form_html(df_matchs, my_team), unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.markdown(f'<div class="metric-card"><div class="metric-label">Classement</div><div class="metric-value">{int(stats_team["rang"])}e</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="metric-card"><div class="metric-label">Points</div><div class="metric-value">{int(stats_team["total_points"])}</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="metric-card"><div class="metric-label">Buts Pour</div><div class="metric-value">{int(stats_team["total_bp"])}</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ================= STATS & BETTING =================
st.subheader("📈 Stats Jeu & Paris Sportifs")
stake = st.number_input("Mise par match (€)", min_value=1, value=10, step=5)
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
        colors = ['#2ECC71' if x > 0 else '#E74C3C' for x in bet_df['Profit']]
        bet_df['Text'] = bet_df.apply(lambda x: f"{x['Profit']:+.0f}€ ({x['ROI']:+.1f}%)", axis=1)
        fig_bet = go.Figure(go.Bar(x=bet_df['Type'], y=bet_df['Profit'], text=bet_df['Text'], marker_color=colors, textposition='auto'))
        fig_bet.update_layout(title="", font=dict(color='#E0E0E0'), paper_bgcolor='rgba(255,255,255,0.1)', plot_bgcolor='rgba(255,255,255,0.1)', height=350)
        st.plotly_chart(fig_bet, use_container_width=True)

with col_radar:
    st.markdown("##### 🕸️ Comparateur de Style")
    comp_target = st.selectbox("Comparer avec :", ["Moyenne Ligue"] + [t for t in teams if t != my_team])
    tgt = None if comp_target == "Moyenne Ligue" else comp_target
    cats, v1, v2, _ = get_spider_data_normalized(df_matchs, my_team, tgt)
    if cats:
        fig_rad = go.Figure()
        fig_rad.add_trace(go.Scatterpolar(r=v1, theta=cats, fill='toself', name=my_team, line_color='#DAE025'))
        c2 = '#95A5A6' if comp_target == "Moyenne Ligue" else '#E74C3C'
        fig_rad.add_trace(go.Scatterpolar(r=v2, theta=cats, fill='toself', name=comp_target, line_color=c2, opacity=0.6))
        fig_rad.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], color='#E0E0E0', gridcolor='#555'), angularaxis=dict(color='#E0E0E0')), font=dict(color='#E0E0E0'), paper_bgcolor='rgba(255,255,255,0.1)', plot_bgcolor='rgba(255,255,255,0.1)', margin=dict(t=30, b=30, l=40, r=40), legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_rad, use_container_width=True)

st.markdown("---")

# ================= SIMULATION OU VUE GLOBALE =================
st.subheader("⚔️ Analyse Adversaire")
opp_selection = st.selectbox("Adversaire", ["Vue d'ensemble (Tous)"] + [t for t in teams if t != my_team])

if opp_selection == "Vue d'ensemble (Tous)":
    st.markdown(f"##### 📜 Probabilités par adversaire - {my_team}")
    mode_filter = st.radio("Filtre Lieu :", ["Global", "Domicile", "Extérieur"], horizontal=True)
    df_probs = get_probabilities_table(df_history, my_team, teams, mode_filter)
    if not df_probs.empty: st.dataframe(df_probs, use_container_width=True)
    else: st.info("Pas assez de données.")

else:
    opp = opp_selection
    c_opt, c_res = st.columns([1, 2])
    with c_opt:
        loc = st.radio("Lieu", ["Domicile", "Extérieur"])
        is_home = "Domicile" in loc
    
    # 1. Calculs Probabilités Poisson & Score Exact
    th, ta = (my_team, opp) if is_home else (opp, my_team)
    hm = df_history[df_history['home_team']==th]
    am = df_history[df_history['away_team']==ta]
    
    if not hm.empty and not am.empty:
        av_h = df_history['full_time_home_goals'].mean()
        av_a = df_history['full_time_away_goals'].mean()
        att_h, def_h = hm['full_time_home_goals'].mean()/av_h, hm['full_time_away_goals'].mean()/av_a
        att_a, def_a = am['full_time_away_goals'].mean()/av_a, am['full_time_home_goals'].mean()/av_h
        
        # Le coeur du calcul
        res = calculate_match_probabilities_detailed(att_h, def_h, att_a, def_a, av_h, av_a)
        
        # Probabilité de victoire pour MON équipe
        if is_home: 
            my_win_prob = res['win']
            my_loss_prob = res['loss']
        else: 
            my_win_prob = res['loss'] # Car 'loss' pour Home = Win pour Away
            my_loss_prob = res['win']
            
        with c_res:
            # Score EXACT le plus probable (Mode)
            s_h, s_a = res['exact_score']
            
            st.markdown(f"""
                <div class="score-card">
                    <div style="color:#DAE025;">SCORE LE PLUS PROBABLE</div>
                    <div class="score-display">{th} {s_h} - {s_a} {ta}</div>
                    <div style="font-size:0.8rem; color:#AAA;">xG: {res['xg_h']:.2f} - {res['xg_a']:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # --- CONSEIL PARI ---
            # Logique de sécurité : On prend la plus haute proba 1N2
            probs = {'Victoire': my_win_prob, 'Nul': res['draw'], 'Défaite': my_loss_prob}
            best_opt = max(probs, key=probs.get)
            best_val = probs[best_opt]
            
            color = "#2ECC71" if best_val > 45 else "#F1C40F"
            st.markdown(f"""
                <div class="advice-box" style="border-color: {color}; background-color: rgba(255,255,255,0.05);">
                    💡 <b>Option la plus sûre (1N2) :</b> <br>
                    <span style="font-size: 1.2rem; font-weight: bold; color: {color};">
                        {best_opt} {my_team}
                    </span> 
                    <span style="color: #AAA;">({best_val:.1f}% de chance)</span>
                </div>
            """, unsafe_allow_html=True)
            
            # --- CONSEIL BUTS ---
            goals_opt = "Plus de 2.5 buts" if res['over_2_5'] > 50 else "Moins de 2.5 buts"
            goals_prob = max(res['over_2_5'], res['under_2_5'])
            st.markdown(f"""
                <div class="advice-box" style="border-color: #3498DB; background-color: rgba(255,255,255,0.05);">
                    ⚽ <b>Conseil Buts :</b> <br>
                    <span style="font-size: 1.1rem; font-weight: bold; color: #3498DB;">{goals_opt}</span>
                    <span style="color: #AAA;">({goals_prob:.1f}%)</span>
                </div>
            """, unsafe_allow_html=True)

st.markdown("---")
# ================= CALENDRIER & CLASSEMENT =================
c_cal, c_rank = st.columns([1, 1])

with c_rank:
    st.subheader("🏆 Classement Live")
    cols_show = ['rang', 'equipe', 'total_points', 'total_diff', 'total_V', 'total_N', 'total_D']
    st.dataframe(df_snap[cols_show].set_index('rang'), height=400, use_container_width=True)

with c_cal:
    st.subheader("🔮 Calendrier")
    # On simule un calendrier futur ou replay
    cutoff = pd.to_datetime(stats_team['match_timestamp']).replace(tzinfo=None)
    replay = cutoff < (pd.Timestamp.now() - pd.Timedelta(days=7))
    nxt = pd.DataFrame()
    if replay:
        fut = df_matchs[df_matchs['date'] > cutoff].sort_values('date')
        if not fut.empty:
            nxt = fut[fut['date'] <= (fut.iloc[0]['date'] + pd.Timedelta(days=5))]
            nxt = nxt.rename(columns={'date':'DateUtc', 'home_team':'HomeTeam', 'away_team':'AwayTeam'})
    else:
        api = get_live_schedule()
        if not api.empty:
            nxt = api[api['DateUtc'] <= (api.iloc[0]['DateUtc'] + pd.Timedelta(days=5))]
            # Simple mapping si nécessaire
            NAME_MAPPING = { "Paris Saint Germain": "Paris SG", "Olympique de Marseille": "Marseille", "Olympique Lyonnais": "Lyon", "AS Monaco": "Monaco", "Lille OSC": "Lille", "Stade Rennais": "Rennes", "OGC Nice": "Nice", "RC Lens": "Lens", "Stade de Reims": "Reims", "Strasbourg Alsace": "Strasbourg", "Montpellier HSC": "Montpellier", "FC Nantes": "Nantes", "Toulouse FC": "Toulouse", "Stade Brestois 29": "Brest", "FC Lorient": "Lorient", "Clermont Foot": "Clermont", "Le Havre AC": "Le Havre", "FC Metz": "Metz", "AJ Auxerre": "Auxerre", "Angers SCO": "Angers", "AS Saint-Etienne": "Saint Etienne" }
            nxt['HomeTeam'] = nxt['HomeTeam'].apply(lambda x: NAME_MAPPING.get(x, x))
            nxt['AwayTeam'] = nxt['AwayTeam'].apply(lambda x: NAME_MAPPING.get(x, x))

    if not nxt.empty:
        preds = []
        for _, m in nxt.iterrows():
            d, e = m['HomeTeam'], m['AwayTeam']
            # Petit calcul poisson rapide pour le tableau
            h_m = df_history[df_history['home_team']==d]
            a_m = df_history[df_history['away_team']==e]
            sc = "N/A"
            if not h_m.empty and not a_m.empty:
                ah, dh = h_m['full_time_home_goals'].mean(), h_m['full_time_away_goals'].mean()
                aa, da = a_m['full_time_away_goals'].mean(), a_m['full_time_home_goals'].mean()
                # On réutilise la fonction détaillée pour avoir le score exact (mode)
                # On triche un peu en prenant les moyennes globales de l'historique chargé
                av_h = df_history['full_time_home_goals'].mean()
                av_a = df_history['full_time_away_goals'].mean()
                res = calculate_match_probabilities_detailed(
                    h_m['full_time_home_goals'].mean()/av_h, h_m['full_time_away_goals'].mean()/av_a,
                    a_m['full_time_away_goals'].mean()/av_a, a_m['full_time_home_goals'].mean()/av_h,
                    av_h, av_a
                )
                s1, s2 = res['exact_score']
                sc = f"{s1}-{s2}"
                
            preds.append({"Date": m['DateUtc'].strftime('%d/%m %H:%M'), "Match": f"{d} - {e}", "Pred": sc})
        st.dataframe(pd.DataFrame(preds).set_index("Date"), use_container_width=True)
    else: st.info("Pas de matchs proches.")

st.markdown("---")
# GRAPH TRAJECTOIRE
hist = df_class[df_class['equipe'] == my_team]
fig = px.line(hist, x='journee_team', y='total_points', title=f"Trajectoire {my_team}", markers=True)
fig.update_traces(line_color='#DAE025', line_width=4)
fig.update_layout(paper_bgcolor='rgba(255,255,255,0.1)', plot_bgcolor='rgba(255,255,255,0.1)', font=dict(color='#E0E0E0'))
st.plotly_chart(fig, use_container_width=True)