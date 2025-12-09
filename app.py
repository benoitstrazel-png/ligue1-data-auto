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

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Ligue 1 Data Center", layout="wide", page_icon="⚽")

# --- 2. STYLE CSS AVANCÉ (THEME SOMBRE) ---
st.markdown("""
    <style>
    .stApp { background-color: #1A1C23 !important; }
    h1, h2, h3, h4, h5, p, span, div, label, .stDataFrame, .stTable, li { color: #FFFFFF !important; }
    
    /* Titre Principal */
    .main-title {
        font-size: 3rem; font-weight: 900; text-transform: uppercase;
        background: -webkit-linear-gradient(left, #DAE025, #FFFFFF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    
    /* Cartes */
    .metric-card {
        background-color: #2C3E50; padding: 15px; border-radius: 12px;
        text-align: center; border: 1px solid #444; margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-value { font-size: 1.8rem; font-weight: 800; margin: 0; color: #2ECC71 !important; }
    .metric-label { font-size: 0.9rem; color: #CCC !important; }
    
    /* Score & Conseils */
    .score-card {
        background: linear-gradient(135deg, #091C3E 0%, #1A1C23 100%);
        border: 2px solid #DAE025; border-radius: 15px; padding: 20px;
        text-align: center; margin-top: 10px;
    }
    .score-display { font-size: 3rem; font-weight: bold; color: #FFFFFF !important; font-family: monospace; }
    
    .advice-box {
        background-color: rgba(46, 204, 113, 0.1); border-left: 5px solid #2ECC71;
        padding: 15px; margin-top: 10px; border-radius: 5px;
    }
    .insight-box {
        background-color: rgba(218, 224, 37, 0.1); border-left: 5px solid #DAE025;
        padding: 15px; margin-top: 10px; color: #FFFFFF; border-radius: 5px;
        font-size: 0.9rem;
    }
    
    /* Badges Forme */
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
    [data-testid="stSidebar"] { background-color: #091C3E !important; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stDataFrame"] { background-color: #2C3E50; }
    
    /* Tooltip Helper */
    .tooltip-icon { font-size: 0.8rem; cursor: help; color: #DAE025; vertical-align: super; }
    </style>
""", unsafe_allow_html=True)

# --- 3. NAVIGATION (Au début pour éviter les erreurs) ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Menu", ["Dashboard", "Classement Prédictif"])

st.sidebar.markdown("---")
use_dev_mode = st.sidebar.toggle("🛠️ Mode Développeur", value=False)
TARGET_DATASET = "historic_datasets_dev" if use_dev_mode else "historic_datasets"
if use_dev_mode: st.sidebar.warning(f"⚠️ Source : {TARGET_DATASET}")

# Logo Partenaire
betclic_logo = "https://upload.wikimedia.org/wikipedia/commons/3/36/Logo_Betclic.svg"
st.sidebar.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px; margin-top: 10px;">
        <a href="https://www.betclic.fr" target="_blank">
            <img src="{betclic_logo}" width="140" style="background-color: white; padding: 10px; border-radius: 5px;">
        </a>
        <p style="color: #CCC; font-size: 0.8rem; margin-top: 5px;">Partenaire Paris Sportifs</p>
    </div>
""", unsafe_allow_html=True)

# --- 4. CHARGEMENT DONNÉES ---
@st.cache_resource
def get_db_client():
    key_dict = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(key_dict)
    return bigquery.Client(credentials=creds, project=key_dict["project_id"])

@st.cache_data(ttl=3600)
def get_seasons_list(dataset_id):
    client = get_db_client()
    query = f"SELECT DISTINCT season FROM `{client.project}.{dataset_id}.matchs_clean` ORDER BY season DESC"
    return client.query(query).to_dataframe()['season'].tolist()

@st.cache_data(ttl=600)
def load_data_complete(season_name, history_seasons, dataset_id):
    client = get_db_client()
    p, d = client.project, dataset_id
    
    # Saison Focus
    q_curr = f"SELECT * FROM `{p}.{d}.matchs_clean` WHERE season = '{season_name}' ORDER BY date ASC"
    df_curr = client.query(q_curr).to_dataframe()
    
    # Historique Multi-Saisons
    s_str = "', '".join(history_seasons)
    q_hist = f"SELECT * FROM `{p}.{d}.matchs_clean` WHERE season IN ('{s_str}')"
    df_hist = client.query(q_hist).to_dataframe()
    
    # Classement
    q_class = f"SELECT * FROM `{p}.{d}.classement_live` WHERE saison = '{season_name}' ORDER BY journee_team ASC"
    df_class = client.query(q_class).to_dataframe()
    
    # Calendrier Futur
    q_cal = f"SELECT * FROM `{p}.{d}.referentiel_calendrier` WHERE datetime_match > CURRENT_TIMESTAMP() ORDER BY datetime_match ASC LIMIT 50"
    try: df_cal = client.query(q_cal).to_dataframe()
    except: df_cal = pd.DataFrame()
    
    for df in [df_curr, df_hist, df_class]:
        if not df.empty:
            for col in ['date', 'match_timestamp']:
                if col in df.columns: df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
                
    return df_curr, df_hist, df_class, df_cal

# --- 5. LOGIQUE MÉTIER & CALCULS ---

def calculate_probabilities(xg_home, xg_away):
    max_goals = 8
    pm = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            pm[i, j] = poisson.pmf(i, xg_home) * poisson.pmf(j, xg_away)
    
    p_win = np.sum(np.tril(pm, -1))
    p_draw = np.sum(np.diag(pm))
    p_loss = np.sum(np.triu(pm, 1))
    p_over = np.sum([pm[i,j] for i in range(max_goals) for j in range(max_goals) if (i+j)>2.5])
    exact_idx = np.unravel_index(np.argmax(pm, axis=None), pm.shape)
    
    return {'win': p_win, 'draw': p_draw, 'loss': p_loss, 'over25': p_over, 'exact': exact_idx}

def predict_match_advanced(df_history, team_home, team_away):
    if df_history.empty: return None
    avg_h = df_history['full_time_home_goals'].mean()
    avg_a = df_history['full_time_away_goals'].mean()
    
    h_hist = df_history[df_history['home_team'] == team_home]
    a_hist = df_history[df_history['away_team'] == team_away]
    if h_hist.empty or a_hist.empty: return None
    
    att_h = h_hist['full_time_home_goals'].mean() / avg_h
    def_h = h_hist['full_time_away_goals'].mean() / avg_a
    att_a = a_hist['full_time_away_goals'].mean() / avg_a
    def_a = a_hist['full_time_home_goals'].mean() / avg_h
    
    xg_h = att_h * def_a * avg_h
    xg_a = att_a * def_h * avg_a
    
    return calculate_probabilities(xg_h, xg_a)

def get_team_form_html(df_matchs, team, limit=5):
    matches = df_matchs[((df_matchs['home_team'] == team) | (df_matchs['away_team'] == team)) & (df_matchs['date'] < pd.Timestamp.now())].sort_values('date', ascending=False).head(limit).sort_values('date')
    html = '<div style="display:flex; align-items:center;">'
    for _, r in matches.iterrows():
        is_h = r['home_team'] == team
        res, cls, flame = 'D', 'loss', ''
        if r['full_time_result'] == 'D': res, cls = 'N', 'draw'
        elif (is_h and r['full_time_result']=='H') or (not is_h and r['full_time_result']=='A'):
            res, cls = 'V', 'win'
            if abs(r['full_time_home_goals'] - r['full_time_away_goals']) >= 3: flame = "🔥"
        opp = r['away_team'] if is_h else r['home_team']
        html += f'<div class="form-badge {cls}" title="vs {opp}">{res}{flame}</div>'
    html += '</div>'
    return html

def get_spider_data_normalized(df, team):
    # 1. Sécurité : Si le dataframe est vide ou None, on renvoie 3 listes vides
    if df is None or df.empty:
        return [], [], []

    metrics = {
        'Buts': 'full_time_home_goals', 'Tirs': 'home_shots', 
        'Cadrés': 'home_shots_on_target', 'Corners': 'home_corners'
    }
    
    # 2. Vérification que les colonnes nécessaires existent
    for col in metrics.values():
        if col not in df.columns:
            return [], [], []

    # 3. Calcul des Moyennes Ligue (Evite les crashs si données manquantes)
    avg_league = {}
    for k, v in metrics.items():
        v_away = v.replace('home', 'away')
        # On utilise fillna(0) pour éviter de propager des NaN
        m_home = df[v].mean()
        m_away = df[v_away].mean() if v_away in df.columns else 0
        avg_league[k] = (m_home + m_away) / 2
        
    # 4. Filtrage pour l'équipe
    df_t = df[df['home_team'] == team]
    if df_t.empty: 
        return [], [], []
    
    team_vals = []
    league_vals = []
    labels = []
    
    # 5. Construction des données du Radar
    for k, v in metrics.items():
        val = df_t[v].mean()
        ref_avg = avg_league.get(k, 1)
        
        # Sécurité Division par Zéro
        if ref_avg == 0: ref_avg = 1
            
        # Normalisation (Ligue = 50)
        norm_t = min(100, (val / ref_avg) * 50)
        norm_l = 50 
        
        team_vals.append(norm_t)
        league_vals.append(norm_l)
        labels.append(k)
        
    # 6. Toujours renvoyer 3 valeurs
    return labels, team_vals, league_vals

def calculate_nemesis_stats(df, team):
    stats = []
    df_team = df[(df['home_team'] == team) | (df['away_team'] == team)]
    if df_team.empty: return None
    for _, row in df_team.iterrows():
        is_h = row['home_team'] == team
        opp = row['away_team'] if is_h else row['home_team']
        res = 'N'
        if row['full_time_result'] == 'D': res = 'N'
        elif (is_h and row['full_time_result'] == 'H') or (not is_h and row['full_time_result'] == 'A'): res = 'V'
        else: res = 'D'
        stats.append({'opponent': opp, 'result': res, 'goals_against': row['full_time_away_goals'] if is_h else row['full_time_home_goals'], 'reds': row['home_red_cards'] if is_h else row['away_red_cards']})
    agg = pd.DataFrame(stats).groupby('opponent').agg({'result': list, 'goals_against': 'sum', 'reds': 'sum'})
    agg['wins'] = agg['result'].apply(lambda x: x.count('V'))
    agg['losses'] = agg['result'].apply(lambda x: x.count('D'))
    return agg

def calculate_advanced_stats_and_betting(df, team, stake=10):
    df_t = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
    if df_t.empty: return None, None
    strats = {'Victoire':0, 'Nul':0, 'Défaite':0} 
    invest = {k:0 for k in strats} 
    
    shots, target, yel, red = 0, 0, 0, 0
    
    for _, r in df_t.iterrows():
        is_h = r['home_team'] == team
        shots += r['home_shots'] if is_h else r['away_shots']
        target += r['home_shots_on_target'] if is_h else r['away_shots_on_target']
        yel += r['home_yellow_cards'] if is_h else r['away_yellow_cards']
        red += r['home_red_cards'] if is_h else r['away_red_cards']
        
        # Simulation Betting simple
        res = r['full_time_result']
        ho, do, ao = r.get('bet365_home_win_odds', 0), r.get('bet365_draw_odds', 0), r.get('bet365_away_win_odds', 0)
        
        if pd.notna(ho) and ho:
            invest['Victoire'] += stake
            if (is_h and res=='H') or (not is_h and res=='A'): strats['Victoire'] += (ho if is_h else ao)*stake - stake
            else: strats['Victoire'] -= stake
            
    nb = len(df_t)
    stats = {'avg_shots': shots/nb, 'avg_target': target/nb, 'avg_yellow': yel/nb, 'avg_red': red/nb}
    res_df = pd.DataFrame([{'Type': k, 'Profit': v} for k, v in strats.items() if invest[k]>0])
    return stats, res_df

def apply_standings_style(df):
    def style_rows(row):
        rank = row.name
        style = ''
        if rank <= 4: 
            style = 'background-color: rgba(66, 133, 244, 0.6); color: white;' 
        elif rank == 5: 
            style = 'background-color: rgba(255, 165, 0, 0.6); color: white;' 
        elif rank == 6: 
            style = 'background-color: rgba(46, 204, 113, 0.6); color: white;' 
        elif rank >= 16: 
            style = 'background-color: rgba(231, 76, 60, 0.5); color: white;' 
        
        # CORRECTIF CRITIQUE : Retourner une liste, pas un string
        return [style] * len(row)

    df_styled = df.copy()
    if 1 in df_styled.index: 
        df_styled.loc[1, 'equipe'] = "👑 " + df_styled.loc[1, 'equipe']
    
    return df_styled.style.apply(style_rows, axis=1)

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
        team_forces[t] = {'ah': h['full_time_home_goals'].mean()/avg_h, 'dh': h['full_time_away_goals'].mean()/avg_a, 'aa': a['full_time_away_goals'].mean()/avg_a, 'da': a['full_time_home_goals'].mean()/avg_h}
    res = []
    for h, a in remaining:
        if h not in team_forces or a not in team_forces: continue
        th, ta = team_forces[h], team_forces[a]
        mh = th['ah'] * ta['da'] * avg_h
        ma = ta['aa'] * th['dh'] * avg_a
        ph = 3 if mh > ma + 0.2 else (1 if abs(mh-ma) <= 0.2 else 0)
        pa = 3 if ma > mh + 0.2 else (1 if abs(mh-ma) <= 0.2 else 0)
        res.append({'equipe': h, 'pts': ph})
        res.append({'equipe': a, 'pts': pa})
    return pd.DataFrame(res)

def prepare_calendar_table(df_cal, df_history):
    if df_cal.empty: return pd.DataFrame()
    next_journee = df_cal['journee'].min()
    df_next = df_cal[df_cal['journee'] == next_journee].copy()
    rows = []
    for _, row in df_next.iterrows():
        home, away = row['home_team'], row['away_team']
        pred = predict_match_advanced(df_history, home, away)
        bg_h, bg_a, bg_d, best_bet = "", "", "", "Indécis"
        
        if pred:
            p_w, p_d, p_l, p_o = pred['win'], pred['draw'], pred['loss'], pred['over25']
            if p_w > 0.5: bg_h = "background-color: rgba(46, 204, 113, 0.4)"
            if p_l > 0.5: bg_a = "background-color: rgba(46, 204, 113, 0.4)"
            if p_d > 0.3: bg_d = "background-color: rgba(255, 165, 0, 0.4)"
            
            options = {f"Victoire {home}": p_w, f"Victoire {away}": p_l, "Nul": p_d, "+2.5 Buts": p_o}
            if p_o < 0.6: del options["+2.5 Buts"] # Seuil de confiance 60%
            
            best_opt = max(options, key=options.get)
            best_bet = f"{best_opt} ({options[best_opt]*100:.0f}%)"
            
            rows.append({
                "Date": row['datetime_match'].strftime('%d/%m %H:%M'), "Domicile": home,
                "Prob. 1": f"{p_w*100:.0f}%", "Prob. N": f"{p_d*100:.0f}%", "Prob. 2": f"{p_l*100:.0f}%",
                "Extérieur": away, "Meilleur Pari 💡": best_bet,
                "_bg_h": bg_h, "_bg_d": bg_d, "_bg_a": bg_a
            })
    return pd.DataFrame(rows)

def calculate_global_stats(df):
    if df.empty: return {}
    nb = len(df)
    g = df['full_time_home_goals'].sum() + df['full_time_away_goals'].sum()
    s = df['home_shots'].sum() + df['away_shots'].sum()
    return {'Buts Totaux': int(g), 'Buts / Match': round(g/nb, 2), 'Tirs / Match': round(s/nb, 1)}

# --- 6. INTERFACE ---
st.sidebar.markdown("---")
st.sidebar.title("🔍 Filtres")
all_seasons = get_seasons_list(TARGET_DATASET)
# Par défaut TOUTES les saisons sont sélectionnées
selected_seasons = st.sidebar.multiselect("Historique (Poisson)", all_seasons, default=all_seasons)
focus_season = sorted(selected_seasons, reverse=True)[0] if selected_seasons else all_seasons[0]

df_curr, df_hist, df_class, df_cal = load_data_complete(focus_season, selected_seasons, TARGET_DATASET)
teams = sorted(df_curr['home_team'].unique())

if page == "Dashboard":
    st.markdown('<div class="main-title">LIGUE 1 DATA CENTER</div>', unsafe_allow_html=True)
    
    # 1. SCORECARDS
    st.markdown(f"### 🌍 Statistiques {focus_season}")
    g_stats = calculate_global_stats(df_curr)
    cols = st.columns(len(g_stats))
    for i, (k, v) in enumerate(g_stats.items()):
        cols[i].markdown(f"""<div class="global-card"><p class="global-val">{v}</p><p class="global-lbl">{k}</p></div>""", unsafe_allow_html=True)
    
    st.markdown("---")

    # 2. ANALYSE CLUB
    st.subheader("🛡️ Analyse Club")
    my_team = st.sidebar.selectbox("Sélectionner un Club", teams)
    max_j = int(df_class['journee_team'].max()) if not df_class.empty else 1
    cur_j = st.sidebar.slider("Arrêter à la journée :", 1, max_j, max_j)
    
    df_snap = df_class[df_class['journee_team'] <= cur_j].sort_values('match_timestamp').groupby('equipe').last().reset_index()
    df_snap['rang'] = df_snap['total_points'].rank(ascending=False, method='min')
    
    if not df_snap[df_snap['equipe'] == my_team].empty:
        stats_team = df_snap[df_snap['equipe'] == my_team].iloc[0]
        
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
        
        st.markdown("---")
        # Visualisations Avancées (ROI + Radar)
        c_bet, c_rad = st.columns(2)
        with c_bet:
            st.subheader("💰 Rentabilité Paris")
            stake = 10
            adv_stats, bet_df = calculate_advanced_stats_and_betting(df_curr, my_team, stake)
            if bet_df is not None:
                fig_b = go.Figure(go.Bar(x=bet_df['Type'], y=bet_df['Profit'], marker_color=['#2ECC71' if x>0 else '#E74C3C' for x in bet_df['Profit']]))
                fig_b.update_layout(title=f"Profit/Perte (Mise {stake}€)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=300)
                st.plotly_chart(fig_b, use_container_width=True)
        
        with c_rad:
            st.subheader("🕸️ Style de Jeu")
            cats, v1, v2 = get_spider_data_normalized(df_curr, my_team)
            if cats:
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatterpolar(r=v1, theta=cats, fill='toself', name=my_team, line_color='#DAE025'))
                fig_r.add_trace(go.Scatterpolar(r=v2, theta=cats, fill='toself', name='Moyenne Ligue', line_color='#95A5A6'))
                fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], color='white')), paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=300)
                st.plotly_chart(fig_r, use_container_width=True)

        st.markdown("---")
        st.subheader("⚔️ Analyse des Confrontations (Bête Noire)")
        agg = calculate_nemesis_stats(df_hist, my_team)
        if agg is not None:
            best = agg.sort_values('wins', ascending=False).index[0]
            worst = agg.sort_values('losses', ascending=False).index[0]
            k1, k2 = st.columns(2)
            k1.success(f"**Proie Favorite :** {best} ({agg.loc[best, 'wins']} victoires)")
            k2.error(f"**Bête Noire :** {worst} ({agg.loc[worst, 'losses']} défaites)")

        st.markdown("---")
        st.subheader("🔮 Prédiction & Paris")
        
        c_sel, c_res = st.columns([1, 2])
        with c_sel:
            opp_sel = st.selectbox("Adversaire", ["Vue Globale"] + [t for t in teams if t != my_team])
            loc = st.radio("Lieu", ["Domicile", "Extérieur"])
            is_home = "Domicile" in loc
            th, ta = (my_team, opp_sel) if is_home else (opp_sel, my_team)

        if opp_sel == "Vue Globale":
            rows = []
            for o in teams:
                if o == my_team: continue
                r = predict_match_advanced(df_hist, my_team, o) 
                if r: rows.append({'Adversaire': o, 'Victoire': r['win'], 'Nul': r['draw'], 'Défaite': r['loss']})
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index('Adversaire').style.format("{:.1f}%").background_gradient(subset=['Victoire'], cmap='Greens'), use_container_width=True)
        else:
            res = predict_match_advanced(df_hist, th, ta)
            if res:
                with c_res:
                    s1, s2 = res['exact']
                    st.markdown(f"""<div class="score-card"><div style="color:#DAE025;">SCORE PROBABLE</div><div class="score-display">{th} {s1}-{s2} {ta}</div></div>""", unsafe_allow_html=True)
                    
                    pw, pl, pd_ = (res['win'], res['loss'], res['draw']) if is_home else (res['loss'], res['win'], res['draw'])
                    if pw > max(pl, pd_): txt, prob, col = f"Victoire {my_team}", pw, "#2ECC71"
                    elif pl > max(pw, pd_): txt, prob, col = f"Victoire {opp_sel}", pl, "#E74C3C"
                    else: txt, prob, col = "Match Nul", pd_, "#F1C40F"
                    
                    goal_txt = "+2.5 Buts" if res['over25'] > 0.6 else ("-2.5 Buts" if (1-res['over25']) > 0.6 else "Pas de tendance (Cote serrée)")
                    goal_prob = res['over25'] if res['over25'] > 0.6 else (1-res['over25'])
                    
                    c1, c2 = st.columns(2)
                    c1.markdown(f"""<div class="advice-box" style="border-color:{col}"><span style="font-weight:bold;color:{col}">{txt}</span> ({prob*100:.1f}%)</div>""", unsafe_allow_html=True)
                    if goal_prob > 0.6:
                        c2.markdown(f"""<div class="advice-box" style="border-color:#3498DB"><span style="font-weight:bold;color:#3498DB">{goal_txt}</span> ({goal_prob*100:.1f}%)</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📅 Prochaine Journée (Smart Analysis)")
    df_smart = prepare_calendar_table(df_cal, df_hist)
    if not df_smart.empty:
        def style_cal(row):
            return ['', '', row['_bg_h'], row['_bg_d'], row['_bg_a'], '', '', '', '', '']
        st.dataframe(df_smart.style.apply(style_cal, axis=1), column_config={"_bg_h":None, "_bg_d":None, "_bg_a":None}, use_container_width=True, hide_index=True)
    else: st.info("Aucun match futur trouvé.")

    st.markdown("---")
    st.subheader("🏆 Classement Live")
    df_show = df_snap[['rang', 'equipe', 'total_points', 'total_diff', 'total_V', 'total_N', 'total_D']].set_index('rang').sort_index()
    # Utilisation correcte de la fonction
    st.dataframe(apply_standings_style(df_show), use_container_width=True)

elif page == "Classement Prédictif":
    st.markdown('<div class="main-title">🔮 CLASSEMENT FINAL PROJETÉ</div>', unsafe_allow_html=True)
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
                st.success("Projection terminée !")
                st.dataframe(apply_standings_style(final), use_container_width=True, height=600)