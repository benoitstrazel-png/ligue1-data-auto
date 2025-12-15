import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import poisson
import itertools

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Ligue 1 Data Center", layout="wide", page_icon="⚽")

# --- 2. STYLE CSS AVANCÉ (THEME SOMBRE & SCIENTIFIQUE) ---
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
    
    /* Cartes Metrics */
    .metric-card {
        background-color: #2C3E50; padding: 15px; border-radius: 12px;
        text-align: center; border: 1px solid #444; margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-value { font-size: 1.8rem; font-weight: 800; margin: 0; color: #2ECC71 !important; }
    .metric-label { font-size: 0.9rem; color: #CCC !important; }
    
    /* Global Stats Card */
    .global-card {
        background-color: #2C3E50; padding: 15px; border-radius: 12px;
        text-align: center; border: 1px solid #444; margin-bottom: 10px;
    }
    .global-val { font-size: 2rem; font-weight: bold; color: #DAE025 !important; }
    .global-lbl { font-size: 0.8rem; text-transform: uppercase; color: #AAA !important; }

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
    .value-bet-box {
        background-color: rgba(218, 224, 37, 0.1); border-left: 5px solid #DAE025;
        padding: 15px; margin-top: 10px; color: #FFFFFF; border-radius: 5px;
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
    </style>
""", unsafe_allow_html=True)

# --- 3. NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Menu", ["Dashboard", "Classement Prédictif"])

st.sidebar.markdown("---")
use_dev_mode = st.sidebar.toggle("🛠️ Mode Développeur", value=False)
TARGET_DATASET = "historic_datasets_dev" if use_dev_mode else "historic_datasets"

# Logo Partenaire (Préservé)
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
    
    # Données saison en cours + historique
    q_curr = f"SELECT * FROM `{p}.{d}.matchs_clean` WHERE season = '{season_name}' ORDER BY date ASC"
    df_curr = client.query(q_curr).to_dataframe()
    
    s_str = "', '".join(history_seasons)
    q_hist = f"SELECT * FROM `{p}.{d}.matchs_clean` WHERE season IN ('{s_str}')"
    df_hist = client.query(q_hist).to_dataframe()
    
    q_class = f"SELECT * FROM `{p}.{d}.classement_live` WHERE saison = '{season_name}' ORDER BY journee_team ASC"
    df_class = client.query(q_class).to_dataframe()
    
    q_cal = f"SELECT * FROM `{p}.{d}.referentiel_calendrier` WHERE datetime_match > CURRENT_TIMESTAMP() ORDER BY datetime_match ASC LIMIT 100"
    try: df_cal = client.query(q_cal).to_dataframe()
    except: df_cal = pd.DataFrame()
    
    # Nettoyage et Création xG Proxy (NOUVEAU MOTEUR SCIENTIFIQUE)
    for df in [df_curr, df_hist, df_class, df_cal]:
        if not df.empty:
            for col in ['date', 'match_timestamp', 'datetime_match']:
                if col in df.columns: df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)

    # Ajout xG Proxy (Estimé: 0.3 * Cadré + 0.07 * Non-Cadré)
    if not df_hist.empty and 'home_shots_on_target' in df_hist.columns:
        for df in [df_curr, df_hist]:
            if not df.empty and 'home_shots' in df.columns:
                df['home_xg_proxy'] = (df['home_shots_on_target'] * 0.30) + ((df['home_shots'] - df['home_shots_on_target']) * 0.07)
                df['away_xg_proxy'] = (df['away_shots_on_target'] * 0.30) + ((df['away_shots'] - df['away_shots_on_target']) * 0.07)
            else:
                # Fallback si stats tirs manquantes
                df['home_xg_proxy'] = df['full_time_home_goals']
                df['away_xg_proxy'] = df['full_time_away_goals']

    return df_curr, df_hist, df_class, df_cal

# --- 5. MOTEUR SCIENTIFIQUE (EMA + POISSON + MONTE CARLO) ---

def calculate_ema_strength(df, team, is_home, metric='att', alpha=0.15):
    """Calcule la force pondérée par le temps (Exponential Moving Average)."""
    team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('date')
    if team_matches.empty: return 1.0

    values = []
    for _, row in team_matches.iterrows():
        if metric == 'att':
            val = row['home_xg_proxy'] if row['home_team'] == team else row['away_xg_proxy']
        else: # defense
            val = row['away_xg_proxy'] if row['home_team'] == team else row['home_xg_proxy']
        values.append(val)
    
    return pd.Series(values).ewm(alpha=alpha, adjust=False).mean().iloc[-1]

def predict_match_scientific(df_history, home_team, away_team):
    """Moteur de prédiction : Poisson + EMA + xG Proxy (Remplace predict_match_advanced)"""
    if df_history.empty: return None

    # Moyennes Ligue
    avg_h = df_history['home_xg_proxy'].mean()
    avg_a = df_history['away_xg_proxy'].mean()
    if pd.isna(avg_h) or avg_h == 0: avg_h, avg_a = 1.3, 1.0 # Sécurité
    
    # Forces Pondérées
    home_att = calculate_ema_strength(df_history, home_team, True, 'att') / avg_h
    home_def = calculate_ema_strength(df_history, home_team, True, 'def') / avg_a
    away_att = calculate_ema_strength(df_history, away_team, False, 'att') / avg_a
    away_def = calculate_ema_strength(df_history, away_team, False, 'def') / avg_h
    
    # Lambdas (xG attendus)
    xg_h = home_att * away_def * avg_h
    xg_a = away_att * home_def * avg_a
    
    # Distribution de Poisson
    max_goals = 8
    pm = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            pm[i, j] = poisson.pmf(i, xg_h) * poisson.pmf(j, xg_a)
            
    p_win = np.sum(np.tril(pm, -1))
    p_draw = np.sum(np.diag(pm))
    p_loss = np.sum(np.triu(pm, 1))
    p_over = np.sum([pm[i,j] for i in range(max_goals) for j in range(max_goals) if (i+j)>2.5])
    
    exact_idx = np.unravel_index(np.argmax(pm, axis=None), pm.shape)
    
    # Structure compatible avec l'ancien code + nouveautés
    return {
        'win': p_win, 'draw': p_draw, 'loss': p_loss, 
        'over25': p_over, 'exact': exact_idx,
        'xg_home': xg_h, 'xg_away': xg_a
    }

def detect_value_bet(prediction, odds_h, odds_d, odds_a):
    """Détecte les Value Bets (EV+)"""
    if not (odds_h and odds_d and odds_a): return None
    ev_h = (prediction['win'] * odds_h) - 1
    ev_d = (prediction['draw'] * odds_d) - 1
    ev_a = (prediction['loss'] * odds_a) - 1
    
    best_ev = max(ev_h, ev_d, ev_a)
    if best_ev < 0: return None
    
    if best_ev == ev_h: return {'sel': '1', 'ev': best_ev, 'odd': odds_h, 'prob': prediction['win']}
    if best_ev == ev_d: return {'sel': 'N', 'ev': best_ev, 'odd': odds_d, 'prob': prediction['draw']}
    return {'sel': '2', 'ev': best_ev, 'odd': odds_a, 'prob': prediction['loss']}

# --- 6. FONCTIONS UTILITAIRES DASHBOARD (PRÉSERVÉES) ---

def get_team_form_html(df_matchs, team, limit=5):
    matches = df_matchs[((df_matchs['home_team'] == team) | (df_matchs['away_team'] == team)) & (df_matchs['date'] < pd.Timestamp.now())].sort_values('date', ascending=False).head(limit).sort_values('date')
    html = '<div style="display:flex; align-items:center;">'
    for _, r in matches.iterrows():
        is_h = r['home_team'] == team
        res, cls = 'D', 'loss'
        if r['full_time_result'] == 'D': res, cls = 'N', 'draw'
        elif (is_h and r['full_time_result']=='H') or (not is_h and r['full_time_result']=='A'): res, cls = 'V', 'win'
        html += f'<div class="form-badge {cls}">{res}</div>'
    html += '</div>'
    return html

def get_spider_data_normalized(df, team):
    if df is None or df.empty: return [], [], []
    # Mise à jour avec xG Proxy
    metrics = {'Buts': 'full_time_home_goals', 'xG (Est.)': 'home_xg_proxy', 'Tirs': 'home_shots', 'Cadrés': 'home_shots_on_target', 'Corners': 'home_corners'}
    
    avg_league = {}
    for k, v in metrics.items():
        v_away = v.replace('home', 'away')
        if v not in df.columns: continue
        m_home = df[v].mean()
        m_away = df[v_away].mean() if v_away in df.columns else 0
        avg_league[k] = (m_home + m_away) / 2
        
    df_t = df[df['home_team'] == team]
    if df_t.empty: return [], [], []
    
    team_vals, league_vals, labels = [], [], []
    for k, v in metrics.items():
        if v not in df_t.columns: continue
        val = df_t[v].mean()
        ref_avg = avg_league.get(k, 1)
        norm_t = min(100, (val / ref_avg) * 50)
        team_vals.append(norm_t); league_vals.append(50); labels.append(k)
        
    return labels, team_vals, league_vals

def calculate_nemesis_stats(df, team):
    stats = []
    df_team = df[(df['home_team'] == team) | (df['away_team'] == team)]
    if df_team.empty: return None
    for _, row in df_team.iterrows():
        is_h = row['home_team'] == team
        opp = row['away_team'] if is_h else row['home_team']
        # Détection défaite
        res = 'D'
        if row['full_time_result'] == 'D': res = 'N'
        elif (is_h and row['full_time_result'] == 'H') or (not is_h and row['full_time_result'] == 'A'): res = 'V'
        
        if res == 'D': # On ne garde que les défaites pour la bête noire
            stats.append({'opponent': opp})
            
    if not stats: return None
    agg = pd.DataFrame(stats).groupby('opponent').size().reset_index(name='losses')
    return agg.sort_values('losses', ascending=False).head(1)

def calculate_advanced_stats_and_betting(df, team, stake=10):
    df_t = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
    if df_t.empty: return None
    strats = {'Victoire':0, 'Nul':0} 
    for _, r in df_t.iterrows():
        is_h = r['home_team'] == team
        res = r['full_time_result']
        ho = r.get('bet365_home_win_odds', 0)
        do = r.get('bet365_draw_odds', 0)
        ao = r.get('bet365_away_win_odds', 0)
        
        if pd.isna(ho): continue
        
        # Stratégie Victoire
        if (is_h and res=='H') or (not is_h and res=='A'): strats['Victoire'] += (ho if is_h else ao)*stake - stake
        else: strats['Victoire'] -= stake
        # Stratégie Nul
        if res == 'D': strats['Nul'] += do*stake - stake
        else: strats['Nul'] -= stake
            
    return pd.DataFrame([{'Type': k, 'Profit': v} for k, v in strats.items()])

def run_monte_carlo(df_played, df_cal, df_history, n_simulations=500):
    """Simulation de fin de saison (Remplace simulate_season_end)"""
    teams = sorted(list(set(df_played['home_team'].unique()) | set(df_played['away_team'].unique())))
    current_points = {t: 0 for t in teams}
    # Points actuels
    for _, r in df_played.iterrows():
        if r['full_time_result'] == 'H': current_points[r['home_team']] += 3
        elif r['full_time_result'] == 'A': current_points[r['away_team']] += 3
        elif r['full_time_result'] == 'D': 
            current_points[r['home_team']] += 1
            current_points[r['away_team']] += 1
            
    # Paramètres équipes
    avg_h = df_history['home_xg_proxy'].mean()
    avg_a = df_history['away_xg_proxy'].mean()
    team_stats = {}
    for t in teams:
        team_stats[t] = {
            'att_h': calculate_ema_strength(df_history, t, True, 'att')/avg_h,
            'def_h': calculate_ema_strength(df_history, t, True, 'def')/avg_a,
            'att_a': calculate_ema_strength(df_history, t, False, 'att')/avg_a,
            'def_a': calculate_ema_strength(df_history, t, False, 'def')/avg_h
        }
    
    # Matchs à jouer
    future = []
    for _, row in df_cal.iterrows():
        h, a = row['home_team'], row['away_team']
        if h in teams and a in teams:
            lh = team_stats[h]['att_h'] * team_stats[a]['def_a'] * avg_h
            la = team_stats[a]['att_a'] * team_stats[h]['def_h'] * avg_a
            future.append({'h': h, 'a': a, 'lh': lh, 'la': la})
            
    results = {t: 0 for t in teams}
    prog = st.progress(0)
    
    for i in range(n_simulations):
        sim_pts = current_points.copy()
        for m in future:
            # Tirage aléatoire Poisson
            gh, ga = np.random.poisson(m['lh']), np.random.poisson(m['la'])
            if gh > ga: sim_pts[m['h']] += 3
            elif ga > gh: sim_pts[m['a']] += 3
            else: sim_pts[m['h']] += 1; sim_pts[m['a']] += 1
        
        # Vainqueur
        champion = max(sim_pts, key=sim_pts.get)
        results[champion] += 1
        if i % 50 == 0: prog.progress(i/n_simulations)
    
    prog.empty()
    df_res = pd.DataFrame(list(results.items()), columns=['equipe', 'titres'])
    df_res['Prob. Titre'] = (df_res['titres'] / n_simulations * 100).map('{:.1f}%'.format)
    return df_res.sort_values('titres', ascending=False).drop(columns=['titres'])

def apply_standings_style(df):
    def style_rows(row):
        rank = row.name
        style = ''
        if rank <= 4: style = 'background-color: rgba(66, 133, 244, 0.4);' 
        elif rank >= 16: style = 'background-color: rgba(231, 76, 60, 0.4);' 
        return [style] * len(row)
    return df.style.apply(style_rows, axis=1)

def calculate_global_stats(df):
    if df.empty: return {}
    g = df['full_time_home_goals'].sum() + df['full_time_away_goals'].sum()
    return {'Buts Totaux': g, 'Buts / Match': round(g/len(df), 2) if len(df) else 0}

# --- 7. INTERFACE GRAPHIQUE ---
all_seasons = get_seasons_list(TARGET_DATASET)
selected_seasons = st.sidebar.multiselect("Historique (Training)", all_seasons, default=all_seasons[:4] if len(all_seasons)>0 else [])
focus_season = all_seasons[0] if len(all_seasons)>0 else "2024-2025"

df_curr, df_hist, df_class, df_cal = load_data_complete(focus_season, selected_seasons, TARGET_DATASET)
teams = sorted(df_curr['home_team'].unique()) if not df_curr.empty else []

if page == "Dashboard":
    st.markdown('<div class="main-title">LIGUE 1 DATA CENTER</div>', unsafe_allow_html=True)
    
    # 1. SCORECARDS GLOBALES
    g_stats = calculate_global_stats(df_curr)
    if g_stats:
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="global-card"><div class="global-val">{g_stats["Buts Totaux"]}</div><div class="global-lbl">Buts Inscrits</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="global-card"><div class="global-val">{g_stats["Buts / Match"]}</div><div class="global-lbl">Moyenne / Match</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="global-card"><div class="global-val">{len(teams)}</div><div class="global-lbl">Clubs en Lice</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")

    # 2. ANALYSE CLUB
    st.subheader("🛡️ Analyse Club")
    if teams:
        my_team = st.sidebar.selectbox("Sélectionner un Club", teams)
        
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            # Forme et Stats
            st.markdown(f"## {my_team}")
            st.markdown(get_team_form_html(df_curr, my_team), unsafe_allow_html=True)
            
            # Ranking Info
            if not df_class.empty:
                info = df_class[df_class['equipe']==my_team]
                if not info.empty:
                    pts = info.iloc[0]['total_points']
                    rk = info['total_points'].rank(ascending=False).iloc[0]
                    st.metric("Classement Actuel", f"{int(rk)}e", f"{int(pts)} pts")

            # ROI Chart (Préservé)
            st.markdown("##### 💰 Profit/Pertes (Mise 10€)")
            bet_df = calculate_advanced_stats_and_betting(df_curr, my_team)
            if bet_df is not None:
                fig_roi = px.bar(bet_df, x='Type', y='Profit', color='Profit', color_continuous_scale=['#E74C3C', '#2ECC71'])
                fig_roi.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig_roi, use_container_width=True)
                
            # Bête Noire (Préservé)
            nem = calculate_nemesis_stats(df_hist, my_team)
            if nem is not None and not nem.empty:
                opp = nem.iloc[0]['opponent']
                nb = nem.iloc[0]['losses']
                st.error(f"☠️ Bête Noire : {opp} ({nb} défaites)")

        with col_right:
            # Spider Chart (Mis à jour avec xG)
            cats, v1, v2 = get_spider_data_normalized(df_curr, my_team)
            if cats:
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatterpolar(r=v1, theta=cats, fill='toself', name=my_team, line_color='#DAE025'))
                fig_r.add_trace(go.Scatterpolar(r=v2, theta=cats, fill='toself', name='Moyenne Ligue', line_color='#95A5A6'))
                fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=350, showlegend=True)
                st.plotly_chart(fig_r, use_container_width=True)

        st.markdown("---")
        st.subheader("🔮 Prédiction & Paris")
        
        c_sel, c_res = st.columns([1, 2])
        with c_sel:
            opp_sel = st.selectbox("Adversaire", ["Vue Globale"] + [t for t in teams if t != my_team])
            loc = st.radio("Lieu", ["Domicile", "Extérieur"])
            is_home = "Domicile" in loc
            th, ta = (my_team, opp_sel) if is_home else (opp_sel, my_team)

        if opp_sel == "Vue Globale":
            # Tableau matriciel préservé
            rows = []
            for o in teams:
                if o == my_team: continue
                # Utilisation du nouveau moteur scientifique
                r = predict_match_scientific(df_hist, my_team, o) 
                if r: rows.append({'Adversaire': o, 'Victoire': r['win'], 'Nul': r['draw'], 'Défaite': r['loss']})
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index('Adversaire').style.format("{:.1f}%").background_gradient(subset=['Victoire'], cmap='Greens'), use_container_width=True)
        else:
            # Prédiction Match Unique (Moteur Scientifique)
            res = predict_match_scientific(df_hist, th, ta)
            if res:
                with c_res:
                    s1, s2 = res['exact']
                    st.markdown(f"""<div class="score-card"><div style="color:#DAE025;">SCORE PROBABLE</div><div class="score-display">{th} {s1}-{s2} {ta}</div>
                    <div style="font-size:0.8rem; color:#aaa;">xG Estimés: {res['xg_home']:.2f} - {res['xg_away']:.2f}</div></div>""", unsafe_allow_html=True)
                    
                    # Probabilités (Gauge Chart)
                    ph, pd_, pa = (res['win'], res['draw'], res['loss']) if is_home else (res['loss'], res['draw'], res['win'])
                    
                    fig_g = go.Figure(go.Bar(
                        x=[ph, pd_, pa], y=[''], orientation='h',
                        text=[f"{ph:.0%}", f"{pd_:.0%}", f"{pa:.0%}"], textposition='auto',
                        marker=dict(color=['#2ECC71', '#95A5A6', '#E74C3C'])
                    ))
                    fig_g.update_layout(barmode='stack', height=60, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(visible=False))
                    st.plotly_chart(fig_g, use_container_width=True)
                    
                    # Détection Value Bet
                    # Simulation Cotes (Si pas dispos dans future data)
                    odds_h, odds_d, odds_a = 2.5, 3.2, 2.8 # Placeholders pour démo si pas de cotes réelles
                    val = detect_value_bet(res, odds_h, odds_d, odds_a)
                    if val:
                         st.markdown(f"""<div class="value-bet-box">💎 <b>Value Detectée :</b> {val['sel']} (Cote {val['odd']}) - EV: +{val['ev']*100:.1f}%</div>""", unsafe_allow_html=True)

    # 3. PROCHAINE JOURNÉE (SMART TABLE AVEC CALENDRIER CSV)
    st.markdown("---")
    st.subheader("📅 Analyse Prochaine Journée")
    if not df_cal.empty:
        # Trouver la prochaine journée non jouée
        next_j = df_cal['journee'].min()
        df_next = df_cal[df_cal['journee'] == next_j].copy()
        
        if not df_next.empty:
            rows = []
            for _, row in df_next.iterrows():
                h, a = row['home_team'], row['away_team']
                # Prédiction
                p = predict_match_scientific(df_hist, h, a)
                if p:
                    # Logique de Value simple
                    fav = "Indécis"
                    if p['win'] > 0.5: fav = h
                    elif p['loss'] > 0.5: fav = a
                    
                    rows.append({
                        "Date": row['datetime_match'].strftime('%d/%m %H:%M'),
                        "Match": f"{h} - {a}",
                        "Favori (Modèle)": fav,
                        "Proba. Win": f"{max(p['win'], p['loss']):.0%}",
                        "xG Diff": f"{p['xg_home'] - p['xg_away']:.2f}"
                    })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Aucun match trouvé pour la prochaine journée.")
    else: st.info("Calendrier vide.")

    # 4. CLASSEMENT LIVE
    st.markdown("---")
    st.subheader("🏆 Classement Live")
    if not df_class.empty:
        df_show = df_class.sort_values('match_timestamp').groupby('equipe').last().reset_index()
        df_show = df_show.sort_values('total_points', ascending=False).reset_index(drop=True)
        df_show.index += 1
        st.dataframe(apply_standings_style(df_show[['equipe', 'total_points', 'total_diff', 'total_V', 'total_D']]), use_container_width=True)

elif page == "Classement Prédictif":
    st.markdown('<div class="main-title">🔮 Simulation Monte Carlo</div>', unsafe_allow_html=True)
    st.markdown("Simulation de la fin de saison (500 itérations) basée sur la force xG des équipes.")
    
    if st.button("Lancer la Simulation"):
        with st.spinner("Simulation de la saison en cours..."):
            df_mc = run_monte_carlo(df_curr, df_cal, df_curr)
            st.dataframe(df_mc.style.background_gradient(subset=['Prob. Titre'], cmap='Greens'), use_container_width=True, height=600)