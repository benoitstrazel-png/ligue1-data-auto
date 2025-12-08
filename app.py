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

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Ligue 1 Data Center", layout="wide", page_icon="⚽")

# --- 2. STYLE CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #1A1C23 !important; }
    h1, h2, h3, h4, h5, p, span, div, label, .stDataFrame, .stTable { color: #FFFFFF !important; }
    .main-title {
        font-size: 3rem; font-weight: 900; text-transform: uppercase;
        background: -webkit-linear-gradient(left, #DAE025, #FFFFFF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    .global-card {
        background-color: #091C3E; border-left: 4px solid #DAE025;
        padding: 10px; border-radius: 8px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .global-val { font-size: 1.5rem; font-weight: 800; color: #DAE025; margin: 0; }
    .global-lbl { font-size: 0.8rem; color: #E0E0E0; text-transform: uppercase; }
    .metric-card {
        background-color: #2C3E50; padding: 15px; border-radius: 12px;
        text-align: center; border: 1px solid #444; margin-bottom: 10px;
    }
    .metric-value { font-size: 1.8rem; font-weight: 800; margin: 0; color: #2ECC71; }
    .metric-label { font-size: 0.9rem; color: #CCC; }
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
        font-size: 0.9rem;
    }
    .form-badge {
        display: inline-block; width: 32px; height: 32px; line-height: 32px;
        border-radius: 50%; text-align: center; font-weight: bold;
        color: white !important; margin-right: 4px; font-size: 0.8rem;
        border: 2px solid #1A1C23;
    }
    .win { background-color: #2ECC71; }
    .draw { background-color: #95A5A6; }
    .loss { background-color: #E74C3C; }
    
    /* Tableaux */
    [data-testid="stSidebar"] { background-color: #091C3E !important; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stDataFrame"] { background-color: #2C3E50; }
    
    /* Tooltip */
    .tooltip-icon { font-size: 0.8rem; cursor: help; color: #DAE025; vertical-align: super; }
    </style>
""", unsafe_allow_html=True)

# --- NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Menu", ["Dashboard", "Classement Prédictif"])

st.sidebar.markdown("---")
use_dev_mode = st.sidebar.toggle("🛠️ Mode Développeur", value=False)
TARGET_DATASET = "historic_datasets_dev" if use_dev_mode else "historic_datasets"
if use_dev_mode: st.sidebar.warning(f"⚠️ Source : {TARGET_DATASET}")

betclic_logo = "https://upload.wikimedia.org/wikipedia/commons/3/36/Logo_Betclic.svg"
st.sidebar.markdown(f"""<div style="text-align:center;margin:10px 0 20px 0;"><a href="https://www.betclic.fr" target="_blank"><img src="{betclic_logo}" width="140" style="background-color:white;padding:10px;border-radius:5px;"></a><p style="color:#CCC;font-size:0.8rem;margin-top:5px;">Partenaire Paris Sportifs</p></div>""", unsafe_allow_html=True)

# --- DATA LOADING ---
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
    
    q_curr = f"SELECT * FROM `{p}.{d}.matchs_clean` WHERE season = '{season_name}' ORDER BY date ASC"
    df_curr = client.query(q_curr).to_dataframe()
    
    s_str = "', '".join(history_seasons)
    q_hist = f"SELECT * FROM `{p}.{d}.matchs_clean` WHERE season IN ('{s_str}')"
    df_hist = client.query(q_hist).to_dataframe()
    
    q_class = f"SELECT * FROM `{p}.{d}.classement_live` WHERE saison = '{season_name}' ORDER BY journee_team ASC"
    df_class = client.query(q_class).to_dataframe()
    
    # Récupération Calendrier (Matchs futurs)
    q_cal = f"SELECT * FROM `{p}.{d}.referentiel_calendrier` WHERE datetime_match > CURRENT_TIMESTAMP() ORDER BY datetime_match ASC LIMIT 50"
    try: df_cal = client.query(q_cal).to_dataframe()
    except: df_cal = pd.DataFrame()
    
    for df in [df_curr, df_hist, df_class]:
        if not df.empty:
            for col in ['date', 'match_timestamp']:
                if col in df.columns: df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
    
    return df_curr, df_hist, df_class, df_cal

# --- LOGIQUE PRÉDICTION POISSON ---
def calculate_probabilities(xg_home, xg_away):
    max_goals = 8
    pm = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            pm[i, j] = poisson.pmf(i, xg_home) * poisson.pmf(j, xg_away)
    
    p_win = np.sum(np.tril(pm, -1))
    p_draw = np.sum(np.diag(pm))
    p_loss = np.sum(np.triu(pm, 1))
    
    p_over = 0
    for i in range(max_goals):
        for j in range(max_goals):
            if (i+j) > 2.5: p_over += pm[i, j]
            
    # Most likely exact score
    exact_idx = np.unravel_index(np.argmax(pm, axis=None), pm.shape)
    
    return {'win': p_win, 'draw': p_draw, 'loss': p_loss, 'over25': p_over, 'exact': exact_idx}

def predict_match_advanced(df_history, team_home, team_away):
    if df_history.empty: return None
    
    # 1. Moyennes Ligue (sur la période sélectionnée)
    avg_h = df_history['full_time_home_goals'].mean()
    avg_a = df_history['full_time_away_goals'].mean()
    
    # 2. Stats Equipes
    h_hist = df_history[df_history['home_team'] == team_home]
    a_hist = df_history[df_history['away_team'] == team_away]
    
    if h_hist.empty or a_hist.empty: return None
    
    # Force Attaque / Défense
    att_h = h_hist['full_time_home_goals'].mean() / avg_h
    def_h = h_hist['full_time_away_goals'].mean() / avg_a
    att_a = a_hist['full_time_away_goals'].mean() / avg_a
    def_a = a_hist['full_time_home_goals'].mean() / avg_h
    
    # 3. xG Match
    xg_h = att_h * def_a * avg_h
    xg_a = att_a * def_h * avg_a
    
    # 4. Probabilités détaillées
    return calculate_probabilities(xg_h, xg_a)

# --- AUTRES FONCTIONS MÉTIER ---
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

def apply_standings_style(df):
    def style_rows(row):
        rank = row.name
        if rank <= 4: return 'background-color: rgba(66, 133, 244, 0.6); color: white;' # Bleu C1
        elif rank == 5: return 'background-color: rgba(255, 165, 0, 0.6); color: white;' # Orange C3
        elif rank == 6: return 'background-color: rgba(46, 204, 113, 0.6); color: white;' # Vert C4
        elif rank == 16: return 'background-color: rgba(231, 76, 60, 0.5); color: white;' 
        elif rank == 17: return 'background-color: rgba(231, 76, 60, 0.35); color: white;' 
        elif rank >= 18: return 'background-color: rgba(231, 76, 60, 0.2); color: white;' 
        return ''
    
    df_styled = df.copy()
    if 1 in df_styled.index: df_styled.loc[1, 'equipe'] = "👑 " + df_styled.loc[1, 'equipe']
    return df_styled.style.apply(style_rows, axis=1)

def prepare_calendar_table(df_cal, df_history):
    if df_cal.empty: return pd.DataFrame()
    
    # On prend la prochaine journée disponible
    next_journee = df_cal['journee'].min()
    df_next = df_cal[df_cal['journee'] == next_journee].copy()
    
    rows = []
    for _, row in df_next.iterrows():
        home, away = row['home_team'], row['away_team']
        pred = predict_match_advanced(df_history, home, away)
        
        best_bet, conf = "Indécis", 0.0
        bg_home, bg_away, bg_draw = "", "", ""
        
        if pred:
            p_w, p_d, p_l = pred['win'], pred['draw'], pred['loss']
            p_o25, p_u25 = pred['over25'], 1-pred['over25']
            
            # Gestion couleurs (Vert si > 50%, Orange si > 30% pour nul)
            if p_w > 0.5: bg_home = "background-color: rgba(46, 204, 113, 0.4)"
            if p_l > 0.5: bg_away = "background-color: rgba(46, 204, 113, 0.4)"
            if p_d > 0.30: bg_draw = "background-color: rgba(255, 165, 0, 0.4)"
            
            # Calcul Best Bet Intelligent
            options = {
                f"Victoire {home}": p_w,
                f"Victoire {away}": p_l,
                "Match Nul": p_d,
                "+2.5 Buts": p_o25,
                "-2.5 Buts": p_u25
            }
            # On cherche l'option la plus haute
            best_opt = max(options, key=options.get)
            best_val = options[best_opt]
            
            # Filtre de cohérence Score Exact vs Over/Under
            s1, s2 = pred['exact']
            total_goals_exact = s1 + s2
            
            # Si le score exact est 2-0 (Under) mais que la proba Over est 51%, on évite de conseiller l'Over
            # On ne conseille l'Over que s'il est très probable (>60%)
            if "Buts" in best_opt and best_val < 0.60:
                # On se rabat sur le résultat 1N2 le plus probable
                del options["+2.5 Buts"]; del options["-2.5 Buts"]
                best_opt = max(options, key=options.get)
                best_val = options[best_opt]
            
            best_bet = f"{best_opt} ({best_val*100:.0f}%)"
            
            rows.append({
                "Date": row['datetime_match'].strftime('%d/%m %H:%M'),
                "Domicile": home,
                "Prob. Dom.": f"{p_w*100:.0f}%",
                "Prob. Nul": f"{p_d*100:.0f}%",
                "Prob. Ext.": f"{p_l*100:.0f}%",
                "Extérieur": away,
                "Meilleur Pari 💡": best_bet,
                "_bg_h": bg_home, "_bg_d": bg_draw, "_bg_a": bg_away
            })
            
    return pd.DataFrame(rows)

# --- 6. FILTRES & CHARGEMENT ---
st.sidebar.markdown("---")
st.sidebar.title("🔍 Filtres")
all_seasons = get_seasons_list(TARGET_DATASET) 

# Filtre Saisons: Tout sélectionné par défaut
selected_seasons = st.sidebar.multiselect("Historique (Poisson)", all_seasons, default=all_seasons)
focus_season = sorted(selected_seasons, reverse=True)[0] if selected_seasons else all_seasons[0]

df_curr, df_hist, df_class, df_cal = load_data_complete(focus_season, selected_seasons, TARGET_DATASET)
teams = sorted(df_curr['home_team'].unique())

# =================================================================================
# PAGE 1 : DASHBOARD
# =================================================================================
if page == "Dashboard":
    st.markdown('<div class="main-title">LIGUE 1 DATA CENTER</div>', unsafe_allow_html=True)
    
    # 1. SCORECARDS
    # ... (Code existant inchangé pour les KPIs globaux) ...
    
    st.markdown("---")

    # 3. ANALYSE CLUB
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
        
        # 4. SIMULATION & ANALYSE ADVERSAIRE
        st.subheader("⚔️ Analyse Adversaire", help="Prédictions basées sur l'historique filtré.")
        
        opp_sel = st.selectbox("Adversaire", ["Vue Globale"] + [t for t in teams if t != my_team])
        
        if opp_sel == "Vue Globale":
            # TABLEAU DYNAMIQUE AVEC POISSON SUR CHAQUE LIGNE
            rows = []
            for o in teams:
                if o == my_team: continue
                # Calcul spécifique Home/Away standardisé (sur terrain neutre ou moyenne)
                # Ici on simule un match à domicile pour 'my_team' pour simplifier la vue globale
                res = predict_match_advanced(df_hist, my_team, o)
                if res:
                    rows.append({'Adversaire': o, 'Prob. Victoire': res['win'], 'Prob. Nul': res['draw'], 'Prob. Défaite': res['loss']})
            
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index('Adversaire').style.format("{:.1f}%").background_gradient(subset=['Prob. Victoire'], cmap='Greens'), use_container_width=True)
                
        else:
            c_opt, c_res = st.columns([1, 2])
            with c_opt:
                loc = st.radio("Lieu", ["Domicile", "Extérieur"])
                is_home = "Domicile" in loc
            
            # Définition Home/Away pour le calcul
            th, ta = (my_team, opp_sel) if is_home else (opp_sel, my_team)
            
            # Calcul Poisson Spécifique
            res = predict_match_advanced(df_hist, th, ta)
            
            if res:
                with c_res:
                    s1, s2 = res['exact']
                    # Affichage Score
                    st.markdown(f"""
                        <div class="score-card">
                            <div style="color:#DAE025;">SCORE PROBABLE</div>
                            <div class="score-display">{th} {s1}-{s2} {ta}</div>
                            <div style="font-size:0.8rem; color:#AAA;">xG: {res['exact'][0]} - {res['exact'][1]} (Proba Exacte)</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Logique Cohérence Paris
                    p_w = res['win'] if is_home else res['loss'] # Proba Victoire Mon Equipe
                    p_l = res['loss'] if is_home else res['win'] # Proba Défaite
                    p_d = res['draw']
                    
                    # 1N2
                    if p_w > max(p_l, p_d): opt_1n2, p_1n2 = f"Victoire {my_team}", p_w
                    elif p_l > max(p_w, p_d): opt_1n2, p_1n2 = f"Victoire {opp_sel}", p_l
                    else: opt_1n2, p_1n2 = "Match Nul", p_d
                    
                    col_1n2 = "#2ECC71" if p_1n2 > 0.5 else "#F1C40F"
                    
                    # Buts (Seuil 60% pour éviter incohérence avec score exact)
                    if res['over25'] > 0.60: opt_goal, p_goal = "+2.5 Buts", res['over25']
                    elif (1 - res['over25']) > 0.60: opt_goal, p_goal = "-2.5 Buts", 1-res['over25']
                    else: opt_goal, p_goal = "Pas de tendance franche", 0.0
                    
                    c_adv1, c_adv2 = st.columns(2)
                    c_adv1.markdown(f"""<div class="advice-box" style="border-color:{col_1n2}"><span style="font-weight:bold;color:{col_1n2}">{opt_1n2}</span> <span style="color:#EEE">({p_1n2*100:.1f}%)</span></div>""", unsafe_allow_html=True)
                    if p_goal > 0:
                        c_adv2.markdown(f"""<div class="advice-box" style="border-color:#3498DB"><span style="font-weight:bold;color:#3498DB">{opt_goal}</span> <span style="color:#EEE">({p_goal*100:.1f}%)</span></div>""", unsafe_allow_html=True)

    st.markdown("---")
    
    # 5. CALENDRIER INTELLIGENT
    st.subheader("📅 Prochaine Journée (Smart Analysis)", help="Analyse automatique des rencontres à venir.")
    
    df_smart_cal = prepare_calendar_table(df_cal, df_hist)
    
    if not df_smart_cal.empty:
        # Application des styles conditionnels (Vert/Orange)
        def style_cal(row):
            return [
                '', '', # Date, Dom
                row['_bg_h'], # Prob Dom
                row['_bg_d'], # Prob Nul
                row['_bg_a'], # Prob Ext
                '', '', # Ext, Best Bet
                '', '', '' # Colonnes cachées
            ]
        
        st.dataframe(
            df_smart_cal.style.apply(style_cal, axis=1),
            column_config={
                "_bg_h": None, "_bg_d": None, "_bg_a": None # Cacher les colonnes techniques
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Aucun match futur trouvé dans le calendrier référentiel.")

    st.markdown("---")
    st.subheader("🏆 Classement Live")
    
    df_show = df_snap[['rang', 'equipe', 'total_points', 'total_diff', 'total_V', 'total_N', 'total_D']].set_index('rang').sort_index()
    st.dataframe(apply_standings_style(df_show), use_container_width=True)

# =================================================================================
# PAGE 2 : CLASSEMENT PRÉDICTIF (Reste inchangé)
# =================================================================================
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