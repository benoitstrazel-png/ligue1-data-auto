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
st.set_page_config(page_title="Ligue 1 Science & Bet", layout="wide", page_icon="🧪")

# --- 2. STYLE CSS AVANCÉ (SCIENTIFIC THEME) ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117 !important; }
    h1, h2, h3, h4, h5, p, span, div, label, .stDataFrame, .stTable, li { color: #E0E0E0 !important; }
    
    /* Titre Principal */
    .main-title {
        font-size: 3rem; font-weight: 900; text-transform: uppercase;
        background: -webkit-linear-gradient(left, #6C5CE7, #00CEC9);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    
    /* Cartes KPI */
    .kpi-card {
        background-color: #1A1C24; border: 1px solid #333; border-radius: 8px;
        padding: 15px; text-align: center; margin-bottom: 10px;
    }
    .kpi-val { font-size: 1.5rem; font-weight: bold; color: #00CEC9; }
    .kpi-lbl { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: #888; }
    
    /* Value Bet Highlights */
    .value-bet {
        border: 1px solid #00CEC9; background-color: rgba(0, 206, 201, 0.1);
        padding: 10px; border-radius: 5px; margin-top: 5px;
    }
    .ev-positive { color: #00CEC9 !important; font-weight: bold; }
    .ev-negative { color: #FF7675 !important; }
    
    /* Tableaux */
    [data-testid="stDataFrame"] { background-color: #1A1C24; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- 3. NAVIGATION ---
st.sidebar.title("🔬 Labo Foot")
page = st.sidebar.radio("Navigation", ["Analyse Scientifique", "Simulateur Monte Carlo"])
st.sidebar.markdown("---")
use_dev_mode = st.sidebar.toggle("🛠️ Mode Dev", value=False)
TARGET_DATASET = "historic_datasets_dev" if use_dev_mode else "historic_datasets"

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
    
    # Récupération des données
    q_curr = f"SELECT * FROM `{p}.{d}.matchs_clean` WHERE season = '{season_name}' ORDER BY date ASC"
    df_curr = client.query(q_curr).to_dataframe()
    
    s_str = "', '".join(history_seasons)
    q_hist = f"SELECT * FROM `{p}.{d}.matchs_clean` WHERE season IN ('{s_str}')"
    df_hist = client.query(q_hist).to_dataframe()
    
    q_cal = f"SELECT * FROM `{p}.{d}.referentiel_calendrier` WHERE datetime_match > CURRENT_TIMESTAMP() ORDER BY datetime_match ASC LIMIT 100"
    try: df_cal = client.query(q_cal).to_dataframe()
    except: df_cal = pd.DataFrame()
    
    # Nettoyage & Conversion
    for df in [df_curr, df_hist]:
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)

    # --- FEATURE ENGINEERING (SCIENTIFIQUE) ---
    # Création du Proxy xG si les données de tirs existent
    # Formule approx: 0.30 x Tirs Cadrés + 0.07 x Tirs Non Cadrés
    if 'home_shots_on_target' in df_hist.columns:
        for df in [df_curr, df_hist]:
            df['home_xg_proxy'] = (df['home_shots_on_target'] * 0.30) + ((df['home_shots'] - df['home_shots_on_target']) * 0.07)
            df['away_xg_proxy'] = (df['away_shots_on_target'] * 0.30) + ((df['away_shots'] - df['away_shots_on_target']) * 0.07)
    else:
        # Fallback si pas de stats détaillées
        for df in [df_curr, df_hist]:
            df['home_xg_proxy'] = df['full_time_home_goals']
            df['away_xg_proxy'] = df['full_time_away_goals']

    return df_curr, df_hist, df_cal

# --- 5. MOTEUR STATISTIQUE & MODÉLISATION ---

def calculate_ema_strength(df, team, is_home, metric='xg_proxy', window=10, alpha=0.3):
    """
    Calcule la force offensive/défensive pondérée par le temps (Exponential Moving Average).
    Les matchs récents ont plus de poids.
    """
    # Filtrer les matchs de l'équipe
    team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('date')
    
    if team_matches.empty: return 1.0 # Neutre par défaut

    values = []
    for _, row in team_matches.iterrows():
        # Si on cherche la force offensive (ATT)
        if metric == 'attack':
            val = row['home_xg_proxy'] if row['home_team'] == team else row['away_xg_proxy']
        # Si on cherche la force défensive (DEF - buts encaissés)
        else: 
            val = row['away_xg_proxy'] if row['home_team'] == team else row['home_xg_proxy']
        values.append(val)
    
    # Calcul EMA (Pandas le fait très bien)
    return pd.Series(values).ewm(alpha=alpha, adjust=False).mean().iloc[-1]

def get_league_averages(df):
    """Calcule les moyennes globales de la ligue pour normalisation"""
    avg_home_xg = df['home_xg_proxy'].mean()
    avg_away_xg = df['away_xg_proxy'].mean()
    return avg_home_xg, avg_away_xg

def predict_match_scientific(df_history, home_team, away_team):
    """
    Modèle Poisson Hybride (Pondéré par la forme récente + xG Proxy)
    """
    if df_history.empty: return None

    # 1. Moyennes Ligue
    avg_h, avg_a = get_league_averages(df_history)
    
    # 2. Forces Pondérées (EMA)
    # Domicile (Attaque vs Moyenne Ligue, Défense vs Moyenne Ligue)
    home_att = calculate_ema_strength(df_history, home_team, True, 'attack') / avg_h
    home_def = calculate_ema_strength(df_history, home_team, True, 'defense') / avg_a
    
    # Extérieur
    away_att = calculate_ema_strength(df_history, away_team, False, 'attack') / avg_a
    away_def = calculate_ema_strength(df_history, away_team, False, 'defense') / avg_h
    
    # 3. Calcul Lambdas (Buts attendus pour ce match spécifique)
    lambda_home = home_att * away_def * avg_h
    lambda_away = away_att * home_def * avg_a
    
    # 4. Distribution de Poisson
    max_goals = 8
    pm = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            pm[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
            
    p_win = np.sum(np.tril(pm, -1))
    p_draw = np.sum(np.diag(pm))
    p_loss = np.sum(np.triu(pm, 1))
    
    # Expected Value Calculation (si cotes dispo - placeholder ici)
    return {
        'xg_home_predicted': lambda_home,
        'xg_away_predicted': lambda_away,
        'probs': {'H': p_win, 'D': p_draw, 'A': p_loss}
    }

def detect_value_bet(prediction, odds_h, odds_d, odds_a):
    """Calcule l'Expected Value (EV)"""
    ev_h = (prediction['probs']['H'] * odds_h) - 1
    ev_d = (prediction['probs']['D'] * odds_d) - 1
    ev_a = (prediction['probs']['A'] * odds_a) - 1
    
    best_ev = max(ev_h, ev_d, ev_a)
    selection = ""
    
    if best_ev == ev_h: selection, odd, prob = "Victoire Domicile", odds_h, prediction['probs']['H']
    elif best_ev == ev_d: selection, odd, prob = "Nul", odds_d, prediction['probs']['D']
    else: selection, odd, prob = "Victoire Extérieur", odds_a, prediction['probs']['A']
    
    return {
        'selection': selection,
        'ev': best_ev, # Pourcentage de rendement espéré (ex: 0.05 = 5%)
        'odd': odd,
        'prob': prob
    }

# --- 6. SIMULATION MONTE CARLO ---

def run_monte_carlo(df_played, df_cal, df_history, n_simulations=100):
    """
    Simule la fin de saison N fois pour obtenir des probabilités de classement.
    """
    teams = sorted(list(set(df_played['home_team'].unique()) | set(df_played['away_team'].unique())))
    
    # Classement actuel (points initiaux)
    current_points = {t: 0 for t in teams}
    # On recalcule les points réels rapidement
    for _, r in df_played.iterrows():
        res = r['full_time_result']
        if res == 'H': current_points[r['home_team']] += 3
        elif res == 'A': current_points[r['away_team']] += 3
        elif res == 'D':
            current_points[r['home_team']] += 1
            current_points[r['away_team']] += 1
            
    # Pré-calcul des lambdas pour les matchs restants (pour aller vite)
    # On stocke les probas H/D/A pour chaque match du calendrier
    future_matches = []
    
    # On ne prend que les matchs futurs de la saison en cours
    # Note: df_cal contient tout le futur. Il faudrait filtrer pour la saison en cours idéalement.
    # Ici on suppose que df_cal est propre.
    
    avg_h, avg_a = get_league_averages(df_history)
    team_stats = {}
    for t in teams:
        team_stats[t] = {
            'att_h': calculate_ema_strength(df_history, t, True, 'attack') / avg_h,
            'def_h': calculate_ema_strength(df_history, t, True, 'defense') / avg_a,
            'att_a': calculate_ema_strength(df_history, t, False, 'attack') / avg_a,
            'def_a': calculate_ema_strength(df_history, t, False, 'defense') / avg_h
        }

    for _, row in df_cal.iterrows():
        h, a = row['home_team'], row['away_team']
        if h not in teams or a not in teams: continue
        
        # Calcul lambdas rapide
        lh = team_stats[h]['att_h'] * team_stats[a]['def_a'] * avg_h
        la = team_stats[a]['att_a'] * team_stats[h]['def_h'] * avg_a
        
        # Probas simplifiées (pas besoin de Poisson complet pour 1000 sims, trop lent)
        # Approximation : Win si diff > 0.3, Draw si proche
        # Pour Monte Carlo précis, on tire un score aléatoire via Poisson
        future_matches.append({'h': h, 'a': a, 'lh': lh, 'la': la})

    results = {t: {'champion': 0, 'top4': 0, 'relegation': 0} for t in teams}
    
    progress_bar = st.progress(0)
    
    for i in range(n_simulations):
        sim_points = current_points.copy()
        
        for m in future_matches:
            # Tirage aléatoire des buts
            g_h = np.random.poisson(m['lh'])
            g_a = np.random.poisson(m['la'])
            
            if g_h > g_a: sim_points[m['h']] += 3
            elif g_a > g_h: sim_points[m['a']] += 3
            else:
                sim_points[m['h']] += 1
                sim_points[m['a']] += 1
        
        # Classement final de cette simulation
        # Tri par points (on ignore diff buts pour simplicité ici, mais crucial en vrai)
        sorted_teams = sorted(sim_points.items(), key=lambda x: x[1], reverse=True)
        
        # Enregistrement Stats
        results[sorted_teams[0][0]]['champion'] += 1
        for j in range(4): # Top 4
            results[sorted_teams[j][0]]['top4'] += 1
        for j in range(16, 18): # Relégation (Ligue 1 à 18 clubs)
            if j < len(sorted_teams):
                results[sorted_teams[j][0]]['relegation'] += 1
        
        if i % 10 == 0: progress_bar.progress(i / n_simulations)
            
    progress_bar.empty()
    
    # Conversion en DataFrame %
    df_res = pd.DataFrame(results).T
    df_res = df_res / n_simulations * 100
    df_res = df_res.sort_values('champion', ascending=False)
    return df_res

# --- 7. INTERFACE GRAPHIQUE ---

all_seasons = get_seasons_list(TARGET_DATASET)
selected_seasons = st.sidebar.multiselect("Données d'entrainement", all_seasons, default=all_seasons[:3]) # Par défaut 3 dernières
focus_season = all_seasons[0] # Saison courante

df_curr, df_hist, df_cal = load_data_complete(focus_season, selected_seasons, TARGET_DATASET)

if page == "Analyse Scientifique":
    st.markdown('<div class="main-title">Analyse Statistique Avancée</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Paramètres du Match")
        teams = sorted(df_curr['home_team'].unique())
        team_h = st.selectbox("Domicile", teams, index=0)
        team_a = st.selectbox("Extérieur", teams, index=1)
        
        # Stats contextuelles
        st.markdown("---")
        st.markdown("#### Indicateurs Clés (Forme Récente)")
        
        # Calcul xG Proxy Moyen (5 derniers matchs)
        xh = calculate_ema_strength(df_curr, team_h, True, 'attack')
        xa = calculate_ema_strength(df_curr, team_a, False, 'attack')
        
        st.metric(f"xG Force {team_h}", f"{xh:.2f}", help="Basé sur l'EMA des tirs cadrés")
        st.metric(f"xG Force {team_a}", f"{xa:.2f}", help="Basé sur l'EMA des tirs cadrés")
        
    with col2:
        if st.button("Lancer la Prédiction Scientifique 🚀", type="primary"):
            pred = predict_match_scientific(df_curr, team_h, team_a)
            
            if pred:
                ph, pd_, pa = pred['probs']['H'], pred['probs']['D'], pred['probs']['A']
                
                # Jauge Probabilités
                fig_gauge = go.Figure(go.Bar(
                    x=[ph, pd_, pa],
                    y=['Probabilités'],
                    orientation='h',
                    text=[f"{team_h} {ph:.0%}", f"Nul {pd_:.0%}", f"{team_a} {pa:.0%}"],
                    textposition='auto',
                    marker=dict(color=['#00b894', '#ffeaa7', '#ff7675'])
                ))
                fig_gauge.update_layout(barmode='stack', height=100, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False, showticklabels=False))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Expected Goals
                c1, c2 = st.columns(2)
                c1.markdown(f"""<div class="kpi-card"><div class="kpi-val">{pred['xg_home_predicted']:.2f}</div><div class="kpi-lbl">xG Attendus {team_h}</div></div>""", unsafe_allow_html=True)
                c2.markdown(f"""<div class="kpi-card"><div class="kpi-val">{pred['xg_away_predicted']:.2f}</div><div class="kpi-lbl">xG Attendus {team_a}</div></div>""", unsafe_allow_html=True)
                
                # Analyse de Value (Simulation de cotes fictives pour l'exemple)
                # Dans un vrai cas, on ferait un merge avec df_curr['avg_home_win_odds'] etc.
                st.subheader("💸 Détection de Value (Exemple)")
                st.info("Comparaison automatique avec les cotes moyennes du marché (si disponibles)")
                
                # Récupération des cotes moyennes SI le match existe déjà dans la base
                match_data = df_curr[((df_curr['home_team'] == team_h) & (df_curr['away_team'] == team_a)) | ((df_curr['home_team'] == team_a) & (df_curr['away_team'] == team_h))]
                
                if not match_data.empty:
                    last_m = match_data.iloc[-1]
                    # On prend les cotes Pinnacle ou Moyenne
                    oh = last_m.get('pinnacle_home_win_odds', last_m.get('avg_home_win_odds', 0))
                    od = last_m.get('pinnacle_draw_odds', last_m.get('avg_draw_odds', 0))
                    oa = last_m.get('pinnacle_away_win_odds', last_m.get('avg_away_win_odds', 0))
                    
                    if oh > 1:
                        val_res = detect_value_bet(pred, oh, od, oa)
                        color_ev = "ev-positive" if val_res['ev'] > 0 else "ev-negative"
                        st.markdown(f"""
                        <div class="value-bet">
                            <strong>Marché :</strong> 1 ({oh}) | N ({od}) | 2 ({oa})<br>
                            <strong>Modèle :</strong> 1 ({1/ph:.2f}) | N ({1/pd_:.2f}) | 2 ({1/pa:.2f})<br>
                            <hr>
                            <span class="{color_ev}">EV (Rendement) : {val_res['ev']*100:.1f}%</span> sur <strong>{val_res['selection']}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("Pas de cotes disponibles pour ce match.")
                else:
                    st.write("Match futur ou non trouvé dans la base pour récupérer les cotes.")

elif page == "Simulateur Monte Carlo":
    st.markdown('<div class="main-title">🔮 Monte Carlo Season</div>', unsafe_allow_html=True)
    st.markdown("""
    Cette simulation joue les matchs restants **1000 fois** en utilisant les forces offensives/défensives pondérées (xG Proxy).
    Cela permet de quantifier l'incertitude.
    """)
    
    if st.button("Lancer la Simulation (1000 itérations)"):
        with st.spinner("Calcul des probabilités en cours..."):
            df_mc = run_monte_carlo(df_curr, df_cal, df_curr, n_simulations=1000)
            
            st.dataframe(
                df_mc.style.format("{:.1f}%")
                .background_gradient(subset=['champion'], cmap='Greens')
                .background_gradient(subset=['top4'], cmap='Blues')
                .background_gradient(subset=['relegation'], cmap='Reds'),
                use_container_width=True,
                height=800
            )
            
            # Graphique
            st.subheader("Probabilité de titre")
            fig = px.bar(df_mc[df_mc['champion']>0], y='champion', title="Favoris pour le titre (%)")
            st.plotly_chart(fig, use_container_width=True)