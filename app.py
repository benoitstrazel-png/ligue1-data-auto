import streamlit as st
import pandas as pd

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Ligue 1 Data", layout="wide", page_icon="⚽")

# --- STYLE CSS (Couleurs Ligue 1) ---
st.markdown("""
    <style>
    .stApp { background-color: #F5F5F5; }
    h1, h2, h3 { color: #091C3E; }
    .metric-card { background-color: #091C3E; color: #DAE025; padding: 20px; border-radius: 10px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- CHARGEMENT DES DONNÉES (Depuis GitHub) ---
@st.cache_data # Garde les données en mémoire pour aller vite
def load_data():
    # Remplace par l'URL "RAW" de ton fichier sur GitHub
    url = "https://raw.githubusercontent.com/benoitstrazel-png/ligue1-data-auto/refs/heads/main/ligue1_history.csv"
    df = pd.read_csv(url)
    return df

try:
    df = load_data()
    st.sidebar.success("Données chargées !")
except:
    st.sidebar.warning("Données non trouvées (Mode Démo)")
    # Données fictives pour l'exemple si le CSV n'est pas encore là
    df = pd.DataFrame({'home_team': ['PSG', 'OM', 'Lens', 'Lille'], 'goals': [45, 30, 28, 35]})

# --- EN-TÊTE ---
st.title("⚽ Dashboard Ligue 1")
st.markdown("---")

# --- FILTRES ---
# Changed from 'team_name' to 'home_team' to match the column that exists in the DataFrame
equipe = st.sidebar.selectbox("Choisir une équipe", df['home_team'].unique())

# --- STATS CLÉS (KPIs) ---
# On filtre sur l'équipe choisie
subset = df[df['home_team'] == equipe]
total_buts = subset['goals'].sum()

col1, col2, col3 = st.columns(3)
col1.markdown(f'<div class="metric-card"><h3>Buts Marqués</h3><h1>{total_buts}</h1></div>', unsafe_allow_html=True)
col2.metric("Classement (Simulé)", "3ème", "+1 place")
col3.metric("Forme", "V-V-N-D-V")

# --- GRAPHIQUES ---
st.markdown("### 📊 Performance Offensive")
# Changed from 'team_name' to 'home_team' to match the column that exists in the DataFrame
st.bar_chart(df.set_index('home_team')['goals'])
