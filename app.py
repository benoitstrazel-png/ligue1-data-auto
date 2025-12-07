import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
import plotly.express as px

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Ligue 1 Data Live", layout="wide", page_icon="⚽")

# --- STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #091C3E;
        color: #DAE025;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.8;
    }
    </style>
""", unsafe_allow_html=True)

# --- CONNEXION BIGQUERY ---
@st.cache_resource
def get_db_client():
    # Récupération des secrets (Configurés dans .streamlit/secrets.toml ou sur le Cloud)
    key_dict = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(key_dict)
    client = bigquery.Client(credentials=creds, project=key_dict["project_id"])
    return client

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data(ttl=600) # Mise en cache pour 10 minutes pour éviter de payer trop cher BQ
def load_data():
    client = get_db_client()
    
    # 1. Récupérer le classement LIVE (Ta vue SQL)
    # On prend la dernière journée disponible pour chaque équipe
    query_classement = """
        SELECT * FROM `ligue1-data.historic_datasets.classement_live`
        WHERE saison = '2024-2025' -- Adapte dynamiquement si besoin
        QUALIFY ROW_NUMBER() OVER(PARTITION BY equipe ORDER BY match_timestamp DESC) = 1
        ORDER BY total_points DESC, total_diff DESC
    """
    df_classement = client.query(query_classement).to_dataframe()
    
    # 2. Récupérer l'historique des matchs pour les courbes
    query_matchs = """
        SELECT * FROM `ligue1-data.historic_datasets.classement_live`
        WHERE saison = '2024-2025'
        ORDER BY date_match
    """
    df_history = client.query(query_matchs).to_dataframe()
    
    return df_classement, df_history

# --- INTERFACE ---
try:
    df_classement, df_history = load_data()
    st.sidebar.success("Données connectées à BigQuery 🟢")
except Exception as e:
    st.error(f"Erreur de connexion BigQuery : {e}")
    st.stop()

# --- EN-TÊTE ---
st.title("⚽ Dashboard Ligue 1 - Live Data")
st.markdown("---")

# --- SIDEBAR & FILTRES ---
equipe_choisie = st.sidebar.selectbox("Choisir une équipe", df_classement['equipe'].sort_values())

# --- DONNÉES DE L'ÉQUIPE ---
stats_equipe = df_classement[df_classement['equipe'] == equipe_choisie].iloc[0]

# Calcul du rang actuel
df_classement['rang'] = range(1, len(df_classement) + 1)
rang_actuel = df_classement[df_classement['equipe'] == equipe_choisie]['rang'].values[0]

# --- KPIS ---
col1, col2, col3, col4 = st.columns(4)

def kpi_card(col, label, value):
    col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)

kpi_card(col1, "Classement Actuel", f"{rang_actuel}e")
kpi_card(col2, "Points", stats_equipe['total_points'])
kpi_card(col3, "Buts Marqués", stats_equipe['total_bp'])
kpi_card(col4, "Diff. de Buts", f"{stats_equipe['total_diff']:+d}") # Le :+d force le signe + ou -

# --- GRAPHIQUES ---
st.markdown("### 📈 Évolution de la saison")

# On filtre l'historique pour l'équipe choisie
history_team = df_history[df_history['equipe'] == equipe_choisie]

# Graphique interactif avec Plotly
fig = px.line(history_team, x='journee_team', y='total_points', 
              title=f"Trajectoire de points - {equipe_choisie}",
              markers=True,
              labels={'journee_team': 'Journée', 'total_points': 'Points Cumulés'})
fig.update_traces(line_color='#DAE025', line_width=4)
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')

st.plotly_chart(fig, use_container_width=True)

# --- TABLEAU DE CLASSEMENT ---
st.markdown("### 🏆 Classement Général")
st.dataframe(
    df_classement[['rang', 'equipe', 'total_points', 'total_V', 'total_N', 'total_D', 'total_diff']],
    hide_index=True,
    use_container_width=True
)