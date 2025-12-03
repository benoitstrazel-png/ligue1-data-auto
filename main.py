import requests
import pandas as pd
import time
import os
import sys ### MODIF : Nécessaire pour arrêter le script proprement

# ------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------

### MODIF 1 : RÉCUPÉRATION SÉCURISÉE DE LA CLÉ
# On ne met pas la clé en dur. On la cherche dans l'environnement du serveur.
try:
    API_KEY = os.environ["API_FOOTBALL_KEY"]
except KeyError:
    print("❌ Erreur : La variable d'environnement API_FOOTBALL_KEY est introuvable.")
    print("Assure-toi de l'avoir ajoutée dans les 'Secrets' de GitHub.")
    sys.exit(1) # Arrête le script immédiatement

base_url = "https://v3.football.api-sports.io/fixtures/players"
headers = {"x-apisports-key": API_KEY}

# Fichiers
FILENAME_STATS = "stats_joueurs_L1_2023.csv"
FILENAME_CALENDRIER = "calendrier_L1_2023.csv" ### MODIF : Nom de ton fichier calendrier
LIMIT_REQUESTS = 80 

# ------------------------------------------------------------------
# 2. CHARGEMENT DU CALENDRIER (CRUCIAL)
# ------------------------------------------------------------------

### MODIF 2 : CHARGEMENT DU DATAFRAME DEPUIS LE CSV
# Le script doit lire le fichier calendrier présent dans le dossier
if os.path.exists(FILENAME_CALENDRIER):
    print(f"📖 Lecture du calendrier : {FILENAME_CALENDRIER}")
    df_calendrier = pd.read_csv(FILENAME_CALENDRIER)
else:
    print(f"❌ Erreur : Le fichier {FILENAME_CALENDRIER} est introuvable.")
    print("Tu dois générer ce fichier (Etape précédente) et le mettre sur GitHub.")
    sys.exit(1)

# On filtre uniquement les matchs TERMINÉS (FT)
# Note : on s'assure que 'fixture_id' est bien reconnu (parfois lu comme float)
matchs_termines = df_calendrier[df_calendrier['status'] == 'FT']['fixture_id'].astype(int).unique().tolist()

# Vérification de ce qu'on a déjà en local (Stats joueurs)
ids_deja_recuperes = []
if os.path.exists(FILENAME_STATS):
    df_existant = pd.read_csv(FILENAME_STATS)
    ids_deja_recuperes = df_existant['fixture_id'].unique().tolist()
    print(f"📂 Fichier stats existant trouvé : {len(ids_deja_recuperes)} matchs déjà stockés.")
else:
    print("📂 Aucun fichier stats existant. On commence à zéro.")

# On calcule la différence
ids_a_traiter = [id_ for id_ in matchs_termines if id_ not in ids_deja_recuperes]
print(f"🎯 Il reste {len(ids_a_traiter)} matchs à récupérer.")

# ------------------------------------------------------------------
# 3. BOUCLE DE RÉCUPÉRATION (BATCH)
# ------------------------------------------------------------------
new_data = []
count = 0

if len(ids_a_traiter) == 0:
    print("✅ Tout est à jour ! Reviens après les prochains matchs.")

else:
    print(f"🚀 Démarrage du batch (Max {LIMIT_REQUESTS} appels)...")
    
    for fixture_id in ids_a_traiter:
        if count >= LIMIT_REQUESTS:
            print("🛑 Limite de sécurité atteinte pour aujourd'hui.")
            break
            
        print(f"[{count+1}/{LIMIT_REQUESTS}] Récupération match ID {fixture_id}...", end="\r")
        
        try:
            params = {"fixture": fixture_id}
            response = requests.get(base_url, headers=headers, params=params)
            data = response.json()
            
            if 'response' in data and len(data['response']) > 0:
                for team_data in data['response']:
                    team_id = team_data['team']['id']
                    team_name = team_data['team']['name']
                    
                    for player in team_data['players']:
                        # On vérifie que les stats existent
                        if len(player['statistics']) > 0:
                            stats = player['statistics'][0]
                            
                            row = {
                                'fixture_id': fixture_id,
                                'team_id': team_id,
                                'team_name': team_name,
                                'player_id': player['player']['id'],
                                'player_name': player['player']['name'],
                                'minutes_played': stats['games']['minutes'],
                                'rating': stats['games']['rating'],
                                'goals': stats['goals']['total'] or 0,
                                'assists': stats['goals']['assists'] or 0,
                                'shots_total': stats['shots']['total'] or 0,
                                'shots_on': stats['shots']['on'] or 0,
                                'passes_total': stats['passes']['total'] or 0,
                                'passes_accuracy': stats['passes']['accuracy'],
                                'tackles': stats['tackles']['total'] or 0,
                                'duels_total': stats['duels']['total'] or 0,
                                'duels_won': stats['duels']['won'] or 0,
                                'dribbles_success': stats['dribbles']['success'] or 0,
                                'fouls_drawn': stats['fouls']['drawn'] or 0,
                                'fouls_committed': stats['fouls']['committed'] or 0,
                                'cards_yellow': stats['cards']['yellow'] or 0,
                                'cards_red': stats['cards']['red'] or 0
                            }
                            new_data.append(row)
            
            count += 1
            time.sleep(0.2) 
            
        except Exception as e:
            print(f"\n❌ Erreur sur match {fixture_id}: {e}")

    # ------------------------------------------------------------------
    # 4. SAUVEGARDE
    # ------------------------------------------------------------------
    if len(new_data) > 0:
        df_new = pd.DataFrame(new_data)
        
        if not os.path.exists(FILENAME_STATS):
            df_new.to_csv(FILENAME_STATS, index=False)
        else:
            df_new.to_csv(FILENAME_STATS, mode='a', header=False, index=False)
            
        print(f"\n✅ {len(df_new)} lignes ajoutées à {FILENAME_STATS}")
    else:
        print("\n⚠️ Aucune donnée récupérée lors de ce batch.")