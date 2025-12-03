import requests
import pandas as pd
import time
import os
import sys

# ------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------

# Récupération sécurisée de la clé
try:
    API_KEY = os.environ["API_FOOTBALL_KEY"]
except KeyError:
    print("❌ Erreur : La variable d'environnement API_FOOTBALL_KEY est introuvable.")
    sys.exit(1)

base_url = "https://v3.football.api-sports.io/fixtures/players"
headers = {"x-apisports-key": API_KEY}

# NOMS DES FICHIERS (Doivent être identiques à ceux sur GitHub)
FILENAME_STATS = "stats_joueurs_L1_2023.csv"
# ATTENTION : Je mets ici le nom du fichier que tu as uploadé. 
# Si sur GitHub il s'appelle autrement, change cette ligne !
FILENAME_CALENDRIER = "ligue1_history.csv" 

LIMIT_REQUESTS = 80 

# ------------------------------------------------------------------
# 2. CHARGEMENT DU CALENDRIER
# ------------------------------------------------------------------
print(f"🔍 Vérification du fichier calendrier : {FILENAME_CALENDRIER}")

if os.path.exists(FILENAME_CALENDRIER):
    print(f"✅ Fichier trouvé. Lecture en cours...")
    df_calendrier = pd.read_csv(FILENAME_CALENDRIER)
else:
    print(f"❌ ERREUR FATALE : Le fichier '{FILENAME_CALENDRIER}' est introuvable au chemin : {os.getcwd()}")
    print("👉 Vérifie que le fichier est bien à la racine du dépôt GitHub (pas dans un dossier).")
    print("👉 Vérifie que le nom est EXACTEMENT le même (Majuscules/Minuscules).")
    sys.exit(1)

# On filtre uniquement les matchs TERMINÉS (FT)
# On convertit 'fixture_id' en entier pour éviter les erreurs de format (123.0 vs 123)
matchs_termines = df_calendrier[df_calendrier['status'] == 'FT']['fixture_id'].astype(int).unique().tolist()

# ------------------------------------------------------------------
# 3. PRÉPARATION DU BATCH
# ------------------------------------------------------------------
ids_deja_recuperes = []
if os.path.exists(FILENAME_STATS):
    df_existant = pd.read_csv(FILENAME_STATS)
    ids_deja_recuperes = df_existant['fixture_id'].unique().tolist()
    print(f"📂 Fichier stats existant trouvé : {len(ids_deja_recuperes)} matchs déjà stockés.")
else:
    print("📂 Aucun fichier stats existant. Création prévue.")

ids_a_traiter = [id_ for id_ in matchs_termines if id_ not in ids_deja_recuperes]
print(f"🎯 Il reste {len(ids_a_traiter)} matchs à récupérer.")

# ------------------------------------------------------------------
# 4. BOUCLE DE RÉCUPÉRATION
# ------------------------------------------------------------------
new_data = []
count = 0

if len(ids_a_traiter) == 0:
    print("✅ Tout est à jour !")
else:
    print(f"🚀 Démarrage du batch (Max {LIMIT_REQUESTS} appels)...")
    
    for fixture_id in ids_a_traiter:
        if count >= LIMIT_REQUESTS:
            print("🛑 Limite atteinte.")
            break
            
        print(f"[{count+1}/{LIMIT_REQUESTS}] Match ID {fixture_id}...", end="\r")
        
        try:
            params = {"fixture": fixture_id}
            response = requests.get(base_url, headers=headers, params=params)
            data = response.json()
            
            if 'response' in data and len(data['response']) > 0:
                for team_data in data['response']:
                    team_id = team_data['team']['id']
                    team_name = team_data['team']['name']
                    
                    for player in team_data['players']:
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
                                'shots_on': stats['shots']['on'] or 0
                            }
                            new_data.append(row)
            count += 1
            time.sleep(0.2)
        except Exception as e:
            print(f"\n❌ Erreur match {fixture_id}: {e}")

    # SAUVEGARDE
    if len(new_data) > 0:
        df_new = pd.DataFrame(new_data)
        if not os.path.exists(FILENAME_STATS):
            df_new.to_csv(FILENAME_STATS, index=False)
        else:
            df_new.to_csv(FILENAME_STATS, mode='a', header=False, index=False)
        print(f"\n✅ {len(df_new)} lignes sauvegardées.")
