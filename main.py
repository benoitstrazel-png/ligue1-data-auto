import requests
import pandas as pd
import time
import os
import sys

# ------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------

# Récupération de la clé API
try:
    API_KEY = os.environ["API_FOOTBALL_KEY"]
except KeyError:
    print("❌ Erreur : La variable d'environnement API_FOOTBALL_KEY est introuvable.")
    sys.exit(1)

# Configuration API
headers = {"x-apisports-key": API_KEY}
LEAGUE_ID = 61  # Ligue 1
SEASON = 2023   # Saison 2023-2024

# Configuration Fichier de Sortie
FILENAME_STATS = "stats_joueurs_L1_2023.csv"
LIMIT_REQUESTS = 80 # Nombre max de matchs à scrapper par jour (garde une marge)

# ------------------------------------------------------------------
# 2. RÉCUPÉRATION DU CALENDRIER (Directement depuis l'API)
# ------------------------------------------------------------------
print("📡 Récupération du calendrier officiel depuis l'API...")
url_fixtures = "https://v3.football.api-sports.io/fixtures"
params_fixtures = {
    "league": str(LEAGUE_ID),
    "season": str(SEASON)
}

try:
    # Cette requête coûte 1 crédit API
    resp = requests.get(url_fixtures, headers=headers, params=params_fixtures)
    data_cal = resp.json()
    
    if "response" not in data_cal:
        print(f"❌ Erreur API Calendrier : {data_cal}")
        sys.exit(1)
        
    # On extrait uniquement les IDs des matchs TERMINÉS (Status = FT)
    matchs_termines_ids = []
    for match in data_cal["response"]:
        if match["fixture"]["status"]["short"] == "FT":
            matchs_termines_ids.append(match["fixture"]["id"])
            
    print(f"✅ Calendrier récupéré : {len(matchs_termines_ids)} matchs terminés trouvés sur la saison.")

except Exception as e:
    print(f"❌ Erreur fatale lors de la récupération du calendrier : {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# 3. FILTRAGE (Ce qu'il reste à faire)
# ------------------------------------------------------------------
ids_deja_recuperes = []

# Si le fichier de stats existe déjà, on regarde ce qu'il y a dedans
if os.path.exists(FILENAME_STATS):
    try:
        df_existant = pd.read_csv(FILENAME_STATS)
        # On vérifie que la colonne fixture_id existe bien
        if 'fixture_id' in df_existant.columns:
            ids_deja_recuperes = df_existant['fixture_id'].unique().tolist()
            print(f"📂 Reprise : {len(ids_deja_recuperes)} matchs déjà stockés dans le CSV.")
        else:
            print("⚠️ Le fichier CSV existe mais semble vide ou corrompu. On reprend à zéro.")
    except pd.errors.EmptyDataError:
        print("⚠️ Le fichier CSV existe mais est vide. On reprend à zéro.")
else:
    print("📂 Aucun fichier stats trouvé. Création d'un nouveau fichier.")

# On ne garde que les matchs qu'on n'a PAS encore
ids_a_traiter = [mid for mid in matchs_termines_ids if mid not in ids_deja_recuperes]
ids_a_traiter.sort() # On trie pour faire les matchs dans l'ordre

print(f"🎯 Il reste {len(ids_a_traiter)} matchs à récupérer.")

# ------------------------------------------------------------------
# 4. BOUCLE DE RÉCUPÉRATION DES JOUEURS
# ------------------------------------------------------------------
url_players = "https://v3.football.api-sports.io/fixtures/players"
new_data = []
count = 0

if len(ids_a_traiter) == 0:
    print("✅ Tout est à jour ! Aucune action nécessaire.")
else:
    print(f"🚀 Démarrage du batch (Max {LIMIT_REQUESTS} requêtes)...")
    
    for fixture_id in ids_a_traiter:
        # Sécurité pour ne pas dépasser le quota
        if count >= LIMIT_REQUESTS:
            print(f"🛑 Limite de {LIMIT_REQUESTS} requêtes atteinte. La suite demain !")
            break
            
        print(f"[{count+1}/{LIMIT_REQUESTS}] Récupération match ID {fixture_id}...", end="\r")
        
        try:
            params_p = {"fixture": fixture_id}
            response = requests.get(url_players, headers=headers, params=params_p)
            data = response.json()
            
            # Si on a une réponse valide
            if 'response' in data and len(data['response']) > 0:
                # Pour chaque équipe (Domicile / Extérieur)
                for team_data in data['response']:
                    team_id = team_data['team']['id']
                    team_name = team_data['team']['name']
                    
                    # Pour chaque joueur
                    for player in team_data['players']:
                        if len(player['statistics']) > 0:
                            stats = player['statistics'][0]
                            
                            # Création de la ligne de données
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
                                'interceptions': stats['tackles']['interceptions'] or 0,
                                'duels_total': stats['duels']['total'] or 0,
                                'duels_won': stats['duels']['won'] or 0,
                                'dribbles_success': stats['dribbles']['success'] or 0,
                                'fouls_drawn': stats['fouls']['drawn'] or 0,
                                'fouls_committed': stats['fouls']['committed'] or 0,
                                'cards_yellow': stats['cards']['yellow'] or 0,
                                'cards_red': stats['cards']['red'] or 0,
                                'position': stats['games']['position']
                            }
                            new_data.append(row)
            
            count += 1
            # Petite pause pour respecter le rate-limit API
            time.sleep(0.2)
            
        except Exception as e:
            print(f"\n❌ Erreur sur le match {fixture_id}: {e}")

    # ------------------------------------------------------------------
    # 5. SAUVEGARDE
    # ------------------------------------------------------------------
    if len(new_data) > 0:
        df_new = pd.DataFrame(new_data)
        
        # Si le fichier n'existe pas, on l'écrit avec les entêtes
        if not os.path.exists(FILENAME_STATS):
            df_new.to_csv(FILENAME_STATS, index=False)
        else:
            # Sinon on ajoute à la suite (mode 'a') sans remettre les entêtes
            df_new.to_csv(FILENAME_STATS, mode='a', header=False, index=False)
            
        print(f"\n✅ Succès : {len(df_new)} lignes de stats ajoutées au fichier {FILENAME_STATS}.")
    else:
        print("\n⚠️ Aucune nouvelle donnée récupérée (peut-être des matchs sans stats disponibles).")
