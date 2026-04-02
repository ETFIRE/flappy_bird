import sys
import os
import statistics

# chemin vers le jeu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'game'))
from game_engine import FlappyBirdEnv

def naive_action(state):
    # distance avec le haut du tuyau
    dist_haut = state[3]
    v = state[1] # velocite de l'oiseau
        
    # tester d'autres valeurs que -0.3 si le score est trop nul
    seuil_test = -0.3 
    
    # On saute pas si on est trop haut ou si on monte deja vite
    if dist_haut < 0 or v < seuil_test:
        return 0
    else:
        return 1

def run(n_games=50):
    env = FlappyBirdEnv()
    mes_scores = []
    
    print("Lancement des", n_games, "parties")
    
    for i in range(n_games):
        state = env.reset()
        fini = False
        
        while not fini:
            # saute en boucle ?
            act = naive_action(state)
            state, reward, fini = env.step(act)
            
        mes_scores.append(env.score)
        
    print("\n--- STATS ---")
    print("Max :", max(mes_scores))
    print("Min :", min(mes_scores))
    
    moy = sum(mes_scores) / len(mes_scores)
    print("Moyenne :", round(moy, 2))
    
    if n_games > 1:
        print("Ecart type :", round(statistics.stdev(mes_scores), 2))

if __name__ == '__main__':
    run()