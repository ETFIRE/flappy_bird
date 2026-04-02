import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'game'))
from game_engine import FlappyBirdEnv

class Perceptron:
    def __init__(self, nb_entrees=5):
        # initialisation entre -1 et 1
        self.w = np.random.uniform(-1, 1, nb_entrees)
        self.b = np.random.uniform(-1, 1)

    def forward(self, x):
        # z = w*x + b
        z = np.dot(self.w, x) + self.b
        return np.tanh(z)

    def decide(self, x):
        res = self.forward(x)
        # on saute que si le resultat est positif
        if res > 0:
            return 1
        return 0 # sinon on fait rien

def run(n_games=100):
    jeu = FlappyBirdEnv()
    reseau = Perceptron()
    historique = []
    
    for i in range(n_games):
        etat = jeu.reset()
        mort = False
        
        while mort == False:
            action = reseau.decide(etat)
            etat, r, mort = jeu.step(action)
            
        historique.append(jeu.score)
        # print("score de la partie", i+1, ":", jeu.score)
        
    moyenne = sum(historique) / len(historique)
    print(f"\nMoyenne sur {n_games} parties : {moyenne:.1f}")

if __name__ == '__main__':
    run()