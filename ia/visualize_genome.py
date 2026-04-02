import sys
import os
import neat
import pickle

sys.path.insert(0, os.path.dirname(__file__))
import visualize 

CHEMIN_CERVEAU = os.path.join(os.path.dirname(__file__), 'best_genome.pkl')
CHEMIN_CONF = os.path.join(os.path.dirname(__file__), 'neat_config.txt')

# Charge le meilleur oiseau
with open(CHEMIN_CERVEAU, 'rb') as f:
    mon_genome = pickle.load(f)

# Charge les regles
ma_config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    CHEMIN_CONF
)

# Dessine le reseau !
print("Génération du dessin en cours...")
visualize.draw_net(ma_config, mon_genome, view=True, filename='ia/mon_reseau')
print("Le fichier SVG a été créé.")