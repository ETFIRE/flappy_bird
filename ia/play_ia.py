import sys
import os
import neat
import pickle
import pygame

# on ajoute le dossier game pour trouver les fichiers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'game'))
from game_engine import FlappyBirdEnv
from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS

CHEMIN_CERVEAU = os.path.join(os.path.dirname(__file__), 'best_genome.pkl')
CHEMIN_CONF = os.path.join(os.path.dirname(__file__), 'neat_config.txt')

def charger_les_donnees():
    # pkl sauvegardé
    with open(CHEMIN_CERVEAU, 'rb') as fichier:
        gen = pickle.load(fichier)
        
    conf = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CHEMIN_CONF
    )
    return gen, conf

def lancer_le_jeu(genome, config):
    pygame.init()
    ecran = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird - L'IA joue toute seule")
    
    horloge = pygame.time.Clock()
    police = pygame.font.SysFont('monospace', 20)
    
    # creation du reseau a partir du meilleur genome
    reseau = neat.nn.FeedForwardNetwork.create(genome, config)
    jeu = FlappyBirdEnv()
    
    en_cours = True
    while en_cours:
        etat = jeu.reset()
        mort = False
        nb_frames = 0
        
        while mort == False and en_cours == True:
            # gestion des touches pour quitter
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    en_cours = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:  # echap pour quitter
                        en_cours = False
                        
            # l'IA reflechit
            sortie = reseau.activate(etat)
            
            if sortie[0] > 0.5:
                choix = 1
                txt_action = "SAUT !"
            else:
                choix = 0
                txt_action = "il attend..."
                
            etat, r, mort = jeu.step(choix)
            nb_frames += 1

            # Rendre l'affichage lisible
            pygame.draw.rect(ecran, (0, 0, 0), (0, 0, 350, 120))
            
            # affichage du texte sur l'ecran (score, etc)
            lignes_texte = [
                "Score : " + str(jeu.score),
                "Frames : " + str(nb_frames),
                "Valeur reseau : " + str(round(sortie[0], 3))
            ]
            
            for i in range(len(lignes_texte)):
                texte = police.render(lignes_texte[i], True, (255, 255, 255))
                ecran.blit(texte, (10, 10 + i * 24))
                
            # affichage de l'action
            surf_action = police.render("Action : " + txt_action, True, (255, 255, 0))
            ecran.blit(surf_action, (10, 10 + 3 * 24))
            
            pygame.display.flip()
            horloge.tick(FPS)
            
    pygame.quit()

if __name__ == '__main__':
    mon_genome, ma_config = charger_les_donnees()
    lancer_le_jeu(mon_genome, ma_config)
