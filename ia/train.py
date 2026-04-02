import sys
import os
import neat
import pickle
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'game'))
from game_engine import FlappyBirdEnv

# chemin du txt
CHEMIN_CONF = os.path.join(os.path.dirname(__file__), 'neat_config.txt')
NB_GEN = 50 

def eval_fitness(genome, config):
    reseau = neat.nn.FeedForwardNetwork.create(genome, config)
    env = FlappyBirdEnv()
    s = env.reset()
    game_over = False
    
    while not game_over:
        out = reseau.activate(s)
        
        # decision de saut
        if out[0] > 0.5:
            a = 1
        else:
            a = 0
            
        s, reward, game_over = env.step(a)
        
    # Calcul de la fitness
    fit = env.frames + (500 * env.score)
    return fit

def eval_tous_les_genomes(genomes, config):
    for gen_id, gen in genomes:
        gen.fitness = eval_fitness(gen, config)

def faire_le_graphique(stats, chemin_sortie):
    gens = range(len(stats.most_fit_genomes))
    meilleurs = [g.fitness for g in stats.most_fit_genomes]
    moyennes = stats.get_fitness_mean()
    
    # Taille des especes
    nb_esp = []
    for tailles in stats.get_species_sizes():
        nb_esp.append(len(tailles))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # graphe 1
    ax1.plot(gens, meilleurs, label='Max', color='blue')
    ax1.plot(gens, moyennes, label='Moyenne', color='orange')
    ax1.set_ylabel('Score Fitness')
    ax1.set_title('Evol de la fitness globale')
    ax1.legend()
    
    # graphe 2
    ax2.plot(gens, nb_esp, color='green', label="Especes")
    ax2.set_xlabel('Generations')
    ax2.set_ylabel('Nb especes')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(chemin_sortie)
    plt.close()
    # print("Graphe generé")

def run():
    conf = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CHEMIN_CONF
    )
    
    pop = neat.Population(conf)
    
    pop.add_reporter(neat.StdOutReporter(True))
    mes_stats = neat.StatisticsReporter()
    pop.add_reporter(mes_stats)
    
    # creation du dossier checkpoints si il existe pas
    dossier_chkpt = os.path.join(os.path.dirname(__file__), 'checkpoints')
    if not os.path.exists(dossier_chkpt):
        os.makedirs(dossier_chkpt)
        
    sauvegarde = neat.Checkpointer(10, filename_prefix=os.path.join(dossier_chkpt, 'checkpoint-'))
    pop.add_reporter(sauvegarde)
    
    # lancement de l'entrainement
    le_meilleur = pop.run(eval_tous_les_genomes, NB_GEN)
    
    chemin_save = os.path.join(os.path.dirname(__file__), 'best_genome.pkl')
    with open(chemin_save, 'wb') as fichier:
        pickle.dump(le_meilleur, fichier)
        
    # Courbes
    img_path = os.path.join(os.path.dirname(__file__), 'fitness_courbe.png')
    faire_le_graphique(mes_stats, img_path)
    
    print("\n>> Meilleur sauvé dans", chemin_save)
    print("Fitness max obtenue :", round(le_meilleur.fitness, 1))

if __name__ == '__main__':
    run()