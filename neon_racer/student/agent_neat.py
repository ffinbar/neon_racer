import neat
import pickle
from .agent import Agent
from . import settings, environment

class NeatAgent(Agent):
    def __init__(self, config_path, genome_path):
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_path)
        
        with open(genome_path, 'rb') as f:
            self.genome = pickle.load(f)
            
        self.name = f"Neat_G{self.genome.key}"
        self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

    # ─────────────────────────────────────────────────────────────
    # STUDENT ZONE: Neural Network Output Translation
    # ─────────────────────────────────────────────────────────────
        
    def get_action(self, state):
        if 'neural_inputs' in state:
            inputs = state['neural_inputs']
        else:
            inputs = []
            max_dist = settings.MAX_DIST
            max_speed = settings.MAX_SPEED_INPUT
            for s in state['sensors']:
                inputs.append(s / max_dist)
            inputs.append(state['speed'] / max_speed)
        
        # The neural network produces 3 raw outputs
        output = self.net.activate(inputs)
        
        # use the universal translation layer function in environment.py
        return environment.translate_neural_output(output)

    # ─────────────────────────────────────────────────────────────
