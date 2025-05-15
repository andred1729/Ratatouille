import sys
import torch
import pygame
import numpy as np
from absl import app, flags, logging
from ratatouille.env import RatEnv, MAZES
from ratatouille.utils import set_seed
from ratatouille.data import ReplayBuffer
FLAGS = flags.FLAGS
flags.DEFINE_integer('size', 4, 'Size of the maze.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('max_steps', 100, 'Number of training steps.')

def main(_):
    logging.set_verbosity(logging.DEBUG)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    size = FLAGS.size
    set_seed(FLAGS.seed)
    if size not in MAZES:
        logging.error(f"Maze size {size} is not available in MAZES.")
        return
    
    env = RatEnv(size, MAZES[size])
    observation = env.reset()
    
    buffer = ReplayBuffer(env.observation_dim, env.action_dim, FLAGS.max_steps, device)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # Handle keyboard events
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        
        if env.runnable:
            # action = env.manual_control()
            action = np.tanh(np.random.uniform(-1, 1, env.action_dim))
            logging.info(action)
            next_observation, reward, terminal, truncated, info = env.step(action)
            buffer.insert(observation, action, reward, next_observation, terminal, truncated)
        
        
        env.render()
        env.clock.tick(10)

if __name__ == "__main__":
    app.run(main)