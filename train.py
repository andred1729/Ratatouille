import logging
import pygame
from absl import app, flags
from ratatouille.env import RatEnv, MAZES

FLAGS = flags.FLAGS
flags.DEFINE_integer('size', 4, 'Size of the maze')

logging.basicConfig(
    level=logging.INFO,  # Change to INFO to reduce debug messages
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

def main(argv):
    del argv  # Unused
    size = FLAGS.size
    if size not in MAZES:
        logging.error(f"Maze size {size} is not available in MAZES.")
        return
    
    env = RatEnv(size, MAZES[size])
    
    # Instead of running a fixed loop with constant input,
    # use the built-in run method for keyboard control
    env.run()

if __name__ == "__main__":
    app.run(main)