import logging
import pygame
import time
from absl import app, flags
from ratatouille.ratenv import RatEnv
from ratatouille.const import MAZES

FLAGS = flags.FLAGS
flags.DEFINE_integer('size', 4, 'Size of the maze')

logging.basicConfig(
    level=logging.DEBUG,
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
    running = True
    for _ in range(1000):
        if not running:
            break
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        next_obs = env.step([0.9, 0.7])
        env.render()
        time.sleep(0.1)


if __name__ == "__main__":
    app.run(main)