from ratatouille.ratenv import RatEnv
import logging
import pygame
import time
from ratatouille.const import MAZE_LAYOUT

logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
def main():
    env = RatEnv(4)
    env.maze.load_maze(MAZE_LAYOUT)
    running = True
    for _ in range(50):
        if not running:
            break
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        env.step([0.5, 0.8])
        env.render()
        time.sleep(0.1)



if __name__ == "__main__":
    main()