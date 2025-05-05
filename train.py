import logging
import pygame
import time
from ratatouille.ratenv import RatEnv
from ratatouille.const import TEST_4BY4

logging.basicConfig(
    level = logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
def main():
    env = RatEnv(4, TEST_4BY4)
    running = True
    for _ in range(50):
        if not running:
            break
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        next_obs = env.step([0.7, 0.7])
        env.render()
        time.sleep(0.4)


if __name__ == "__main__":
    main()