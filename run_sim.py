import pygame
import time
from maze import Maze
from robot import Robot
from renderer import Renderer
import random

def main():
    maze = Maze(size=16)
    robot = Robot(maze, start=(2,2))
    renderer = Renderer(maze)

    actions = ['forward','forward']
    idx = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
        action = actions[idx % len(actions)]
        state, collision = robot.step(action)
        if collision == True:
            n = random.randint(1,5)
            for i in range(1, n):
                lefty = random.choice(['left', 'right'])
                action = lefty
                state, collision = robot.step(action)
            print(actions)

        state, collision = robot.step(action)
        renderer.draw(robot)

        idx += 1
        time.sleep(0.2)

    pygame.quit()

if __name__ == '__main__':
    main()
