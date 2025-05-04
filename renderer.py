import pygame
from maze import Maze
from robot import Robot

CELL_SIZE = 40
WALL_T = 1
WALL_COLOR = (30, 30, 30)
FREE_COLOR = (220, 220, 220)
ROBOT_COLOR = (200, 30, 30)

class Renderer:
    def __init__(self, maze):
        pygame.init()
        self.maze = maze
        self.screen = pygame.display.set_mode(
            (maze.size * CELL_SIZE, maze.size * CELL_SIZE)
        )

    def draw(self, robot):

        self.screen.fill(FREE_COLOR)
        
        for row_i, row in enumerate(self.maze.grid):
            for col_i, walls in enumerate(row):
                x = col_i * CELL_SIZE
                y = row_i * CELL_SIZE

                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

                pygame.draw.rect(self.screen, FREE_COLOR, rect)

                top, right, bottom, left = walls
                print(walls)
                if top:
                    pygame.draw.line(self.screen, WALL_COLOR, (x,y), (x+CELL_SIZE,y), WALL_T)
                if right:
                    pygame.draw.line(self.screen, WALL_COLOR, (x+CELL_SIZE,y), (x+CELL_SIZE, y+CELL_SIZE), WALL_T)
                if bottom:
                    pygame.draw.line(self.screen, WALL_COLOR, (x,y+CELL_SIZE), (x+CELL_SIZE, y+CELL_SIZE), WALL_T)
                if left:
                    pygame.draw.line(self.screen, WALL_COLOR, (x,y), (x,y+CELL_SIZE), WALL_T)

                rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                pygame.draw.rect(self.screen, FREE_COLOR, rect)

        # robot rendering 
    
        rx, ry = robot.x * CELL_SIZE + CELL_SIZE/2, robot.y * CELL_SIZE + CELL_SIZE/2
        pygame.draw.circle(self.screen, ROBOT_COLOR, (int(rx), int(ry)), CELL_SIZE//3)

        pygame.display.flip()
