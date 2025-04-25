import pygame

CELL_SIZE = 40
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
        for y in range(self.maze.size):
            for x in range(self.maze.size):
                color = WALL_COLOR if self.maze.grid[y, x] else FREE_COLOR
                rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, color, rect)
        # robot rendering 
    
        rx, ry = robot.x * CELL_SIZE + CELL_SIZE/2, robot.y * CELL_SIZE + CELL_SIZE/2
        pygame.draw.circle(self.screen, ROBOT_COLOR, (int(rx), int(ry)), CELL_SIZE//3)
        pygame.display.flip()
