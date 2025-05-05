import numpy as np
import logging
import pygame
from math import pi
from ratatouille.maze import Maze
from ratatouille.const import SCALING, WALL_T, WALL_COLOR, FREE_COLOR, ROBOT_COLOR, TEST_4BY4

logger = logging.getLogger(__name__)

def bound(x, b):
    return max(-b, min(b, x))

def angle_wrap(theta_deg):
    """Wrap angle to [-180, 180] degrees"""
    return ((theta_deg + 180) % 360) - 180

class RatEnv:
    def __init__(self, size=4, text_maze=TEST_4BY4):
        # Maze and robot state
        self.size = size
        self.maze = Maze(size, text_maze)
        self.x = -(self.size / 2 - 0.5)
        self.y = -(self.size / 2 - 0.5)
        self.theta = pi / 2
        self.vl = 0
        self.vr = 0
        self.dt = 0.1
        self.max_speed = 1.0
        self.radius = 0.1
        self.diam = 2 * self.radius

        self.update_state()

        # Visualization
        pygame.init()
        self.font = pygame.font.SysFont("Arial", 16)
        self.cell_size = SCALING[size]
        self.screen = pygame.display.set_mode((size * self.cell_size, size * self.cell_size))
        pygame.display.set_caption("RatEnv Visualization")

        # logging
        logger.info("Initialized RatEnv")
        logger.info(self.to_string())

    def update_state(self):
        self.theta_deg = self.theta * 180 / pi
        self.state = np.array([self.x, self.y, self.theta, self.vl, self.vr])

    def to_string(self):
        return (f"State: x={self.x:.2f}, y={self.y:.2f}, θ={self.theta_deg:.2f}° "
                f"(wrapped={angle_wrap(self.theta_deg):.2f}°), vl={self.vl:.2f}, vr={self.vr:.2f}")

    def step(self, action):
        al, ar = action
        self.vl = bound(self.vl + al * self.dt, self.max_speed)
        self.vr = bound(self.vr + ar * self.dt, self.max_speed)

        v = (self.vl + self.vr) / 2
        omega = (self.vr - self.vl) / self.diam

        dx = v * np.cos(self.theta) * self.dt
        dy = v * np.sin(self.theta) * self.dt
        dtheta = omega * self.dt

        self.x += dx
        self.y += dy
        self.theta += dtheta

        self.update_state()
        logger.info(self.to_string())
        
        return self.state

    def render(self):
        self.screen.fill(FREE_COLOR)

        # Draw maze walls
        for row_i, row in enumerate(self.maze.grid):
            for col_i, walls in enumerate(row):
                x = col_i * self.cell_size
                y = row_i * self.cell_size
                top, right, bottom, left = walls

                if top:
                    pygame.draw.line(self.screen, WALL_COLOR, (x, y), (x + self.cell_size, y), WALL_T)
                if right:
                    pygame.draw.line(self.screen, WALL_COLOR, (x + self.cell_size, y), (x + self.cell_size, y + self.cell_size), WALL_T)
                if bottom:
                    pygame.draw.line(self.screen, WALL_COLOR, (x, y + self.cell_size), (x + self.cell_size, y + self.cell_size), WALL_T)
                if left:
                    pygame.draw.line(self.screen, WALL_COLOR, (x, y), (x, y + self.cell_size), WALL_T)

                # draw distance
                
                dist = self.maze.dist[row_i][col_i]
                text_surf = self.font.render(str(dist), True, (0, 0, 0))
                text_rect = text_surf.get_rect(center=(x + self.cell_size // 2, y + self.cell_size // 2))
                self.screen.blit(text_surf, text_rect)

        
        screen_x = int((self.x + self.size / 2) * self.cell_size)
        screen_y = int((self.size / 2 - self.y) * self.cell_size)

        # Draw robot
        pygame.draw.circle(
            self.screen,
            ROBOT_COLOR,
            (screen_x, screen_y),
            int(self.radius * self.cell_size)
        )

        # Arrow tip rendering (triangle at the end of heading)
        arrow_length = self.radius * self.cell_size 
        tip_x = screen_x + arrow_length * np.cos(self.theta)
        tip_y = screen_y - arrow_length * np.sin(self.theta)  # Y is inverted

        # Create triangle points for arrow tip
        arrow_width = self.radius * self.cell_size * 0.3
        angle_left = self.theta + np.pi / 2
        angle_right = self.theta - np.pi / 2

        left_x = tip_x + arrow_width * np.cos(angle_left)
        left_y = tip_y - arrow_width * np.sin(angle_left)

        right_x = tip_x + arrow_width * np.cos(angle_right)
        right_y = tip_y - arrow_width * np.sin(angle_right)

        triangle_tip_x = screen_x + 1.5 * arrow_length * np.cos(self.theta)
        triangle_tip_y = screen_y - 1.5 * arrow_length * np.sin(self.theta)  # Y is inverted

        # Draw line from center to tip
        pygame.draw.line(
            self.screen,
            (0, 0, 255),
            (screen_x, screen_y),
            (tip_x, tip_y),
            2
        )

        # Draw the arrowhead triangle
        pygame.draw.polygon(
            self.screen,
            (0, 0, 255),
            [(triangle_tip_x, triangle_tip_y), (left_x, left_y), (right_x, right_y)]
        )
        pygame.display.flip()