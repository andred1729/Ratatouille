from absl import logging
import sys
import numpy as np
import pygame
from math import pi, floor
from typing import Tuple, Dict
from ratatouille.env.maze import Maze
from ratatouille.env.const import SCALING, WALL_T, WALL_COLOR, FREE_COLOR, ROBOT_COLOR

# logger = logging.getLogger(__name__)

def bound(x, b):
    return max(-b, min(b, x))

def angle_wrap(theta_deg):
    """Wrap angle to [-180, 180] degrees"""
    return ((theta_deg + 180) % 360) - 180

class RatEnv:
    """
    Class simulating physics and keeping track of interaction
    """
    x: np.float32
    y: np.float32
    maze_x: int
    maze_y: int
    theta: np.float32
    theta_deg: np.float32
    velocity_left: np.float32
    velocity_right: np.float32
    maze: Maze
    state: np.ndarray
    runnable: bool
    _is_terminal_win: bool
    observation_shape: Tuple
    action_shape: Tuple
    info: Dict[str, np.float32]
    def __init__(self, size, text_maze):
        # Maze and robot state 
        self.size = size
        self.maze = Maze(size, text_maze)
        self.dt = 0.1
        self.max_speed = 0.4
        self.radius = 0.1
        self.diam = 2 * self.radius
        self.action_shape = (2,)

        # Episode tracker
        self.max_episode_length = 5e2
        
        # Discounting
        self.gamma = 0.99
        
        # Reward design
        self.rewards = {
            "wall": -1e4,
            "center": 1e4
        }
                
        # Initialization work
        self.reset()
        self.observation_shape = self.state.shape
        self._vis_init()
        
        logging.info("Initialized RatEnv")
        logging.info(self.to_string())
    
    def _update_state(self):
        self.state = np.array([self.x, self.y, self.theta, self.velocity_left, self.velocity_right])
        self.theta_deg = self.theta * 180 / pi
        
        # top-left corner is (-self.size/2, self.size/2)
        self.maze_x = floor(self.x - (-self.size/2))
        self.maze_y = floor(self.size/2 - self.y)
    
    def _vis_init(self):
        # Visualization
        pygame.init()
        self.font = pygame.font.SysFont("Arial", 16)
        self.cell_size = SCALING[self.size]
        self.screen = pygame.display.set_mode((self.size * self.cell_size, self.size * self.cell_size))
        pygame.display.set_caption("RatEnv Visualization")
        
        # Control settings
        self.acceleration = 1.0  # Acceleration constant for keyboard controls
        self.clock = pygame.time.Clock()
        
        # Add friction to make controls more natural
        self.friction = 0.05  # Friction coefficient to slow down the robot when no keys are pressed

    def to_string(self):
        return (f"State: x={self.x:.2f}, y={self.y:.2f}, θ={self.theta_deg:.2f}° "
                f"(wrapped={angle_wrap(self.theta_deg):.2f}°), vl={self.velocity_left:.2f}, vr={self.velocity_right:.2f}")

    def step(self, action):
        """
            return next_state, reward, terminal, truncated, info
        """
        info = {}
        if not self.runnable:
            return
        
        self.current_episode_length += 1
        action_left, action_right = action
        
        # Apply friction to slow down when no input is given
        if abs(action_left) < 0.01:
            self.velocity_left *= (1 - self.friction)
        if abs(action_right) < 0.01:
            self.velocity_right *= (1 - self.friction)
            
        # Update velocities based on actions
        self.velocity_left = bound(self.velocity_left + action_left * self.dt, self.max_speed)
        self.velocity_right = bound(self.velocity_right + action_right * self.dt, self.max_speed)

        v = (self.velocity_left + self.velocity_right) / 2
        omega =  (self.velocity_right - self.velocity_left) / (3 * self.diam)

        dx = v * np.cos(self.theta) * self.dt
        dy = v * np.sin(self.theta) * self.dt
        dtheta = omega * self.dt

        # Calculate new position
        new_x = self.x + dx
        new_y = self.y + dy

        # If no collision, update position
        self.x = new_x
        self.y = new_y
        self.theta += dtheta
        self._update_state()
        
        logging.debug(self.to_string())
        
        reward = 0
        terminal = False
        truncated = False
        # handle terminal states: collision and getting to the center
        if self.maze.check_collision(new_x, new_y, self.radius):
            logging.info("Collision detected! Simulation ended.")
            self.runnable = False
            self._is_terminal_win = False
            info["is_terminal_win"] = self._is_terminal_win
            reward = self.rewards["wall"]
            terminal = True
        elif self.maze.check_win(new_x, new_y, self.radius):
            logging.info("You win!")
            self.runnable = False
            self._is_terminal_win = True
            info["is_terminal_win"] = self._is_terminal_win
            reward = self.rewards["center"]
            terminal = True
        else:
            # handle non-terminal states, calculate the rewards
            r_maze_dist_to_center = float(self.size)/max(self.maze.dist[self.maze_y][self.maze_x], 1e-5)
            r_euclidean_dist_to_center = 1.0/ max(np.sqrt(self.x**2 + self.y**2), 1.0)
            r_stationary = -1e-4/(1e-5 + max(max(self.velocity_left ** 2, self.velocity_right ** 2) - 1e-5, 0))
            logging.debug(f"r_maze: {r_maze_dist_to_center:.2f}, r_euclid: {r_euclidean_dist_to_center:.2f}, r_stationary: {r_stationary:.2f}")
            reward =  r_maze_dist_to_center + r_euclidean_dist_to_center + r_stationary
            # check truncation
            if (self.current_episode_length == self.max_episode_length):
                logging.info("Episode truncated")
                self.runnable = False
                truncated = True

        self.current_episode_discounted_return += self.gamma**(self.current_episode_length - 1) * reward
        info.update({
            "current_episode_length": self.current_episode_length,
            "current_episode_discounted_return": self.current_episode_discounted_return
        })
        step_output = (self.state, reward, terminal, truncated, info)
        logging.info(step_output)
        return step_output

    def render(self):
        self.screen.fill(FREE_COLOR)
        for row_i, row in enumerate(self.maze.grid):
            for col_i, walls in enumerate((row)):
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
                dist = self.maze.dist[row_i][col_i]
                text_surf = self.font.render(str(dist), True, (0, 0, 0))
                text_rect = text_surf.get_rect(center=(x + self.cell_size // 2, y + self.cell_size // 2))
                self.screen.blit(text_surf, text_rect)
                screen_x = int((self.x + self.size / 2) * self.cell_size)
                screen_y = int((self.size / 2 - self.y) * self.cell_size)

        # Draw different color if simulation ended due to collision
        robot_color = (0, 0, 0) if not self.runnable else ROBOT_COLOR
        
        pygame.draw.circle(
            self.screen,
            robot_color,
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
        
        # Draw current velocity information
        velocity_text = self.font.render(f"vL: {self.velocity_left:.2f} vR: {self.velocity_right:.2f} x: {self.x:.2f} y: {self.y:.2f} maze_x: {self.maze_x} maze_y: {self.maze_y} ep: {self.current_episode_length}", True, (0, 0, 0))
        velocity_rect = velocity_text.get_rect(topleft=(220,0))
        self.screen.blit(velocity_text, velocity_rect)
        
        # Draw status text if simulation ended
        if not self.runnable:
            if self._is_terminal_win: 
                game_over_font = pygame.font.SysFont("Arial", 36)
                game_over_text = game_over_font.render("You Win!", True, (255, 0, 0))
                text_rect = game_over_text.get_rect(center=(self.size * self.cell_size // 2, 30))
                self.screen.blit(game_over_text, text_rect)
                
                restart_text = self.font.render("Press 'R' to restart", True, (0, 0, 0))
                restart_rect = restart_text.get_rect(center=(self.size * self.cell_size // 2, 70))
                self.screen.blit(restart_text, restart_rect)
            else:
                game_over_font = pygame.font.SysFont("Arial", 36)
                game_over_text = game_over_font.render("Simulation Ended - Truncation/Collision!", True, (255, 0, 0))
                text_rect = game_over_text.get_rect(center=(self.size * self.cell_size // 2, 30))
                self.screen.blit(game_over_text, text_rect)
                
                restart_text = self.font.render("Press 'R' to restart", True, (0, 0, 0))
                restart_rect = restart_text.get_rect(center=(self.size * self.cell_size // 2, 70))
                self.screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()
    
    def reset(self):
        """Reset the simulation to initial state"""
        self.x = -(self.size / 2 - 0.5)
        self.y = -(self.size / 2 - 0.5)
        self.theta = pi / 2
        self.velocity_left = 0
        self.velocity_right = 0
        self.runnable = True
        self._is_terminal_win = False
        self.current_episode_length = 0
        self.current_episode_discounted_return = 0
        self._update_state()
        logging.info("Simulation reset")
        logging.info(self.to_string())
        return self.state
        
    def manual_control(self):
        """Get in keyboard control"""
        
        # Continuous key state
        keys = pygame.key.get_pressed()
        
        # Reset actions for this frame
        action_left = 0
        action_right = 0
        
        # Forward/backward movement (both wheels)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action_left = self.acceleration
            action_right = self.acceleration
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action_left = -self.acceleration
            action_right = -self.acceleration
        
        # Turning (differential wheel speeds)
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action_left -= self.acceleration * 0.5
            action_right += self.acceleration * 0.5
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action_left += self.acceleration * 0.5
            action_right -= self.acceleration * 0.5
            
        return np.array([action_left, action_right])