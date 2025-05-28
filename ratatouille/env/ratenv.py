from absl import logging
import sys
import numpy as np
import pygame
from math import pi, floor, fabs, ceil
from typing import Tuple, Dict
import matplotlib.cm as cm
from ratatouille.env.maze import Maze
from ratatouille.env.const import SCALING, WALL_T, WALL_COLOR, FREE_COLOR, ROBOT_COLOR


def bound(x, b):
    return max(-b, min(b, x))

def angle_wrap(theta_deg):
    """Wrap angle to [-180, 180] degrees"""
    return ((theta_deg + 180) % 360) - 180

def expcurve(x, k = 4):
    """Rapidly increasing function f that has f(0) = -1, f(1) = 1. Takes in argument in [0, 1]

    Args:
        x (float): progress

    Returns:
        float: reward for distance
    """
    return (np.exp(k * x) - np.exp(k))/(np.exp(k) - 1)

# def sigmoid(x, k=4):
#     """Sigmoid like function with parameter

#     Args:
#         x (float): progress in [-1, 1]
#         k (float): steepness

#     Returns:
#         float: reward for delta distance
#     """
#     return 2/(1 + np.exp(-k * x)) - 1
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
    is_win: bool
    observation_dim: int
    action_dim: int
    action_range: Tuple[int]
    info: Dict[str, np.float32]
    incremental_training: bool
    incremental_size_training: bool
    max_steps: int
    def __init__(self, size, text_maze, max_episode_length = 300, partition_size = 10, lidar_count = 4, rewards={"wall": -100, "center": 100}, use_pygame=False, incremental_training=False, incremental_size_training=False, max_steps=500000):
        # Maze and robot state 
        self.size = size
        self.maze = Maze(size, text_maze, partition_size)
        self.partition_size = partition_size
        self.dt = 0.1
        self.max_speed = 0.4
        self.radius = 0.1
        self.diam = 2 * self.radius
        self.action_dim = 2
        self.action_range = (-1, 1)
        self.friction = 0.05  # Friction coefficient to slow down the robot when no keys are pressed
        self.lidar_count = lidar_count
        
        # Episode tracker
        self.max_possible_episode_length = max_episode_length
        self.max_steps = max_steps
        self.incremental_training = incremental_training
        self.incremental_size_training = incremental_size_training
        self.eff_size = self.size
        
        # Discounting
        self.discount = 0.99
        
        # Reward design
        self.rewards = rewards
        
        # Initialization work
        self.reset(0)
        self.observation_dim = len(self.state)
        self.use_pygame = use_pygame
        if self.use_pygame:
            self.init_pygame()
        logging.info(f"Initialized RatEnv: size={self.size}, partition_size={self.partition_size}, max_episode_length={self.max_episode_length}, lidar_count={self.lidar_count}, rewards={self.rewards}, use_pygame={self.use_pygame}, eff_size={self.eff_size}")
        
    def _update_state(self):
        self.state = np.array([self.x, self.y, self.theta, self.velocity_left, self.velocity_right] + [self.maze.lidar(self.x, self.y, self.theta + float(y)/self.lidar_count * (2 * pi)) for y in range(self.lidar_count)])
        self.theta_deg = self.theta * 180 / pi
        
        # top-left corner is (-self.size/2, self.size/2)
        self.maze_x = floor(self.x - (-self.size/2))
        self.maze_y = floor(self.size/2 - self.y)
        
        self.partition_maze_x = floor((self.x - (-self.size/2)) * self.partition_size)
        self.partition_maze_y = floor((self.size/2 - self.y) * self.partition_size)
    
    def init_pygame(self):
        # Visualization
        pygame.init()
        self.font = pygame.font.SysFont("Arial", 16)
        self.cell_size = SCALING[self.size]
        self.screen = pygame.display.set_mode((self.size * self.cell_size, self.size * self.cell_size))
        pygame.display.set_caption("RatEnv Visualization")
        
        # Control settings
        self.acceleration = 1.0  # Acceleration constant for keyboard controls
        self.clock = pygame.time.Clock()
        
        self.colormap = cm.get_cmap('plasma')
        
    def to_string(self):
        return (f"State: x={self.x:.2f}, y={self.y:.2f}, θ={self.theta_deg:.2f}° "
                f"(wrapped={angle_wrap(self.theta_deg):.2f}°), vl={self.velocity_left:.2f}, vr={self.velocity_right:.2f}")

    def step_physics(self, action):
        """Run the physics simulation

        Args:
            action (2-element array): [acc_l, acc_r]
        """
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

    def step(self, action):
        """
            return next_state, reward, terminal, truncated, info
            
            takes in action as an action_dim-dimensional numpy array
        """
        info = {}
        if not self.runnable:
            return
        
        self.current_episode_length += 1
        old_partition_dist = self.maze.partition_dist[self.partition_maze_y][self.partition_maze_x]
        old_dist = self.maze.dist[self.maze_y][self.maze_x]
        self.step_physics(action)
        
        logging.debug(self.to_string())
        
        reward = 0
        terminal = False
        truncated = False
        # handle terminal states: collision and getting to the center
        if self.maze.check_collision(self.x, self.y, self.radius):
            logging.debug("Collision detected! Simulation ended.")
            self.runnable = False
            self.is_win = False
            reward = self.rewards["wall"]
            terminal = True
        elif self.maze.check_win(self.x, self.y, self.radius):
            logging.debug("You win!")
            self.runnable = False
            self.is_win = True
            reward = self.rewards["center"]
            terminal = True
        else:
            new_partition_dist = self.maze.partition_dist[self.partition_maze_y][self.partition_maze_x]
            new_dist = self.maze.dist[self.maze_y][self.maze_x]
            reward = 0
            # reward += (old_partition_dist - new_partition_dist)/np.max(self.maze.partition_dist)
            # reward += (old_dist - new_dist)/np.max(self.maze.dist)
            reward += expcurve(1.0 - float(new_partition_dist)/np.max(self.maze.partition_dist), 2)
            reward += expcurve(1.0 - float(new_dist)/np.max(self.maze.dist), 2)
            logging.debug(f"old pdist: {old_partition_dist}, new pdist: {new_partition_dist}, old dist: {old_dist}, new dist: {new_dist}, reward: {reward}")
            # check truncation
            if (self.current_episode_length == self.max_episode_length):
                logging.debug("Episode truncated")
                self.runnable = False
                truncated = True

        self.current_episode_discounted_return += self.discount**(self.current_episode_length - 1) * reward
        info.update({
            "is_win": self.is_win,
            "current_episode_length": self.current_episode_length,
            "current_episode_discounted_return": self.current_episode_discounted_return
        })
        step_output = (self.state, reward, terminal, truncated, info)
        # logging.info(step_output)
        return step_output

    def render(self, added_info):
        if not self.use_pygame:
            return
        self.screen.fill(FREE_COLOR)
        partition_dist = self.maze.partition_dist
        partition_dist_max = np.max(partition_dist)
        for r in range(self.size * self.partition_size):
            for c in range(self.size * self.partition_size):
                p = float(partition_dist[r][c])/partition_dist_max
                color = tuple(int(255 * x) for x in self.colormap(1-p)[:3]) if p >= 0 else (250, 250, 250)
                cell_width = float(self.cell_size/self.partition_size)
                rect= pygame.Rect(int(c * cell_width), int(r * cell_width), ceil(cell_width), ceil(cell_width))
                pygame.draw.rect(self.screen, color, rect)
        
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
                
                text_surf = self.font.render(str(self.maze.dist[row_i][col_i]), True, (0, 0, 0))
                text_rect = text_surf.get_rect(center=(x + self.cell_size // 2, y + self.cell_size // 2))
                self.screen.blit(text_surf, text_rect)


        # Draw different color if simulation ended due to collision
        robot_color = (0, 0, 0) if not self.runnable else ROBOT_COLOR
        
        screen_x = int((self.x + self.size / 2) * self.cell_size)
        screen_y = int((self.size / 2 - self.y) * self.cell_size)
        
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
            (int(screen_x), int(screen_y)),
            (int(tip_x), int(tip_y)),
            # (screen_x, screen_y),
            2
        )

        # Draw the arrowhead triangle
        pygame.draw.polygon(
            self.screen,
            (0, 0, 255),
            [(int(triangle_tip_x), int(triangle_tip_y)), (int(left_x), int(left_y)), (int(right_x), int(right_y))]
        )
        
        # Draw current maze information
        info_text = self.font.render(f"{added_info} ep_len: {self.current_episode_length} ep_ret: {self.current_episode_discounted_return:.2f}", True, (0, 0, 0))
        info_rect = info_text.get_rect(topleft=(220,0))
        self.screen.blit(info_text, info_rect)
        
        # Draw status text if simulation ended
        if not self.runnable:
            if self.is_win: 
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

        for y in range(self.lidar_count):
            # Lidar
            angle = self.theta + float(y)/self.lidar_count * (2 * pi)
            lidar_distance = self.maze.lidar(self.x, self.y, angle)

            end_x = self.x + lidar_distance * np.cos(angle)
            end_y = self.y + lidar_distance * np.sin(angle)

            screen_end_x = int((end_x + self.size / 2) * self.cell_size)
            screen_end_y = int((self.size / 2 - end_y) * self.cell_size)

            # Draw red laser line
            pygame.draw.line(
                self.screen,
                (255, 0, 0), 
                (screen_x, screen_y), 
                (screen_end_x, screen_end_y), )
        
        pygame.display.flip()
    
    def reset(self, step):
        """Reset the simulation to initial state"""
        self.theta = pi / 2
        self.velocity_left = 0
        self.velocity_right = 0
        self.runnable = True
        self.is_win = False
        self.current_episode_length = 0
        self.current_episode_discounted_return = 0
        
        # episode length incremental training
        size_list = {
            6: [4, 6],
            8: [4, 6, 8],
            12: [4, 6, 8, 12]
        }
        if self.incremental_size_training:
            size_l = size_list[self.size]
            minisize = int(self.max_steps/len(size_l))
            self.eff_size = size_l[int(step/minisize)]
            logging.info(f'eff_size: {self.eff_size}')
        
        self.x = -(self.eff_size / 2 - 0.5)
        self.y = -(self.eff_size / 2 - 0.5)
        
        # size incremental training
        self.max_episode_length = self.max_possible_episode_length
        if self.incremental_training:
            if step >= int(3 * self.max_steps/4):
                self.max_episode_length = self.max_possible_episode_length
            elif step >= int(2 * self.max_steps/4):
                self.max_episode_length = int(self.max_possible_episode_length * 0.9)
            elif step >= int(self.max_steps/4):
                self.max_episode_length = int(self.max_possible_episode_length * 0.7)
            else:
                self.max_episode_length = int(self.max_possible_episode_length * 0.5)

        self._update_state()
        logging.debug("Simulation reset")
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
    
    def quit_pygame(self):
        if not self.use_pygame:
            return
        pygame.quit()
        pygame.display.quit()
        
    def clock_tick(self, fps):
        if not self.use_pygame:
            return
        self.clock.tick(fps)