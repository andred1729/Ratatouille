import numpy as np

class Robot:
    def __init__(self, maze, start=(0,0), theta=0.0, motion_noise=0.05):
        self.maze = maze
        self.x, self.y = start
        self.theta = theta        # orientation in radians
        self.motion_noise = motion_noise

    def step(self, action):
        """
        action: one of 'forward', 'left', 'right'
        returns: (new_state, collision_flag)
        """
        # deterministic update
        if action == 'forward':
            dx = int(round(np.cos(self.theta)))
            dy = int(round(np.sin(self.theta)))
            new_x, new_y = self.x + dx, self.y + dy         
        elif action == 'left':
            new_x, new_y = self.x, self.y
            self.theta += np.pi / 2
        elif action == 'right':
            new_x, new_y = self.x, self.y
            self.theta -= np.pi / 2
        else:
            raise ValueError(f"Unknown action {action}")


        # check collision with maze walls
        if action == 'forward':
            collision = True
            print("Collision")
            return (self.x, self.y, self.theta), collision

        # commit the move
        self.x, self.y = new_x, new_y
        self.theta %= 2 * np.pi
        return (self.x, self.y, self.theta), False
