import numpy as np
from collections import deque
import logging
import math


class Maze:
    def __init__(self, size, text_maze, partition_size):
        """
        Initialize a Maze object.

        Parameters:
        ----------
        size : int
            The size of the maze grid (number of rows and columns).

        Attributes:
        ----------
        grid : list of list of list of bool
            A 3D grid representing the maze, where grid[row][col] is a list of four booleans:
            [up, right, down, left], indicating whether each wall is present (True) or absent (False).
        """
        self.size = size
        self.grid = [[[False, False, False, False] for _ in range(size)] for _ in range(size)]
        self.load_maze(text_maze)
        self.partition_size = partition_size
        self.partition_dist = self.compute_min_dist_to_center(self.partition_size)
        self.dist = self.compute_min_dist_to_center(1)
        logging.debug(np.array(self.grid).shape)
        self.cell = 0

    def load_maze(self, text_maze):
        tf = {"T": True, "F": False}
        if len(text_maze) != self.size:
            logging.error("not the right number of rows")
            return
        for row_idx, row_str in enumerate(text_maze):
            cells = row_str.split("|")
            if len(cells) != self.size:
                logging.error(f"not the right number of cols for row: {row_idx}")
                return
            for col_idx in range(self.size):
                cell = cells[col_idx]
                if len(cell) != 4:
                    logging.error(f"not the right chars per cell for cell: {row_idx}, {col_idx}")
                    return
                for i, wall in enumerate(cell):
                    if wall not in tf:
                        logging.error(f"not the right char in cell: {row_idx}, {col_idx}")
                        return
                    self.grid[row_idx][col_idx][i] = tf[wall]
        self.and_maze()
        self.add_border_walls()

    def and_maze(self):
        """
        Ensure wall consistency between adjacent cells.
        """
        # east-west
        for row_idx in range(self.size):
            for col_idx in range(1, self.size):
                self.grid[row_idx][col_idx-1][1] = self.grid[row_idx][col_idx-1][1] and self.grid[row_idx][col_idx][3]
                self.grid[row_idx][col_idx][3]   = self.grid[row_idx][col_idx-1][1] and self.grid[row_idx][col_idx][3]
        # north-south
        for col_idx in range(self.size):
            for row_idx in range(1, self.size):
                self.grid[row_idx-1][col_idx][2] = self.grid[row_idx-1][col_idx][2] and self.grid[row_idx][col_idx][0]
                self.grid[row_idx][col_idx][0]   = self.grid[row_idx-1][col_idx][2] and self.grid[row_idx][col_idx][0]

    def add_border_walls(self):
        """
        Enforce walls on all four borders of the maze.
        """
        for i in range(self.size):
            # Top
            self.grid[0][i][0] = True
            # Bottom
            self.grid[self.size - 1][i][2] = True
            # Left
            self.grid[i][0][3] = True
            # Right
            self.grid[i][self.size - 1][1] = True

    def check_collision(self, pos_x, pos_y_negative, rad):
        #rad = robot radius
        if rad > 0.4:
            print("Robot way too fat")
            return True
        
        pos_y = -pos_y_negative
        abs_pos_x = pos_x + float(self.size / 2.0)
        abs_pos_y = pos_y + float(self.size / 2.0)

        square_x = int(abs_pos_x)
        square_y = int(abs_pos_y)


        cell = self.grid[square_y][square_x]
        if square_x < 0 or square_x >= self.size or square_y < 0 or square_y >= self.size:
            print("Warning: robot out of bounds")
            return True  # Treat out-of-bounds as collision

        dist_top    = abs_pos_y - square_y
        dist_right  = (square_x + 1) - abs_pos_x
        dist_bottom = (square_y + 1) - abs_pos_y
        dist_left   = abs_pos_x - square_x
        if cell[0] and dist_top < rad:
            return True  # top wall
        if cell[1] and dist_right < rad:
            return True  # right wall
        if cell[2] and dist_bottom < rad:
            return True  # bottom wall
        if cell[3] and dist_left < rad:
            return True  # left wall
        
        #TODO: corner checks
                
    def check_win(self, pos_x, pos_y, rad):
        entrancefactor = 1 #how deep into the center does the rat have to go to win
        if abs(pos_x) <= (1 - entrancefactor*rad) and abs(pos_y) <= (1 - entrancefactor*rad):
            return True

    def compute_min_dist_to_center(self, partition_size):
        fine_size = self.size * partition_size
        dist = [[-1 for _ in range(fine_size)] for _ in range(fine_size)]
        queue = deque()

        # Identify the 4 center cells
        centers = [
            (self.size // 2 - 1, self.size // 2 - 1),
            (self.size // 2 - 1, self.size // 2),
            (self.size // 2,     self.size // 2 - 1),
            (self.size // 2,     self.size // 2),
        ]

        for r, c in centers:
            for i in range(partition_size):
                for j in range(partition_size):
                    fr, fc = r * partition_size + i, c * partition_size + j
                    dist[fr][fc] = 0
                    queue.append((fr, fc))

        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

        while queue:
            r, c = queue.popleft()
            for d, (dr, dc) in enumerate(directions):
                nr, nc = r + dr, c + dc
                
                # check for bounds
                if not (0 <= nr < fine_size and 0 <= nc < fine_size):
                    continue

                if dist[nr][nc] != -1:
                    continue
                
                # actual maze cell
                r_cell, c_cell = r // partition_size, c // partition_size
                nr_cell, nc_cell = nr // partition_size, nc // partition_size

                # movement inside same maze cell
                if r_cell == nr_cell and c_cell == nc_cell:
                    dist[nr][nc] = dist[r][c] + 1
                    queue.append((nr, nc))
                else:
                    if (not self.grid[r_cell][c_cell][d]) and (not self.grid[nr_cell][nc_cell][(d + 2) % 4]):
                        # the second argument in the AND is extra
                        dist[nr][nc] = dist[r][c] + 1
                        queue.append((nr, nc))

        return np.array(dist)
    
    def lidar(self, pos_x, pos_y, pos_theta):
        size = self.size
        x = pos_x + size / 2.0
        y = -pos_y + size / 2.0

        if x < 0 or x >= size or y < 0 or y >= size:
            return 0 

        dx = math.cos(pos_theta)
        dy = -math.sin(pos_theta) 

        if dx == 0:
            dx = 1e-10
        if dy == 0:
            dy = 1e-10

        cell_x = int(x)
        cell_y = int(y)

        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1

        # Distance to next vertical and horizontal grid lines
        t_max_x = ((cell_x + (step_x > 0)) - x) / dx
        t_max_y = ((cell_y + (step_y > 0)) - y) / dy

        # Distance between vertical/horizontal grid lines
        t_delta_x = abs(1 / dx)
        t_delta_y = abs(1 / dy)

        # Ray distance traveled
        dist = 0.0
        while 0 <= cell_x < size and 0 <= cell_y < size:
            if t_max_x < t_max_y:
                wall_idx = 1 if step_x > 0 else 3  # right or left
                if self.grid[cell_y][cell_x][wall_idx]:
                    dist = t_max_x
                    break
                cell_x += step_x
                t_max_x += t_delta_x
            else:
                wall_idx = 2 if step_y > 0 else 0  # down or up
                if self.grid[cell_y][cell_x][wall_idx]:
                    dist = t_max_y
                    break
                cell_y += step_y
                t_max_y += t_delta_y

        return dist