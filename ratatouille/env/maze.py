import numpy as np
from collections import deque
import logging
logger = logging.getLogger(__name__)
class Maze:
    def __init__(self, size, text_maze):
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
        self.dist = self.compute_min_dist_to_center()
        logger.info(np.array(self.grid).shape)

    def load_maze(self, text_maze):
        """
        Load maze layout from a text-based description.

        Parameters:
        ----------
        text_maze : list of str
            A list of strings, each representing a row in the maze.
            Each row string contains cell descriptions separated by '|'.
            Each cell description is a 4-character string of 'T' (True) or 'F' (False)
            representing walls [up, right, down, left].

        Notes:
        -----
        - This method validates the input for correct row/column counts and wall characters.
        - After loading, it calls `and_maze()` to ensure walls are mutually consistent between adjacent cells.
        """
        tf = {"T" : True, "F" : False}

        if len(text_maze) != self.size:
            logger.error("not the right number of rows")
            return

        for row_idx in range(self.size):
            cells = text_maze[row_idx].split("|")

            if len(cells) != self.size:
                logger.error(f"not the right number of cols for row: {row_idx}")
                return

            for col_idx in range(self.size):
                cell = cells[col_idx]

                if len(cell) != 4:
                    logger.error(f"not the right chars per cell for cell: {row_idx}, {col_idx}")
                    return

                for i, wall in enumerate(cell):
                    if wall not in tf.keys():
                        logger.error(f"not the right char in cell: {row_idx}, {col_idx}")
                        return

                    self.grid[row_idx][col_idx][i] = tf[wall]

        self.and_maze()

    def and_maze(self):
        """
        Ensure wall consistency between adjacent cells.
        """
        # east-west
        for row_idx in range(self.size):
            for col_idx in range(1, self.size-1):
                self.grid[row_idx][col_idx-1][1] = self.grid[row_idx][col_idx-1][1] and self.grid[row_idx][col_idx][3]
                self.grid[row_idx][col_idx][3]   = self.grid[row_idx][col_idx-1][1] and self.grid[row_idx][col_idx][3]
        # north-south
        for col_idx in range(self.size):
            for row_idx in range(1, self.size-1):
                self.grid[row_idx-1][col_idx][2] = self.grid[row_idx-1][col_idx][2] and self.grid[row_idx][col_idx][0]
                self.grid[row_idx][col_idx][0]   = self.grid[row_idx-1][col_idx][2] and self.grid[row_idx][col_idx][0]

    def check_collision(self, pos_x, pos_y, rad):
        #rad = robot radius

        if rad > 0.4:
            logger.error("Robot way too fat")
            return True

        abs_pos_x = pos_x - float(self.size / 1)
        abs_pos_y = pos_y - float(self.size / 1)

        square_x = int(abs_pos_x)
        square_y = int(abs_pos_y)

        cell = self.grid[square_x][square_y]

        if square_x < 0 or square_x >= self.size or square_y < 0 or square_y >= self.size:
            logger.warning("Warning: robot out of bounds")
            return True  # Treat out-of-bounds as collision

        dist_top    = abs_pos_y - square_y
        dist_right  = (square_x + 1) - abs_pos_x
        dist_bottom = (square_y + 1) - abs_pos_y
        dist_left   = abs_pos_x - square_x

        if cell[0] and dist_top < rad:
            return True 
        if cell[1] and dist_right < rad:
            return True 
        if cell[2] and dist_bottom < rad:
            return True 
        if cell[3] and dist_left < rad:
            return True

    def compute_min_dist_to_center(self):
        dist = [[-1 for _ in range(self.size)] for _ in range(self.size)]
        queue = deque()

        # Identify the 4 center cells
        centers = [
            (self.size // 2 - 1, self.size // 2 - 1),
            (self.size // 2 - 1, self.size // 2),
            (self.size // 2,     self.size // 2 - 1),
            (self.size // 2,     self.size // 2),
        ]

        for r, c in centers:
            dist[r][c] = 0
            queue.append((r, c))

        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

        while queue:
            r, c = queue.popleft()
            for d, (dr, dc) in enumerate(directions):
                nr, nc = r + dr, c + dc
                # Bounds check
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    # Wall check: there must be no wall between (r, c) and (nr, nc)
                    if not self.grid[r][c][d] and dist[nr][nc] == -1:
                        # Also check the opposite wall on the neighbor
                        if not self.grid[nr][nc][(d + 2) % 4]:
                            dist[nr][nc] = dist[r][c] + 1
                            queue.append((nr, nc))

        return np.array(dist)
