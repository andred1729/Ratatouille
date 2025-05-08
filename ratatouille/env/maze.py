import numpy as np
from collections import deque
import logging
import math

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
        self.cell = 0

    def load_maze(self, text_maze):
        tf = {"T": True, "F": False}
        if len(text_maze) != self.size:
            logger.error("not the right number of rows")
            return
        for row_idx, row_str in enumerate(text_maze):
            cells = row_str.split("|")
            if len(cells) != self.size:
                logger.error(f"not the right number of cols for row: {row_idx}")
                return
            grid_row = self.size - 1 - row_idx  # text_maze[0] -> grid[3]
            for col_idx in range(self.size):
                cell = cells[col_idx]
                if len(cell) != 4:
                    logger.error(f"not the right chars per cell for cell: {row_idx}, {col_idx}")
                    return
                for i, wall in enumerate(cell):
                    if wall not in tf:
                        logger.error(f"not the right char in cell: {row_idx}, {col_idx}")
                        return
                    self.grid[grid_row][col_idx][i] = tf[wall]
        self.and_maze()

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
                
    def check_collision(self, pos_x, pos_y, rad):
        half = self.size / 2.0
        # 1) world‐boundary
        if abs(pos_x) > half - rad or abs(pos_y) > half - rad:
            return True

        # 2) convert to grid coords
        maze_x = pos_x + half
        maze_y = pos_y + half
        cell_x = math.floor(maze_x)
        cell_y = math.floor(maze_y)

        # simple out‐of‐bounds catch
        if not (0 <= cell_x < self.size and 0 <= cell_y < self.size):
            return True

        walls   = self.grid[cell_y][cell_x]
        dx, dy  = maze_x - cell_x, maze_y - cell_y

        # —— horizontal boundaries ——
        # bottom edge of this cell
        if dy < rad:
            # bottom wall of current cell OR top wall of the cell *below*
            below = cell_y - 1
            if walls[2] or (below >= 0 and self.grid[below][cell_x][0]):
                return True

        # top edge of this cell
        if (1 - dy) < rad:
            # top wall of current cell OR bottom wall of the cell *above*
            above = cell_y + 1
            if walls[0] or (above < self.size and self.grid[above][cell_x][2]):
                return True

        # —— vertical boundaries (you can do the same trick for left/right) ——
        if dx < rad:
            left = cell_x - 1
            if walls[3] or (left >= 0 and self.grid[cell_y][left][1]):
                return True

        if (1 - dx) < rad:
            right = cell_x + 1
            if walls[1] or (right < self.size and self.grid[cell_y][right][3]):
                return True

        return False

    
    def check_win(self, pos_x, pos_y, rad):
        entrancefactor = 2 #how deep into the center does the rat have to go to win
        if abs(pos_x) <= (1 - entrancefactor*rad) and abs(pos_y) <= (1 - entrancefactor*rad):
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
