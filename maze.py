import numpy as np

class Maze:
    def __init__(self, size=16):
        self.size = size
        # 0 = free cell, 1 = wall
        self.grid = np.ones((size, size), dtype=int)
        self._load_test_maze()


    def _load_test_maze(self):
        layout = [
            "1111111111111111",
            "1000000100000001",
            "1000000101111101",
            "1000010001000001",
            "1111011110111101",
            "1001000000100101",
            "1011011111101101",
            "1000010000000001",
            "1011110111111101",
            "1000000100000001",
            "1111101110111101",
            "1000100010001001",
            "1010111011101011",
            "1000000000000001",
            "1111111111111111",
            "1111111111111111"
        ]
        for y,row in enumerate(layout):
            self.grid[y,:] = [int(c) for c in row]

    def is_free(self, x, y):
        """Return True if (x,y) is inside bounds and not a wall."""
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        return self.grid[y, x] == 0
