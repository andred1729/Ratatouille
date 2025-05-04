class Maze:
    def __init__(self, size=16):
        """
        Initialize a Maze object.

        Parameters:
        ----------
        size : int, optional
            The size of the maze grid (number of rows and columns). Default is 16.

        Attributes:
        ----------
        grid : list of list of list of bool
            A 3D grid representing the maze, where grid[row][col] is a list of four booleans:
            [up, right, down, left], indicating whether each wall is present (True) or absent (False).
        """
        self.size = size
        self.grid = [[[False, False, False, False] * size] for _ in range(size)]

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
            print("not the right number of rows")
            return

        for row_idx in range(self.size):
            cells = text_maze[row_idx].split("|")

            if len(cells) != self.size:
                print("not the right number of cols for row: ", row_idx)
                return

            for col_idx in range(self.size):
                cell = cells[col_idx]

                if len(cell) != 4:
                    print("not the right chars per cell for cell: ", row_idx, col_idx)
                    return

                for i, wall in enumerate(cell):
                    if wall not in tf.keys():
                        print("not the right char in cell: ", row_idx, col_idx)
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
                self.grid[row_idx][col_idx-1][1] = self.grid[row_idx][col_idx-1][1] and self.grid[row_idx][col_idx][4]
                self.grid[row_idx][col_idx][4]   = self.grid[row_idx][col_idx-1][1] and self.grid[row_idx][col_idx][4]
        # north-south
        for col_idx in range(self.size):
            for row_idx in range(1, self.size-1):
                self.grid[row_idx-1][col_idx][3] = self.grid[row_idx-1][col_idx][3] and self.grid[row_idx][col_idx][0]
                self.grid[row_idx][col_idx][0] = self.grid[row_idx-1][col_idx][3] and self.grid[row_idx][col_idx][0]

    def load_4x4_test_maze(self):
        """
        Load a predefined 4x4 test maze.
        """
        layout = [
            "TFFT|TFTF|TTTF|TTFT",
            "FTFT|TFFT|TTFF|FTFT",
            "FTFT|FFFT|FTTF|FTTT",
            "FFTT|FTTF|TFTT|TTTF"
        ]
        self.load_maze(layout)
