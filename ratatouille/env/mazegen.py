"""
This is CHATGPT code that's used to automatically generate mazes.

This Maze class is not the same as the Maze class in maze.py
"""
import random
import math
from collections import deque

class Maze:
    def __init__(self, size, temperature=0.5):
        """
        Initialize a maze of given size with a temperature parameter.
        
        Args:
            size: Size of the maze (size x size)
            temperature: Value between 0 and 1, higher means more walls
        """
        self.size = size
        self.temperature = temperature
        
        # Initialize the maze with walls
        # Each cell has 4 walls: up, right, down, left (URDL)
        self.maze = []
        for _ in range(size):
            row = []
            for _ in range(size):
                # Default: all walls present (T, T, T, T)
                row.append([True, True, True, True])
            self.maze.append(row)
        
        # Generate the maze
        self.generate_maze()
        
    def generate_maze(self):
        """Generate a random maze using the temperature parameter."""
        # Randomly remove walls based on temperature
        for row in range(self.size):
            for col in range(self.size):
                for wall_idx in range(4):
                    # Skip external walls
                    if (wall_idx == 0 and row == 0) or \
                       (wall_idx == 1 and col == self.size - 1) or \
                       (wall_idx == 2 and row == self.size - 1) or \
                       (wall_idx == 3 and col == 0):
                        continue
                    
                    # Apply temperature probability
                    if random.random() > self.temperature:
                        # Remove this wall
                        self.remove_wall(row, col, wall_idx)
        
        # Handle center cells requirement: 7 out of 8 external walls must be present
        self.handle_center_cells()
        
        # Ensure there's a path from center to bottom-left
        self.ensure_path_to_bottom_left()
    
    def remove_wall(self, row, col, wall_idx):
        """Remove a wall and its corresponding wall in the adjacent cell."""
        self.maze[row][col][wall_idx] = False
        
        # Remove the corresponding wall in the adjacent cell
        adj_row, adj_col = row, col
        adj_wall_idx = wall_idx
        
        if wall_idx == 0:  # Up
            adj_row -= 1
            adj_wall_idx = 2  # Down
        elif wall_idx == 1:  # Right
            adj_col += 1
            adj_wall_idx = 3  # Left
        elif wall_idx == 2:  # Down
            adj_row += 1
            adj_wall_idx = 0  # Up
        elif wall_idx == 3:  # Left
            adj_col -= 1
            adj_wall_idx = 1  # Right
        
        # Check if adjacent cell is within the maze bounds
        if 0 <= adj_row < self.size and 0 <= adj_col < self.size:
            self.maze[adj_row][adj_col][adj_wall_idx] = False
    
    def handle_center_cells(self):
        """
        Ensure:
        1. 7 out of 8 external walls around the center cells are present
        2. No internal walls between the 4 center cells
        """
        # Find the center cells
        center_start = self.size // 2 - 1
        center_end = self.size // 2
        
        # Get all external walls of the center cells
        external_walls = []
        
        # Top walls of top center cells
        for col in range(center_start, center_end + 1):
            external_walls.append((center_start, col, 0))
        
        # Right walls of right center cells
        for row in range(center_start, center_end + 1):
            external_walls.append((row, center_end, 1))
        
        # Bottom walls of bottom center cells
        for col in range(center_start, center_end + 1):
            external_walls.append((center_end, col, 2))
        
        # Left walls of left center cells
        for row in range(center_start, center_end + 1):
            external_walls.append((row, center_start, 3))
        
        # First, make all external walls present
        for row, col, wall_idx in external_walls:
            self.maze[row][col][wall_idx] = True
            
            # Update the adjacent cell's wall too
            adj_row, adj_col = row, col
            adj_wall_idx = wall_idx
            
            if wall_idx == 0:  # Up
                adj_row -= 1
                adj_wall_idx = 2  # Down
            elif wall_idx == 1:  # Right
                adj_col += 1
                adj_wall_idx = 3  # Left
            elif wall_idx == 2:  # Down
                adj_row += 1
                adj_wall_idx = 0  # Up
            elif wall_idx == 3:  # Left
                adj_col -= 1
                adj_wall_idx = 1  # Right
            
            # Check if adjacent cell is within the maze bounds
            if 0 <= adj_row < self.size and 0 <= adj_col < self.size:
                self.maze[adj_row][adj_col][adj_wall_idx] = True
        
        # Remove all internal walls between center cells
        
        # Remove horizontal walls between center cells (top row)
        self.remove_wall(center_start, center_start, 1)  # Right wall of top-left
        
        # Remove horizontal walls between center cells (bottom row)
        self.remove_wall(center_end, center_start, 1)    # Right wall of bottom-left
        
        # Remove vertical walls between center cells
        self.remove_wall(center_start, center_start, 2)  # Bottom wall of top-left
        self.remove_wall(center_start, center_end, 2)    # Bottom wall of top-right
        
        # Then randomly pick one external wall to remove
        wall_to_open = random.choice(external_walls)
        row, col, wall_idx = wall_to_open
        self.remove_wall(row, col, wall_idx)
    
    def ensure_path_to_bottom_left(self):
        """Ensure there's a path from any center cell to the bottom-left cell."""
        center_start = self.size // 2 - 1
        center_end = self.size // 2
        
        # First check if a path already exists
        if self.path_exists_to_bottom_left():
            return
        
        # If no path exists, create one
        # Start from a random center cell
        center_cells = []
        for row in range(center_start, center_end + 1):
            for col in range(center_start, center_end + 1):
                center_cells.append((row, col))
        
        start_row, start_col = random.choice(center_cells)
        target_row, target_col = self.size - 1, 0  # Bottom-left cell
        
        # Use a simple approach: carve a path in general direction of the target
        current_row, current_col = start_row, start_col
        
        while current_row != target_row or current_col != target_col:
            # Decide whether to move horizontally or vertically
            if random.random() < 0.5 and current_row != target_row:
                # Move vertically
                next_row = current_row + (1 if current_row < target_row else -1)
                next_col = current_col
                
                # Remove walls between current and next
                if next_row > current_row:  # Moving down
                    self.remove_wall(current_row, current_col, 2)
                else:  # Moving up
                    self.remove_wall(current_row, current_col, 0)
                
                current_row = next_row
            elif current_col != target_col:
                # Move horizontally
                next_row = current_row
                next_col = current_col + (-1 if current_col > target_col else 1)
                
                # Remove walls between current and next
                if next_col > current_col:  # Moving right
                    self.remove_wall(current_row, current_col, 1)
                else:  # Moving left
                    self.remove_wall(current_row, current_col, 3)
                
                current_col = next_col
            else:
                # Move vertically since we're aligned horizontally
                next_row = current_row + (1 if current_row < target_row else -1)
                next_col = current_col
                
                # Remove walls between current and next
                if next_row > current_row:  # Moving down
                    self.remove_wall(current_row, current_col, 2)
                else:  # Moving up
                    self.remove_wall(current_row, current_col, 0)
                
                current_row = next_row
    
    def path_exists_to_bottom_left(self):
        """Check if there's a path from any center cell to the bottom-left cell."""
        center_start = self.size // 2 - 1
        center_end = self.size // 2
        
        # Start BFS from each center cell
        for start_row in range(center_start, center_end + 1):
            for start_col in range(center_start, center_end + 1):
                if self.bfs_to_bottom_left(start_row, start_col):
                    return True
        
        return False
    
    def bfs_to_bottom_left(self, start_row, start_col):
        """Use BFS to find a path from given start to bottom-left cell."""
        target_row, target_col = self.size - 1, 0
        visited = set()
        queue = deque([(start_row, start_col)])
        visited.add((start_row, start_col))
        
        while queue:
            row, col = queue.popleft()
            
            if row == target_row and col == target_col:
                return True
                
            # Try all four directions
            directions = [(0, -1, 3), (0, 1, 1), (-1, 0, 0), (1, 0, 2)]  # Left, Right, Up, Down
            
            for dr, dc, wall_idx in directions:
                # If there's no wall in this direction
                if not self.maze[row][col][wall_idx]:
                    next_row, next_col = row + dr, col + dc
                    
                    if 0 <= next_row < self.size and 0 <= next_col < self.size and (next_row, next_col) not in visited:
                        queue.append((next_row, next_col))
                        visited.add((next_row, next_col))
        
        return False
    
    def to_string_format(self):
        """Convert the maze to the specified string format."""
        result = []
        
        for row in range(self.size):
            row_str = ""
            for col in range(self.size):
                cell_walls = "".join("T" if wall else "F" for wall in self.maze[row][col])
                row_str += cell_walls
                if col < self.size - 1:
                    row_str += "|"
            result.append(row_str)
        
        return result

# Generate mazes of different sizes with different temperatures
def generate_mazes(sizes=[6, 8, 16], temperatures=[0.3, 0.5, 0.7]):
    """Generate mazes of specified sizes with different temperatures."""
    all_mazes = {}
    
    for size in sizes:
        size_mazes = {}
        for temp in temperatures:
            maze = Maze(size, temperature=temp)
            size_mazes[temp] = maze.to_string_format()
        all_mazes[size] = size_mazes
    
    return all_mazes

if __name__ == "__main__":
    # Generate the mazes
    maze_results = generate_mazes()

    # Print results
    for size, temp_mazes in maze_results.items():
        print(f"{size}x{size} Maze:")
        for temp, maze_str in temp_mazes.items():
            print(f"  Temperature: {temp}")
            print(f"  Maze: {maze_str}")
            print()