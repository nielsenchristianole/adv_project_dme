import numpy as np
import heapq
import time


# ---------------------------------------------------------------------------- #
#                                   Heuristcs                                  #
# ---------------------------------------------------------------------------- #
class Heuristic:
    def __init__(self, heightmap_m, km_per_px=1.0):
        """
        Initialize the Euclidean heuristic with the heightmap and a scaling factor.
        
        Parameters:
        - heightmap: 2D array of height values in km.
        - km_per_px: The distance represented by each pixel in the heightmap (in km).
        """
        self.heightmap_km = heightmap_m / 1000
        self.km_per_px = km_per_px
        
    def h(self, state, action, goal):
        """
        Returns heuristic cost; to be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class EuclideanHeuristic(Heuristic):
    def h(self, state, action, goal):
        """
        Calculates a 3D Euclidean distance heuristic that considers height differences.
        
        Parameters:
        - state: The current position as (y, x) in the grid.
        - action: The movement direction (not used here but kept for consistency).
        - goal: The goal position as (y, x) in the grid.
        
        Returns:
        - Heuristic cost: a 3D Euclidean distance scaled by km_per_px and including height.
        """
        # Get the 2D distance in pixels
        dy = (goal[0] - state[0]) * self.km_per_px
        dx = (goal[1] - state[1]) * self.km_per_px
        # Get the height difference
        dz = self.heightmap[goal[1], goal[0]] - self.heightmap[state[1], state[0]]
        
        # Calculate the 3D Euclidean distance
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
class BridgeEuclideanHeuristic(Heuristic):
    def __init__(self, heightmap, km_per_px, cost_per_km, max_slope_cost_factor, bridge_cost_factor):
        super().__init__(heightmap, km_per_px)
        self.bridge_cost_factor = bridge_cost_factor
        self.cost_per_km = cost_per_km
        self.max_slope_cost_factor = max_slope_cost_factor
        
    def h(self, state, action, goal):
        """
        Calculates a 3D Euclidean distance heuristic that considers height differences.
        
        Parameters:
        - state: The current position as (y, x) in the grid.
        - action: The movement direction (not used here but kept for consistency).
        - goal: The goal position as (y, x) in the grid.
        
        Returns:
        - Heuristic cost: a 3D Euclidean distance scaled by km_per_px and including height.
        """
        # Get the 2D distance in pixels
        dy = (goal[0] - state[0]) * self.km_per_px
        dx = (goal[1] - state[1]) * self.km_per_px
        
        goal_height = self.heightmap_km[goal[1], goal[0]]
        cur_height = self.heightmap_km[state[1], state[0]]
        
        # if bridge
        if goal_height <= 0 or cur_height <= 0:
            return np.sqrt(dx**2 + dy**2) * self.cost_per_km * self.bridge_cost_factor
            
        else:
            # Get the height difference
            dz = goal_height - cur_height
            horizontal_distance = np.sqrt(dx**2 + dy**2)

            # Calculate the slope angle (in radians) and clamp it to [0, pi/2], disregard negative slopes
            slope_angle = np.arctan2(abs(dz), horizontal_distance)
            slope_angle = min(slope_angle, np.pi / 2)

            # Scale slope cost quadratically
            slope_cost_factor = 1 + ((slope_angle / (np.pi / 2))**2) * (self.max_slope_cost_factor - 1)

            # 3D Euclidean distance with slope factor
            return np.sqrt(horizontal_distance**2 + dz**2) * self.cost_per_km * slope_cost_factor


# ---------------------------------------------------------------------------- #
#                               Search Algorithms                              #
# ---------------------------------------------------------------------------- #
class SearchAlgo:
    def __init__(self, heightmap):
        self.heightmap = heightmap
        self.width, self.height = heightmap.shape
    
    def find_path(self, start, end):
        """
        Empty method meant to be overridden by subclasses. 
        Should return a path and unused actions variable.
        """
        raise NotImplementedError("This method should be overridden in a subclass.")

    def snap_to_grid(self, point):
        """
        Snap floating-point coordinates to the nearest integer grid point.
        """
        return int(round(point[0])), int(round(point[1]))

    def plot_path(self, start, end, path):
        """
        Plots the heightmap, start and end points, and the path on the map.
        """
        plt.imshow(self.heightmap, cmap='terrain', origin='upper') 
        plt.colorbar(label="Height")
        
        # Plot start and end points
        plt.plot(start[0], start[1], "go", label="Start")  # Start point in green
        plt.plot(end[0], end[1], "ro", label="End")  # End point in red
        
        # Plot the path
        if path:
            path_x, path_y = zip(*path)
            plt.plot(path_x, path_y, "b-", label="Path")  # Path in blue
        
        plt.legend()
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Heightmap with Path")
        plt.show()

class AStar(SearchAlgo):
    def __init__(self, heightmap, heuristic, max_runtime=None):
        super().__init__(heightmap)
        self.heuristic: Heuristic = heuristic
        self.max_runtime = max_runtime

    def find_path(self, start, end):
        start = self.snap_to_grid(start)
        end = self.snap_to_grid(end)
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic.h(start, None, end)}
        
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal directions
            (-1, -1), (1, 1), (-1, 1), (1, -1)  # Diagonals
        ]
        
        start_time = time.time()
        
        while open_set:
            if self.max_runtime and (time.time() - start_time) > self.max_runtime:
                return 'timeout', None
                
            _, current = heapq.heappop(open_set)

            if current == end:
                return self.reconstruct_path(came_from, current), _
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not (0 <= neighbor[0] < self.height and 0 <= neighbor[1] < self.width):
                    continue

                tentative_g_score = g_score[current] + self.heuristic.h(current, (dx, dy), neighbor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic.h(neighbor, (dx, dy), end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None, _  # Return None if no path is found
    
    def reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path


class GreedyBestFirst(SearchAlgo):
    def __init__(self, heightmap, heuristic):
        self.heightmap = heightmap
        self.heuristic = heuristic

    def find_path(self, start, goal):
        start = self.snap_to_grid(start)
        goal = self.snap_to_grid(goal)

        open_set = []
        heapq.heappush(open_set, (self.heuristic.h(start, None, goal), start))
        came_from = {start: None}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            # Check if we've reached the goal
            if current == goal:
                return self._reconstruct_path(came_from, current), _
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                if neighbor not in came_from:  # Only add unvisited nodes
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (self.heuristic.h(neighbor, None, goal), neighbor))
                    
        return None, _  # Return None if no path is found

    def _get_neighbors(self, node):
        """
        Retrieve valid neighbors for the node, considering grid boundaries.
        """
        y, x = node
        neighbors = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.heightmap.shape[0] and 0 <= nx < self.heightmap.shape[1]:
                neighbors.append((ny, nx))
        return neighbors

    def _reconstruct_path(self, came_from, current):
        """
        Reconstructs the path from start to goal by backtracking from the goal.
        """
        path = []
        while current:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path



if __name__ == '__main__':
    from pathlib import Path
    import matplotlib.pyplot as plt
    
    # Load stuff
    patch_path = Path('data/undistorted_data_ortho_2/data/36/82')
    height_map = np.load(patch_path / 'height_map.npy')
    height_map = height_map / 1000
    
    # Example usage:
    heuristic = BridgeEuclideanHeuristic(heightmap=height_map, km_per_px=0.5, bridge_cost_factor=2.0)
    astar = AStar(height_map, heuristic)
    start_point = (365,293)
    goal_point = (380, 380)
    path, actions = astar.find_path(start_point, goal_point)
    astar.plot_path(start_point, goal_point, path)
    
    
    greedy = GreedyBestFirst(height_map, heuristic)
    path, actions = greedy.find_path(start_point, goal_point)
    greedy.plot_path(start_point, goal_point, path)
    
    import cv2
    import shapely
    import matplotlib.cm as cm
    
    norm_height_map = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))
    colormap = cm.get_cmap('terrain')
    rgb_image = colormap(norm_height_map)
    
    line = shapely.LineString(path)
    img = cv2.polylines(rgb_image, [np.array(line.xy, dtype=np.int32).T.reshape((-1,1,2))], isClosed=False, color=(1.0, 0, 1.0), thickness=2)    
    cv2.imshow("Polylines", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
