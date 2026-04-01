import numpy as np
from graph import Node as Point
import random


class ScatterPlain:
    def __init__(self, n, dim=2, gridsize=100, loop=True, seed=42):
        self.n = n
        self.dim = dim
        self.gridsize = gridsize
        self.loop = loop
        self.seed = seed
        self.points = []
        
        random.seed(seed)
        city_coordinates = [random.sample(range(gridsize), dim) for i in range(n)]
        
        for i in range(n):
            self.points.append(Point(i, city_coordinates[i]))
            
    def return_n_point_distance(self, point_indices):
        """Return total distance of path through given point indices."""
        total_distance = 0.0
        for i in range(len(point_indices) - 1):
            p1 = self.points[point_indices[i]]
            p2 = self.points[point_indices[i + 1]]
            loc1 = np.array(p1.get_location())
            loc2 = np.array(p2.get_location())
            total_distance += np.linalg.norm(loc1 - loc2)
        if self.loop:
            p1 = self.points[point_indices[-1]]
            p2 = self.points[point_indices[0]]
            loc1 = np.array(p1.get_location())
            loc2 = np.array(p2.get_location())
            total_distance += np.linalg.norm(loc1 - loc2)
        return total_distance

    def fitness(self, point_indices):
        """Return negative distance (higher is better) for given point indices."""
        return -self.return_n_point_distance(point_indices)
            
            
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create a scatter plain with 10 nodes in 2D
    sp = ScatterPlain(n=50, dim=2, gridsize=100, loop=True, seed=42)
    
    # Extract coordinates for plotting
    coords = np.array([node.get_location() for node in sp.points])
    
    # Plot the nodes
    plt.figure(figsize=(8, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c='lightblue', s=100, zorder=5)
    
    # Pick a random 5-point route and draw it
    route_indices = random.sample(range(sp.n), 5)
    distance = sp.return_n_point_distance(route_indices)
    print(f"Random route indices: {route_indices}, Total distance: {distance:.2f}")
    
    route_indices.append(route_indices[0])
    route_coords = np.array([sp.points[i].get_location() for i in route_indices])
    plt.plot(route_coords[:, 0], route_coords[:, 1], 'r-', linewidth=2, zorder=3)
    plt.scatter(route_coords[:-1, 0], route_coords[:-1, 1], c='red', s=100, zorder=6)
    for idx, node_obj in enumerate(sp.points):
        loc = node_obj.get_location()
        plt.text(loc[0], loc[1], node_obj.index, fontsize=8, color='grey', ha='center', va='center', zorder=7)
    
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.title('Scatter Plain Visualization')
    plt.grid(True, alpha=0.3)
    plt.show()