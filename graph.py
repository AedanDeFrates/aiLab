import numpy as np
import random


class Node:
    def __init__(self, i, coordinates):
        self.index = i
        self.location = coordinates
        
    def get_location(self):
        return self.location
    
    
class Edge:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        # We only use L2 distance for g(n) in A* as it is an exact measurement.
        self.distance = np.linalg.norm(np.array(node1.get_location()) - np.array(node2.get_location()))
        
    def get_nodes(self):
        return (self.node1, self.node2)

    def get_distance(self):
        return self.distance


class Graph:
    def __init__(self, n, k, dim=2, gridsize=100, seed=42):
        
        assert n > 2, "Number of nodes must be greater than 2."
        assert k < n, "Number of neighbors must be less than number of nodes."
        assert k > 0, "Number of neighbors must be greater than 0."
        assert 2 <= dim <= 3, "Dimension must be 2 or 3."
        
        self.n = n
        self.k = k
        self.dim = dim
        self.gridsize = gridsize
        self.nodes = {}
        self.edges = {}

        random.seed(seed)
        city_coordinates = [random.sample(range(gridsize), dim) for i in range(n)]

        for i in range(n):
            self.nodes[i] = Node(i, city_coordinates[i])

        # Create edges to k nearest neighbors for each node
        locations = np.array([self.nodes[i].get_location() for i in range(n)])

        for i in range(n):
            distances = np.linalg.norm(locations - locations[i], axis=1)
            # Deterministic tie-breaker: sort by distance, then by index.
            nearest_indices = np.lexsort((np.arange(n), distances))[1:self.k+1]
            for j in nearest_indices:
                edge = Edge(self.nodes[i], self.nodes[j])
                self.edges[(i, j)] = edge
                self.edges[(j, i)] = edge  # Add reverse edge for undirected graph
                
    def get_neighbors(self, node_index):
        neighbors = []
        for edge in self.edges.values():
            node1, node2 = edge.get_nodes()
            if node1.index == node_index:
                neighbors.append(node2.index)
            elif node2.index == node_index:
                neighbors.append(node1.index)
        return neighbors
                
                
if __name__ == "__main__":
    g = Graph(n=50, k=4, dim=2, gridsize=100, seed=42)
    print(f"Number of nodes: {len(g.nodes)}")
    print(f"Number of edges: {len(g.edges)}")
    
    import matplotlib.pyplot as plt

    # Extract coordinates
    coords = np.array([node.get_location() for node in g.nodes.values()])

    # Plot edges
    for edge_obj in g.edges.values():
        nodes = edge_obj.get_nodes()
        node1_coords = nodes[0].get_location()
        node2_coords = nodes[1].get_location()
        plt.plot([node1_coords[0], node2_coords[0]], 
                 [node1_coords[1], node2_coords[1]], 'gray', alpha=0.5)

    # Plot all nodes
    plt.scatter(coords[:, 0], coords[:, 1], c='lightblue', s=100, zorder=5)
    
    # Print node indices on the plot
    for node_obj in g.nodes.values():
        loc = node_obj.get_location()
        plt.text(loc[0], loc[1], node_obj.index, fontsize=8, color='grey', ha='center', va='center', zorder=7)

    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.title('Graph Visualization')
    plt.grid(True, alpha=0.3)
    plt.show()