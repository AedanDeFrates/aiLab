import numpy as np


class AbstractHeuristic:
    def __init__(self, goal_node):
        self.goal_node = goal_node
        
    def __call__(self, node1, node2):
        raise NotImplementedError("Subclasses must implement this method.")
    
    
class EuclideanHeuristic(AbstractHeuristic):
    def __init__(self, goal_node):
        super().__init__(goal_node)
    
    def __call__(self, node):
        loc1 = np.array(node.get_location())
        loc2 = np.array(self.goal_node.get_location())
        return np.linalg.norm(loc1 - loc2)
    

class ManhattanHeuristic(AbstractHeuristic):
    def __init__(self, goal_node):
        super().__init__(goal_node)
    
    def __call__(self, node):
        loc1 = np.array(node.get_location())
        loc2 = np.array(self.goal_node.get_location())
        return np.sum(np.abs(loc1 - loc2))
    
    
class ChebyshevHeuristic(AbstractHeuristic):
    def __init__(self, goal_node):
        super().__init__(goal_node)
    
    def __call__(self, node):
        loc1 = np.array(node.get_location())
        loc2 = np.array(self.goal_node.get_location())
        return np.max(np.abs(loc1 - loc2))