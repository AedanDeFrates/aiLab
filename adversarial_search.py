import random
from typing import Optional


class GameTree:
    """Tree-based adversarial search with minimax and alpha-beta pruning."""
    
    def __init__(self, depth: int, branch_factor: int = 2, max_first: bool = True, 
                 min_value: int = -10, max_value: int = 10):
        """
        Initialize a game tree for adversarial search.
        
        Args:
            depth: Depth of the tree
            branch_factor: Number of children per node
            max_first: If True, MAX plays at root; if False, MIN plays
            min_value: Minimum random value for terminal nodes
            max_value: Maximum random value for terminal nodes
        """
        self.depth = depth
        self.branch_factor = branch_factor
        self.max_first = max_first
        self.min_value = min_value
        self.max_value = max_value
        self.root = self._build_tree(0)
    
    def _build_tree(self, current_depth: int) -> 'Node':
        """Build tree recursively with random terminal values."""
        is_max = (current_depth % 2 == 0) if self.max_first else (current_depth % 2 == 1)
        
        if current_depth == self.depth:
            # Terminal node with random value
            value = random.randint(self.min_value, self.max_value)
            return Node(value=value, is_max=is_max, is_terminal=True)
        
        # Internal node
        children = [self._build_tree(current_depth + 1) for _ in range(self.branch_factor)]
        return Node(children=children, is_max=is_max)
    
    def minimax(self, node: Optional['Node'] = None) -> int:
        """Minimax algorithm without pruning."""
        if node is None:
            node = self.root
        
        if node.is_terminal:
            return node.value
        
        if node.is_max:
            return max(self.minimax(child) for child in node.children)
        else:
            return min(self.minimax(child) for child in node.children)
    
    def alpha_beta(self, node: Optional['Node'] = None, alpha: int = float('-inf'), 
                   beta: int = float('inf')) -> int:
        """Minimax with alpha-beta pruning."""
        if node is None:
            node = self.root
        
        if node.is_terminal:
            return node.value
        
        if node.is_max:
            for child in node.children:
                alpha = max(alpha, self.alpha_beta(child, alpha, beta))
                if beta <= alpha:
                    break  # Beta cutoff
            return alpha
        else:
            for child in node.children:
                beta = min(beta, self.alpha_beta(child, alpha, beta))
                if beta <= alpha:
                    break  # Alpha cutoff
            return beta


class Node:
    """Node in the game tree."""
    
    def __init__(self, value: Optional[int] = None, is_max: bool = True, 
                 is_terminal: bool = False, children: Optional[list] = None):
        self.value = value
        self.is_max = is_max
        self.is_terminal = is_terminal
        self.children = children or []
        
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Create a sample game tree and run both algorithms
    # 4 is a good depth for visualization
    tree = GameTree(depth=4, branch_factor=2, max_first=True)

    # Run minimax
    minimax_result = tree.minimax()
    print(f"Minimax result: {minimax_result}")

    # Run alpha-beta pruning
    alpha_beta_result = tree.alpha_beta()
    print(f"Alpha-beta result: {alpha_beta_result}")

    # Visualize tree structure with matplotlib
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    def draw_tree(node, x, y, offset, depth=0):
        """Recursively draw tree nodes and edges."""
        if node.is_terminal:
            color = 'lightgreen'
            label = str(node.value)
        else:
            color = 'lightblue' if node.is_max else 'lightcoral'
            label = "MAX" if node.is_max else "MIN"
        
        # Draw node
        circle = patches.Circle((x, y), 0.3, color=color, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, weight='bold')
        
        # Draw children
        if not node.is_terminal:
            child_offset = offset / 2
            for i, child in enumerate(node.children):
                child_x = x + (i - len(node.children) / 2 + 0.5) * child_offset
                child_y = y - 1.5
                
                # Draw edge
                ax.plot([x, child_x], [y - 0.3, child_y + 0.3], 'k-', linewidth=1)
                
                draw_tree(child, child_x, child_y, child_offset, depth + 1)
    
    draw_tree(tree.root, 5, 9, 6)
    plt.title("Game Tree Visualization (Minimax & Alpha-Beta Pruning)", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()