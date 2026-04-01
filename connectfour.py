import numpy as np
import random
from enum import Enum
from typing import Optional, List


class Player(Enum):
    PLAYER1 = 1
    PLAYER2 = 2

class ConnectFour:
    def __init__(self, title="Connect Four", rows=6, cols=7, win_length=4):
        self.title = title
        self.rows = rows
        self.cols = cols
        self.win_length = win_length
        self.board = np.zeros((rows, cols), dtype=int)
        self.current_player = Player.PLAYER1
        
    def get_legal_actions(self):
        """Return list of legal column indices where a piece can be dropped."""
        legal_actions = []
        for col in range(self.cols):
            if self.board[0, col] == 0:
                legal_actions.append(col)
        return legal_actions
    
    def make_move(self, col):
        """Drop a piece in the specified column. Returns True if successful."""
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player.value
                return True
        return False
    
    def check_winner(self, player):
        """Check if the specified player has won."""
        player_val = player.value
        
        # Check horizontal
        for row in range(self.rows):
            for col in range(self.cols - self.win_length + 1):
                if all(self.board[row, col + i] == player_val for i in range(self.win_length)):
                    return True
        
        # Check vertical
        for col in range(self.cols):
            for row in range(self.rows - self.win_length + 1):
                if all(self.board[row + i, col] == player_val for i in range(self.win_length)):
                    return True
        
        # Check diagonal (top-left to bottom-right)
        for row in range(self.rows - self.win_length + 1):
            for col in range(self.cols - self.win_length + 1):
                if all(self.board[row + i, col + i] == player_val for i in range(self.win_length)):
                    return True
        
        # Check diagonal (top-right to bottom-left)
        for row in range(self.rows - self.win_length + 1):
            for col in range(self.win_length - 1, self.cols):
                if all(self.board[row + i, col - i] == player_val for i in range(self.win_length)):
                    return True
        
        return False
    
    def is_terminal(self):
        """Check if the game is over."""
        if self.check_winner(Player.PLAYER1) or self.check_winner(Player.PLAYER2):
            return True
        return len(self.get_legal_actions()) == 0
    
    def switch_player(self):
        """Switch to the other player."""
        self.current_player = Player.PLAYER2 if self.current_player == Player.PLAYER1 else Player.PLAYER1
    
    def copy(self):
        """Return a deep copy of the game state."""
        new_game = ConnectFour(self.title, self.rows, self.cols, self.win_length)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game
    
    def __str__(self):
        """String representation of the board."""
        return str(self.board)


class MCTSNode:
    """Node in the Monte Carlo Tree Search for Connect Four."""
    
    def __init__(self, state: ConnectFour, parent: Optional['MCTSNode'] = None, action: Optional[int] = None):
        """
        Initialize an MCTS node.
        
        Args:
            state: ConnectFour game state
            parent: Parent node in the search tree
            action: Column index that led to this node
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List['MCTSNode'] = []
        self.visit_count = 0
        self.total_reward = 0.0
        self.untried_actions = self._get_legal_actions()
    
    def _get_legal_actions(self) -> List[int]:
        """Get all legal actions from this state."""
        return self.state.get_legal_actions()
    
    def is_terminal(self) -> bool:
        """Check if this node represents a terminal state."""
        return self.state.is_terminal()
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions from this node have been tried."""
        return len(self.untried_actions) == 0
    

# ---------- Game Playing Wrapper ----------
def play_connectfour(game, mcts1, mcts2=None, num_iterations: int = 1000, verbose: bool = True):
    """
    Play Connect Four with MCTS agents.
    
    Args:
        mcts1: MCTS agent for PLAYER1
        mcts2: MCTS agent for PLAYER2 (if None, uses mcts1 for both players)
        num_iterations: Number of MCTS iterations per turn
        verbose: Whether to print the board after each move
    
    Returns:
        Winner: Player.PLAYER1, Player.PLAYER2, or None (draw)
    """
    
    move_count = 0
    max_moves = 100
    
    while not game.is_terminal() and move_count < max_moves:
        if verbose:
            print(f"--- {game.title} ---")
            print(f"--- Move {move_count + 1} ---")
            print(f"Current player: {game.current_player}")
            print(game)
        
        # Choose MCTS based on current player
        if game.current_player == Player.PLAYER1:
            mcts = mcts1
            best_node = mcts.search(num_iterations)
            best_move = best_node.action
        else:
            if mcts2 is not None:
                mcts = mcts2
                best_node = mcts.search(num_iterations)
                best_move = best_node.action
            else:
                # Random move for PLAYER2 if mcts2 not provided
                best_move = random.choice(game.get_legal_actions())
        
        if verbose:
            print(f"Best move: Column {best_move}")
        
        # Make move
        game.make_move(best_move)
        game.switch_player()
        move_count += 1
    
    # Determine winner
    if game.check_winner(Player.PLAYER1):
        winner = Player.PLAYER1
        if verbose:
            print(f"\n{'='*40}\nPLAYER1 wins!\n{'='*40}")
    elif game.check_winner(Player.PLAYER2):
        winner = Player.PLAYER2
        if verbose:
            print(f"\n{'='*40}\nPLAYER2 wins!\n{'='*40}")
    else:
        winner = None
        if verbose:
            print(f"\n{'='*40}\nDraw!\n{'='*40}")
    
    if verbose:
        print(f"\nFinal board:")
        print(game)
    
    return winner