import numpy as np
from typing import Tuple, Optional


class WumpusWorld:
    """
    Wumpus World environment from AIMA textbook.
    
    The agent must navigate to the gold (G) while avoiding pits (P) and the Wumpus (W).
    Breezes (B) appear adjacent to pits. Stenches (S) appear adjacent to Wumpus.
    
    Actions:
        0: Left
        1: Down
        2: Right
        3: Up
        4: Fire arrow
    
    Grid is bottom-up: row 0 is bottom, row N-1 is top.
    Agent starts at (0, 0) - bottom left.
    """
    
    def __init__(self, grid_size: int = 4, max_actions=500, seed: Optional[int] = None):
        """
        Initialize Wumpus World.
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.num_actions = 5
        
        if seed is not None:
            np.random.seed(seed)
        
        # Grid: 0=safe, 1=pit, 2=wumpus, 3=gold
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Place max 2 pits randomly (avoid start and goal)
        num_pits = min(2, grid_size * grid_size - 3)
        pit_positions = np.random.choice(grid_size * grid_size - 2, size=num_pits, replace=False) + 1
        for pit_idx in pit_positions:
            pit_pos = self._state_to_pos(pit_idx)
            self.grid[pit_pos] = 1
        
        # Place Wumpus randomly
        while True:
            wumpus_pos = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
            if self.grid[wumpus_pos] == 0 and wumpus_pos != (0, 0) and wumpus_pos != (1, 1):
                self.grid[wumpus_pos] = 2
                self.wumpus_pos = wumpus_pos
                break
        
        # Place gold randomly
        while True:
            gold_pos = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
            if self.grid[gold_pos] == 0 and gold_pos != (0, 0) and gold_pos != (1, 1):
                self.grid[gold_pos] = 3
                self.gold_pos = gold_pos
                break
        
        self.start_pos = (0, 0)
        self.current_pos = self.start_pos
        self.arrow_available = True
        self.wumpus_alive = True
        self.has_gold = False
        
        self.action_counter = 0
        self.max_actions = max_actions  # To prevent infinite episodes
    
    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Convert (row, col) position to state index."""
        return pos[0] * self.grid_size + pos[1]
    
    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert state index to (row, col) position."""
        return (state // self.grid_size, state % self.grid_size)
    
    def _get_next_pos(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Get next position given current position and action."""
        row, col = pos
        
        if action == 0:  # Left
            col = max(0, col - 1)
        elif action == 1:  # Down
            row = max(0, row - 1)
        elif action == 2:  # Right
            col = min(self.grid_size - 1, col + 1)
        elif action == 3:  # Up
            row = min(self.grid_size - 1, row + 1)
        
        return (row, col)
    
    def _get_adjacent_positions(self, pos: Tuple[int, int]) -> list:
        """Get all adjacent positions (up, down, left, right)."""
        row, col = pos
        adjacent = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                adjacent.append((new_row, new_col))
        return adjacent
    
    def _has_breeze(self, pos: Tuple[int, int]) -> bool:
        """Check if position has a breeze (adjacent to pit)."""
        for adj_pos in self._get_adjacent_positions(pos):
            if self.grid[adj_pos] == 1:
                return True
        return False
    
    def _has_stench(self, pos: Tuple[int, int]) -> bool:
        """Check if position has a stench (adjacent to wumpus)."""
        if not self.wumpus_alive:
            return False
        for adj_pos in self._get_adjacent_positions(pos):
            if self.grid[adj_pos] == 2:
                return True
        return False
    
    def reset(self) -> int:
        """Reset environment and return initial state."""
        self.current_pos = self.start_pos
        self.arrow_available = True
        self.wumpus_alive = True
        self.has_gold = False
        self.action_counter = 0
        return self._pos_to_state(self.current_pos)
    
    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0-4)
        
        Returns:
            state: New state index
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        if not 0 <= action < self.num_actions:
            raise ValueError(f"Invalid action: {action}")
        
        reward = 0.0
        done = False
        
        self.action_counter += 1
        if self.action_counter >= self.max_actions:
            done = True
            reward = -1000.0  # Penalty for exceeding max steps
        elif action == 4:  # Fire arrow
            if self.arrow_available:
                reward -= 10.0
                self.arrow_available = False
                # We assume firing the arrow will kill the Wumpus if it's in the same row or column
                if self.wumpus_alive and (self.current_pos[0] == self.wumpus_pos[0] or self.current_pos[1] == self.wumpus_pos[1]):
                    self.wumpus_alive = False
            else:
                reward -= 1.0
        else:
            # Movement action
            next_pos = self._get_next_pos(self.current_pos, action)
            self.current_pos = next_pos
            
            # Check what's at this position
            if self.grid[next_pos] == 1:  # Pit
                reward = -1000.0
                done = True
            elif self.grid[next_pos] == 2 and self.wumpus_alive:  # Wumpus
                reward = -1000.0
                done = True
            elif self.grid[next_pos] == 3:  # Gold
                self.has_gold = True
                reward = -1.0
            else:
                reward = -1.0
            
            # Check if agent returned to (1,1) with gold
            if self.has_gold and self.current_pos == (1, 1):
                reward = 1000.0
                done = True
        
        state = self._pos_to_state(self.current_pos)
        info = {
            'agent_got_gold': self.has_gold,
            'agent_escaped': self.has_gold and self.current_pos == (1, 1),
            'eaten_by_wumpus': self.grid[self.current_pos] == 2 and self.wumpus_alive and action != 4,
            'fell_in_pit': self.grid[self.current_pos] == 1 and action != 4,
            'wumpus_killed': not self.wumpus_alive,
            'exceeded_max_steps': self.action_counter >= self.max_actions
        }
        return state, reward, done, info
    
    def translate_info_to_sentence(self, info: dict) -> str:
        """Translate info dictionary into a human-readable sentence."""

        status_str = ""
        # Non-terminal states
        if info.get('wumpus_killed'):
            status_str += "Agent fired an arrow and killed the Wumpus. "
        
        if info.get('agent_got_gold') and not info.get('agent_escaped'):
            status_str += "Agent got the gold but hasn't escaped yet. "
        
        # Terminal conditions (mutually exclusive)
        if info.get('exceeded_max_steps'):
            return status_str + "Agent exceeded max steps."
        elif info.get('agent_escaped'):
            return status_str + "Agent escaped with the gold!"
        elif info.get('eaten_by_wumpus'):
            return status_str + "Agent was eaten by the Wumpus."
        elif info.get('fell_in_pit'):
            return status_str + "Agent fell into a pit."
        else:
            return status_str + "Agent is still exploring."
        
    def render(self):
        """Print the current environment state."""
        display_grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                has_breeze = self._has_breeze((i, j))
                has_stench = self._has_stench((i, j))
                
                if self.grid[i, j] == 3:  # Gold overrides everything
                    display_grid[i][j] = 'G'
                elif self.grid[i, j] == 1:  # Pit
                    display_grid[i][j] = 'H'
                elif self.grid[i, j] == 2 and self.wumpus_alive:  # Wumpus
                    display_grid[i][j] = 'W'
                elif has_breeze and has_stench:  # Both breeze and stench
                    display_grid[i][j] = 'D'
                elif has_breeze:  # Just breeze
                    display_grid[i][j] = 'B'
                elif has_stench:  # Just stench
                    display_grid[i][j] = 'S'
        
        display_grid[self.current_pos[0]][self.current_pos[1]] = 'A'
        
        # Display grid bottom-up (reverse rows)
        print()
        for i in range(self.grid_size - 1, -1, -1):
            print(" ".join(display_grid[i]))
        print(f"Arrow: {self.arrow_available}, Wumpus: {self.wumpus_alive}, Has Gold: {self.has_gold}\n")


if __name__ == "__main__":
    env = WumpusWorld(grid_size=4, max_actions=500, seed=0)
    state = env.reset()
    env.render()
    
    done = False
    total_reward = 0
    while not done:
        action = np.random.randint(0, env.num_actions)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        action_name = ['Left', 'Down', 'Right', 'Up', 'Fire'][action]
        print(f"Action: {action_name}, Reward: {reward}, Total: {total_reward}")
        env.render()
    print(f"Episode finished with total reward: {total_reward}")