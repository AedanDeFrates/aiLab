from __future__ import division
import math
import random
import numpy as np
import utilities as utils
from connectfour import MCTSNode, ConnectFour, Player
from wumpus_world import WumpusWorld


# ---------- Q1: Search ----------
def bfs(graph, start_node, goal_node):

    queue = [(start_node, [start_node])]
    visited = [start_node]

    while queue != []:
        v, path = queue.pop(0)

        if v == goal_node:
            return path

        for u in graph.get_neighbors(v):
            if visited.__contains__(u) == False:
                visited.append(u)
                queue.append((u, path + [u]))

    return []



def dfs(graph, start_node, goal_node):
    stack = [(start_node, [start_node])]
    visited = [start_node]

    while stack != []:
        v, path = stack.pop()

        if v == goal_node:
            return path

        for u in graph.get_neighbors(v):
            if visited.__contains__(u) == False:
                visited.append(u)
                stack.append((u, path + [u]))
    return []

def astar(graph, start_node, goal_node, heuristic):
    queue = [(start_node, [start_node], 0, 0)] # curr node , path, f_currNode, g_currNode
    visited = [start_node]

    while queue != []:
        v, path, f_v, g_v = queue.pop(0)

        if v == goal_node:
            return path, g_v
        for u in graph.get_neighbors(v):
            if visited.__contains__(u) == False:
                visited.append(u)
                g_u = g_v + graph.edges[(v, u)].get_distance()
                h_u = heuristic(graph.nodes[u])
                
                f_u = g_u + h_u
                queue.append((u, path + [u], f_u, g_u))
        queue.sort(key=lambda x: x[2]) 
    return [], float.inf


# ---------- Q2: Stochastic optimization ----------
class ScatterPlainEvolutionaryAlgorithm:
    def __init__(self, scatterplain, points: int, seed: int = 0):
        self.scatterplain = scatterplain # DO NOT CHANGE THIS
        self.points = points             # DO NOT CHANGE THIS
        self.population_size = 50        # Change at leisure
        self.generations = 10            # Change at leisure
        self.elitism = 5                 # Change at leisure
        self.mutation_rate = 0.1         # Change at leisure 
        self.seed = seed                 # Change at leisure
        
        random.seed(seed)
        
    def run_optimization(self):
        """Run evolutionary algorithm to find a good route. Return best route and distance."""
        # Initialize population with random permutations
        population = [random.sample(range(self.points), self.points) for _ in range(self.population_size)]
        history = {} # Dict where key is an individual and value is their fitness. Useful for elitism.
        
        best_solution = None
        best_fitness = float('-inf')
        
        for gen in range(self.generations):
            # TODO: Write evolutionary algorithm. Can make separate functions to call. 
            continue

        return best_solution, best_fitness


# ---------- Q3: Adversarial search w/ MCTS ----------
mcts_1_random_config = {"exploration_constant": math.sqrt(2), "simulate_strategy": "random"}
mcts_1_heuristic_config = {"exploration_constant": math.sqrt(2), "simulate_strategy": "heuristic"}
mcts_1_greedy_config = {"exploration_constant": math.sqrt(2), "simulate_strategy": "greedy"}

mcts_2_greedy_config = {"exploration_constant": math.sqrt(2), "simulate_strategy": "greedy"}
mcts_2_heuristic_config = {"exploration_constant": math.sqrt(2), "simulate_strategy": "heuristic"}

class MCTS:
    """Monte Carlo Tree Search implementation for Connect Four."""
    
    def __init__(self, connectfour: ConnectFour, exploration_constant: float = math.sqrt(2), 
                 simulate_strategy: str = "random"):
        """
        Initialize MCTS.
        
        Args:
            connectfour: ConnectFour game object (initial state)
            exploration_constant: Constant c in UCB formula (typically √2)
            simulate_strategy: One of "random", "heuristic", or "greedy"
        """
        self.connectfour = connectfour
        self.exploration_constant = exploration_constant
        self.simulate_strategy = simulate_strategy
    
    def search(self, num_iterations: int) -> 'MCTSNode':
        """
        Run MCTS for a given number of iterations starting from current connectfour state.
        
        Args:
            num_iterations: Number of MCTS iterations to perform
        
        Returns:
            Best child node to play from root
        """
        # Create root from current game state
        root = MCTSNode(self.connectfour.copy())
        
        for _ in range(num_iterations):
            # MCTS consists of four phases:
            node = self._select(root)           # 1. Selection
            if not node.is_terminal():
                node = self._expand(node)       # 2. Expansion
            reward = self._simulate(node)       # 3. Simulation (rollout)
            self._backpropagate(node, reward)   # 4. Backpropagation
        
        # Return the best child based on visit count
        return self._best_child(root, exploration_weight=0)
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: traverse tree using UCB until reaching a node that 
        is not fully expanded or is terminal.
        
        Args:
            node: Starting node
        
        Returns:
            Selected node for expansion
        """
        # TODO: Implement the function
        return node # This is a placeholder.
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion phase: add a new child node for an untried action.
        
        Args:
            node: Node to expand
        
        Returns:
            Newly created child node
        """
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        
        new_state = node.state.copy()
        new_state.make_move(action)
        new_state.switch_player()
        
        child = MCTSNode(new_state, parent=node, action=action)
        node.children.append(child)
        
        return child
    
    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulation phase: perform a random rollout from the given node.
        
        Choose simulation strategy based on self.simulate_strategy:
          - "random": Full random playout until terminal (Strategy 1)
          - "heuristic": Limited playout with board evaluation (Strategy 2)
          - "greedy": Greedy-random playout with biased move selection (Strategy 3)
        
        Args:
            node: Node to simulate from (has a state: ConnectFour)
        
        Returns:
            Reward: float (typically in [-1, 1])
        """
        if self.simulate_strategy == "random":
            return self._simulate_random(node)
        elif self.simulate_strategy == "heuristic":
            return self._simulate_heuristic(node)
        elif self.simulate_strategy == "greedy":
            return self._simulate_greedy(node)
        else:
            raise ValueError(f"Unknown simulate_strategy: {self.simulate_strategy}")
    
    def _simulate_random(self, node: MCTSNode) -> float:
        """
        STRATEGY 1 - Full Random Playout:
        Play random legal moves until game ends, return outcome.
        """
        state = node.state.copy()
        
        while not state.is_terminal():
            legal_actions = state.get_legal_actions()
            action = random.choice(legal_actions)
            state.make_move(action)
            state.switch_player()
        
        # Determine outcome
        if state.check_winner(Player.PLAYER2):
            return 1.0
        elif state.check_winner(Player.PLAYER1):
            return -1.0
        else:
            return 0.0
    
    def _simulate_heuristic(self, node: MCTSNode) -> float:
        """
        STRATEGY 2 - Limited Playout with Board Heuristic:
        Play up to 10 moves, then evaluate board state if game hasn't ended.
        """
        state = node.state.copy()
        move_count = 0
        max_moves = 10
        
        while not state.is_terminal() and move_count < max_moves:
            legal_actions = state.get_legal_actions()
            action = random.choice(legal_actions)
            state.make_move(action)
            state.switch_player()
            move_count += 1
        
        # If game ended, return outcome
        if state.is_terminal():
            if state.check_winner(Player.PLAYER2):
                return 1.0
            elif state.check_winner(Player.PLAYER1):
                return -1.0
            else:
                return 0.0
        
        # Otherwise, evaluate board heuristically
        # Count potential 3-in-a-row opportunities
        ai_score = self._count_threats(state, Player.PLAYER2)
        human_score = self._count_threats(state, Player.PLAYER1)
        
        return (ai_score - human_score) / 10.0  # Normalize to roughly [-1, 1]
    
    def _simulate_greedy(self, node: MCTSNode) -> float:
        """
        STRATEGY 3 - Greedy Random Playout:
        Bias move selection toward creating/blocking threats.
        """
        state = node.state.copy()
        
        while not state.is_terminal():
            legal_actions = state.get_legal_actions()
            
            # Find threatening moves (creates 3-in-a-row for current player)
            threat_moves = []
            block_moves = []
            
            for col in legal_actions:
                # Try the move
                test_state = state.copy()
                test_state.make_move(col)
                
                # Check if current player wins
                if test_state.check_winner(state.current_player):
                    threat_moves.append(col)
            
            # Find blocking moves (opponent has threat)
            opponent = Player.PLAYER1 if state.current_player == Player.PLAYER2 else Player.PLAYER2
            for col in legal_actions:
                test_state = state.copy()
                test_state.make_move(col)
                test_state.switch_player()
                
                if test_state.check_winner(opponent):
                    block_moves.append(col)
            
            # Choose action with bias: 40% threat, 40% block, 20% random
            rand = random.random()
            if rand < 0.4 and threat_moves:
                action = random.choice(threat_moves)
            elif rand < 0.8 and block_moves:
                action = random.choice(block_moves)
            else:
                action = random.choice(legal_actions)
            
            state.make_move(action)
            state.switch_player()
        
        # Determine outcome
        if state.check_winner(Player.PLAYER2):
            return 1.0
        elif state.check_winner(Player.PLAYER1):
            return -1.0
        else:
            return 0.0
    
    def _count_threats(self, state: ConnectFour, player: Player) -> float:
        """
        Helper: count 3-in-a-row opportunities for a player on the board.
        Returns a score (roughly 0-20).
        """
        score = 0.0
        player_val = player.value
        
        # Check horizontal
        for row in range(state.rows):
            for col in range(state.cols - 2):
                window = [state.board[row, col + i] for i in range(3)]
                if window.count(player_val) == 2 and window.count(0) == 1:
                    score += 1.0
        
        # Check vertical
        for col in range(state.cols):
            for row in range(state.rows - 2):
                window = [state.board[row + i, col] for i in range(3)]
                if window.count(player_val) == 2 and window.count(0) == 1:
                    score += 1.0
        
        # Check diagonal (top-left to bottom-right)
        for row in range(state.rows - 2):
            for col in range(state.cols - 2):
                window = [state.board[row + i, col + i] for i in range(3)]
                if window.count(player_val) == 2 and window.count(0) == 1:
                    score += 1.0
        
        # Check diagonal (top-right to bottom-left)
        for row in range(state.rows - 2):
            for col in range(2, state.cols):
                window = [state.board[row + i, col - i] for i in range(3)]
                if window.count(player_val) == 2 and window.count(0) == 1:
                    score += 1.0
        
        return score
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagation phase: propagate reward back up the tree.
        
        Args:
            node: Node to start backpropagation from
            reward: Reward to propagate
        """
        # TODO implement backpropagation
    
    def _ucb(self, node: MCTSNode, child: MCTSNode) -> float:
        """
        Calculate Upper Confidence Bound (UCB1) for a child node.
        
        Args:
            node: Parent node
            child: Child node to calculate UCB for
        
        Returns:
            UCB value
        """
        # This is your hint
        if child.visit_count == 0:
            return float('inf')
        
        return 0 # This is a placeholder. You need to implement UCB. 
    
    def _best_child(self, node: MCTSNode, exploration_weight: float) -> MCTSNode:
        """
        Select best child based on UCB (or just exploitation if weight=0).
        
        Args:
            node: Parent node
            exploration_weight: Weight for exploration term (0 for pure exploitation)
        
        Returns:
            Best child node
        """
        if not node.children:
            return node
        
        # Temporarily swap exploration constant for this selection
        original_c = self.exploration_constant
        self.exploration_constant = exploration_weight
        
        best_child = max(node.children, key=lambda child: self._ucb(node, child))
        
        self.exploration_constant = original_c
        
        return best_child


# ---------- Q4: Tabular RL ----------
sarsa_config = {"learning_rate": 0.05, "discount_factor": 0.9, "epsilon": 0.1, "episodes": 100000, "verbose": False}

class SARSAAgent:
    """
    SARSA (State-Action-Reward-State-Action) agent for Wumpus World.
    On-policy temporal difference learning.
    """
    def __init__(self, env: WumpusWorld, learning_rate: float = 0.05, 
                    discount_factor: float = 0.9, epsilon: float = 0.1, episodes: int = 100000, verbose: bool = False):
        """
        Initialize SARSA agent.
        
        Args:
            env: WumpusWorld environment
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Exploration rate for epsilon-greedy policy
            episodes: Number of training episodes
            verbose: Whether to print training progress a lot or a little
        """
        self.env = env
        self.num_states = env.grid_size * env.grid_size
        self.num_actions = env.num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.episodes = episodes
        self.verbose = verbose
        
        # Initialize Q-table with zeros
        self.Q = np.zeros((self.num_states, self.num_actions))
    
    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: If False, use greedy policy only
        
        Returns:
            Action to take
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int, next_action: int, done: bool):
        # TODO implement the SARSA update rule
        pass
    
    def train(self):
        """
        Train the SARSA agent.
        
        Args:
            episodes: Number of training episodes
        
        Returns:
            List of episode returns
        """
        episode_returns = []
        
        for episode in range(self.episodes):
            state = self.env.reset()
            episode_return = 0.0
            action = self.select_action(state, training=True)
            done = False
            
            i = 0
            while not done:
                next_state, reward, done, info = self.env.step(action)
                next_action = self.select_action(next_state, training=True)
                
                self.update(state, action, reward, next_state, next_action, done)
                
                episode_return += reward
                
                state = next_state
                action = next_action
                i += 1
            
            episode_returns.append(episode_return)
            
            status = self.env.translate_info_to_sentence(info)
            if self.verbose:
                print(f"Train Episode {episode + 1}/{self.episodes} - Return: {episode_return:.4f} - Status: {status}; Steps: {i}")
            else:
                if (episode + 1) % 10000 == 0:
                    print(f"Train Episode {episode + 1}/{self.episodes} - Return: {episode_return:.4f} - Status: {status}; Steps: {i}")

        return episode_returns
    
    def test(self, episodes: int = 100):
        """
        Test the trained agent without exploration.
        
        Args:
            episodes: Number of test episodes
        
        Returns:
            List of episode rewards
        """
        episode_rewards = []
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0.0
            done = False
            
            i = 0
            while not done:
                action = self.select_action(state, training=False)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                i += 1
            episode_rewards.append(episode_reward)
            
            status = self.env.translate_info_to_sentence(info)
            print(f"Test Episode {episode + 1}/{episodes} - Reward: {episode_reward:.4f} - Status: {status}; Steps: {i}")
        
        return episode_rewards

q_learning_config = {"learning_rate": 0.05, "discount_factor": 0.9, "epsilon": 0.1, "episodes": 100000, "verbose": False}

class QLearningAgent:
    """
    Q-Learning agent for Wumpus World.
    Off-policy temporal difference learning.
    """
    def __init__(self, env: WumpusWorld, learning_rate: float = 0.05, 
                    discount_factor: float = 0.9, epsilon: float = 0.1, episodes: int = 100000, verbose: bool = False):
        """
        Initialize Q-Learning agent.
        
        Args:
            env: WumpusWorld environment
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Exploration rate for epsilon-greedy policy
            episodes: Number of training episodes
            verbose: Whether to print training progress a lot or a little
        """
        self.env = env
        self.num_states = env.grid_size * env.grid_size
        self.num_actions = env.num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.episodes = episodes
        self.verbose = verbose
        
        self.Q = np.zeros((self.num_states, self.num_actions))
    
    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: If False, use greedy policy only
        
        Returns:
            Action to take
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        # TODO implement the Q-Learning update rule
        pass
    
    def train(self):
        """
        Train the Q-Learning agent.
        
        Args:
            episodes: Number of training episodes
        
        Returns:
            List of episode rewards
        """
        episode_rewards = []
        
        for episode in range(self.episodes):
            state = self.env.reset()
            episode_reward = 0.0
            done = False
            
            i = 0
            while not done:
                action = self.select_action(state, training=True)
                next_state, reward, done, info = self.env.step(action)
                
                self.update(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                i += 1
            
            episode_rewards.append(episode_reward)
        
            status = self.env.translate_info_to_sentence(info)
            if self.verbose:
                print(f"Training Episode {episode + 1}/{self.episodes} - Return: {episode_reward:.4f} - Status: {status}; Steps: {i}")
            else:
                if (episode + 1) % 10000 == 0:
                    print(f"Training Episode {episode + 1}/{self.episodes} - Return: {episode_reward:.4f} - Status: {status}; Steps: {i}")
        return episode_rewards
    
    def test(self, episodes: int = 100):
        """
        Test the trained agent without exploration.
        
        Args:
            episodes: Number of test episodes
        
        Returns:
            List of episode rewards
        """
        episode_rewards = []
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0.0
            done = False
            
            i = 0
            while not done:
                action = self.select_action(state, training=False)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                i += 1
            
            episode_rewards.append(episode_reward)
            status = self.env.translate_info_to_sentence(info)
            print(f"Test Episode {episode + 1}/{episodes} - Reward: {episode_reward:.4f} - Status: {status}; Steps: {i}")
            
        return episode_rewards
    
    
# ---------- Q5: Gradient Methods ----------
gd_iters = [100, 100, 100]
gd_stepsizes = [0.5, 0.5, 0.5]
gd_h = [1e-6, 1e-6, 1e-6]

def gradient_descent(f, x0, y0, max_iter=100, step_size=0.5, h=1e-6):
    """
    Gradient descent optimization.
    
    Args:
        f: Function handle (e.g., curve0, curve1, curve2)
        x0, y0: Initial point
        max_iter: Maximum iterations
        step_size: Learning rate
        h: Step size for numerical gradient
    
    Returns:
        (x, y, history) where history is list of (x, y, f_value) tuples
    """
    x, y = float(x0), float(y0)
    history = [(x, y, f(x, y))]

    # TODO implement gradient descent
    
    return history

nr_iters = [100, 100, 100]
nr_h = [1e-6, 1e-6, 1e-6]

def newton_raphson(f, x0, y0, max_iter=100, h=1e-6):
    """
    Newton-Raphson optimization (2D).
    
    Args:
        f: Function handle (e.g., curve0, curve1, curve2)
        x0, y0: Initial point
        max_iter: Maximum iterations
        h: Step size for numerical derivatives
    
    Returns:
        (x, y, history) where history is list of (x, y, f_value) tuples
    """
    x, y = float(x0), float(y0)
    history = [(x, y, f(x, y))]
    
    # TODO implement Newton-Raphson method
    
    return history


# ---------- Q5: 2-layer MLP w/ BACKPROPAGATION ----------
GRADUATE_OR_HONORS = False  # Set to True if you are a graduate student or honors undergrad, False otherwise

NEURAL_NET_SETTINGS = { # Used by NeuralNet, only if GRADUATE_OR_HONORS is False
    'nh': 4,
    'transfer': 'sigmoid',
    'stepsize': 0.1,
    'epochs': 100,
    'decay': False,
    'threshold': 0.5
    } 

RMSPROP_SETTINGS = {  # Used by RMSPropNeuralNet, only if GRADUATE_OR_HONORS is True
    'nh2': 4,
    'nh3': 4,
    'transfer': 'sigmoid',
    'stepsize': 0.1,
    'epochs': 100,
    'decay': False,
    'threshold': 0.5,
    'beta': 0.9
    }


class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__( self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest
    
    
class NeuralNet(Classifier):
    """ Implement a neural network with a single hidden layer. Cross entropy is
    used as the cost function.

    Parameters:
    nh -- number of hidden units
    transfer -- transfer function, in this case, sigmoid
    stepsize -- stepsize for gradient descent
    epochs -- learning epochs
    decay -- boolean dictating if stepsize decreases
    threshold -- determines the decision on whether to class as '0' or '1'
    """
    def __init__(self, parameters={}):
        self.params = {'nh': 4,
                    'transfer': 'sigmoid',
                    'stepsize': 0.1,
                    'epochs': 100,
                    'decay': False,
                    'threshold': 0.5}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] == 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        elif self.params['transfer'] == 'relu':
            self.transfer = utils.relu
            self.dtransfer = utils.drelu
        else:
            raise Exception('NeuralNet -> can only handle sigmoid/relu transfer, must set option transfer to string sigmoid/relu')
        self.w_input = None
        self.w_output = None

    def feedforward(self, inputs):
        """
        Returns the output of the current neural network for the given input
        """
        raise NotImplementedError("Implement feedforward first!")

    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_input, nabla_output)`` representing the gradients
        for the cost function with respect to self.w_input and self.w_output.
        """

        ### YOUR CODE HERE

        ### END YOUR CODE

        assert nabla_input.shape == self.w_input.shape
        assert nabla_output.shape == self.w_output.shape
        return (nabla_input, nabla_output)

    def learn(self, Xtrain, ytrain):

        self.w_input = np.random.rand(Xtrain.shape[1], self.params['nh'])
        self.w_output = np.random.rand(self.params['nh'], 1)

        eta = self.params['stepsize'] # Initial step size
        
        XY = np.c_[Xtrain, ytrain] # Concat so we can shuffle without mismatch

        for i in range(self.params['epochs']):

            # Shuffle for SGD
            np.random.shuffle(XY)

            for j in range(XY.shape[0]):
                # NOTE: Your code here. Run backprop and then update the weights accordingly. 
                continue
            
                ### END YOUR CODE

            # Decrease step size if selected
            if self.params['decay']:
                eta = self.params['stepsize'] * pow(i + 1, -1)

    def predict(self, Xtest):

        ytest = np.zeros(Xtest.shape[0], dtype=int)

        for i in range(Xtest.shape[0]):

            result = self.feedforward(np.matrix(Xtest[i, :]))[-1]

            if (result[0, 0] >= self.params['threshold']):
                ytest[i] = 1.0

        return ytest


class RMSPropNeuralNet(Classifier):
    """ Implement a Neural Network with two hidden layers, trained using RMSProp
    Cross-Entropy is used as the cost function

    Parameters:
    nh2 -- number of hidden units in the penultimate (feeds into output) layer
    nh3 -- number of hidden units in the first hidden layer (gets inputs directory)
    transfer -- transfer function, in this case, sigmoid
    stepsize -- stepsize for gradient descent
    epochs -- learning epochs
    decay -- determines if the stepsize will decrease with the epochs
    threshold -- determines what probability is needed to predict 1 rather than zero
    beta -- The RMSProp parameter
    """
    def __init__(self, parameters={}):
        self.params = {'nh2': 4,
                    'nh3': 4,
                    'transfer': 'sigmoid',
                    'stepsize': 0.1,
                    'epochs': 100,
                    'decay': False,
                    'threshold': 0.5,
                    'beta': 0.9}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] == 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        elif self.params['transfer'] == 'relu':
            self.transfer = utils.relu
            self.dtransfer = utils.drelu
        else:
            raise Exception('NeuralNet -> can only handle sigmoid/relu transfer, must set option transfer to string sigmoid/relu')
        self.w_input = None
        self.w_middle = None
        self.w_output = None

    def feedforward(self, inputs):
        """
        Returns the output of the current neural network for the given input
        """
        raise NotImplementedError("Implement feedforward first!")

    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_input, nabla_middle, nabla_output)`` representing the gradients
        for the cost function with respect to self.w_input, self.w_middle and self.w_output.
        """

        ### YOUR CODE HERE

        ### END YOUR CODE

        assert nabla_input.shape == self.w_input.shape
        assert nabla_middle.shape == self.w_middle.shape
        assert nabla_output.shape == self.w_output.shape
        return (nabla_input, nabla_middle, nabla_output)

    def learn(self, Xtrain, ytrain):

        # A list of classes
        self.w_input = np.random.rand(Xtrain.shape[1], self.params['nh3'])
        self.w_middle = np.random.rand(self.params['nh3'], self.params['nh2'])
        self.w_output = np.random.rand(self.params['nh2'], 1)

        # Initialize array of 1's for the average
        inputMS  = np.zeros((Xtrain.shape[1], self.params['nh3']),)
        middleMS = np.zeros((self.params['nh3'], self.params['nh2']),)
        outputMS = np.zeros((self.params['nh2'], 1),)

        eta = self.params['stepsize'] # Initial step size

        XY = np.c_[Xtrain, ytrain] # Concat so we can shuffle without mismatch

        for i in range(self.params['epochs']):

            # Shuffle for SGD
            np.random.shuffle(XY)

            for j in range(XY.shape[0]):
                # NOTE: Your code here. Run backprop and then update the weights accordingly. 
                # Remember: You are implementing RMSProp, not standard gradient descent. 
                continue		    
            
                ### END YOUR CODE

            if self.params['decay']:
                eta = self.params['stepsize'] * pow(i + 1, -1)

    def predict(self, Xtest):

        ytest = np.zeros(Xtest.shape[0], dtype=int)

        for i in range(Xtest.shape[0]):

            result = self.feedforward(np.matrix(Xtest[i, :]))[-1]

            if (result[0, 0] >= self.params['threshold']):
                ytest[i] = 1.0

        return ytest
