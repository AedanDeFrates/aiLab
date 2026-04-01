import numpy as np
import random
from time import time


def eval_q1_search():
    from graph import Graph
    from student import bfs, dfs, astar
    
    q1_graph = Graph(n=50, k=4, dim=2, gridsize=100, seed=42)
    start_node_idx = 15
    end_node_idx = 0
    
    print(f"====== Question 1: Graph Search Algorithms ======")
    
    try:
        print("====== BFS ======")
        start = time() 
        bfs_plan = bfs(q1_graph, start_node_idx, end_node_idx)
        assert type(bfs_plan) == list, "BFS should return a list of nodes representing the path."
        assert bfs_plan[0] == start_node_idx, "BFS path should start with the start node."
        assert bfs_plan[-1] == end_node_idx, "BFS path should end with the end node."
        assert len(bfs_plan) >= 2 and len(bfs_plan) <= 15, "BFS path length should be between 2 and 15."
        print(f"BFS: Success w/ plan {bfs_plan} and length {len(bfs_plan)}")
    except Exception as e:
        print(f"BFS: Failed with error {e}")
    print(f"Time taken {time() - start:.4f} seconds")
    
    try:
        print("====== DFS ======")
        start = time()
        dfs_plan = dfs(q1_graph, start_node_idx, end_node_idx)
        assert type(dfs_plan) == list, "DFS should return a list of nodes representing the path."
        assert dfs_plan[0] == start_node_idx, "DFS path should start with the start node."
        assert dfs_plan[-1] == end_node_idx, "DFS path should end with the end node."
        assert len(dfs_plan) >= 2 and len(dfs_plan) <= 20, "DFS path length should be between 2 and 20."
        print(f"DFS: Success w/ plan {dfs_plan} and length {len(dfs_plan)}")
    except Exception as e:
        print(f"DFS: Failed with error {e}")
    print(f"Time taken {time() - start:.4f} seconds")
    
    try:
        print("====== A* w/ Euclidean Heuristic ======")
        from heuristics import EuclideanHeuristic
        start = time()
        astar_plan, astar_dist = astar(q1_graph, start_node_idx, end_node_idx, heuristic=EuclideanHeuristic(q1_graph.nodes[end_node_idx]))
        assert type(astar_plan) == list, "A* should return a list of nodes representing the path."
        assert astar_plan[0] == start_node_idx, "A* path should start with the start node."
        assert astar_plan[-1] == end_node_idx, "A* path should end with the end node."
        assert astar_dist > 0 and astar_dist < 250, "A* distance should be a positive number."
        print(f"A* w/ Euclidean: Success w/ plan {astar_plan} and distance {astar_dist}")
    except Exception as e:
        print(f"A* w/ Euclidean: Failed with error {e}")
    print(f"Time taken {time() - start:.4f} seconds")
        
    try:
        print("====== A* w/ Manhattan Heuristic ======")
        from heuristics import ManhattanHeuristic
        start = time()
        astar_plan, astar_dist = astar(q1_graph, start_node_idx, end_node_idx, heuristic=ManhattanHeuristic(q1_graph.nodes[end_node_idx]))
        assert type(astar_plan) == list, "A* should return a list of nodes representing the path."
        assert astar_plan[0] == start_node_idx, "A* path should start with the start node."
        assert astar_plan[-1] == end_node_idx, "A* path should end with the end node."
        assert astar_dist > 0 and astar_dist < 250, "A* distance should be a positive number."
        print(f"A* w/ Manhattan: Success w/ plan {astar_plan} and distance {astar_dist}")
    except Exception as e:
        print(f"A* w/ Manhattan: Failed with error {e}")
    print(f"Time taken {time() - start:.4f} seconds")
        
    try:
        print("====== A* w/ Chebyshev Heuristic ======")
        start = time()
        from heuristics import ChebyshevHeuristic
        astar_plan, astar_dist = astar(q1_graph, start_node_idx, end_node_idx, heuristic=ChebyshevHeuristic(q1_graph.nodes[end_node_idx]))
        assert astar_plan[0] == start_node_idx, "A* path should start with the start node."
        assert astar_plan[-1] == end_node_idx, "A* path should end with the end node."
        assert astar_dist > 0 and astar_dist < 250, "A* distance should be a positive number."
        print(f"A* w/ Chebyshev: Success w/ plan {astar_plan} and distance {astar_dist}")
    except Exception as e:
        print(f"A* w/ Chebyshev: Failed with error {e}")
    print(f"Time taken {time() - start:.4f} seconds")


def eval_q2_evol_alg():
    print("====== Question 2: Evolutionary Algorithm ======")
    from scatter import ScatterPlain
    from student import ScatterPlainEvolutionaryAlgorithm
    
    sp_env = ScatterPlain(n=50, dim=2, gridsize=100, loop=True, seed=42)
    try:
        start = time()
        points = 5
        ea = ScatterPlainEvolutionaryAlgorithm(sp_env, points=points)
        best_solution, best_fitness = ea.run_optimization()
        assert type(best_solution) == list, "Best solution should be a list of point indices."
        assert len(set(best_solution)) == len(best_solution), "Best solution should not have duplicate point indices."
        assert len(best_solution) == points, f"Best solution should include all {points} points."
        assert best_fitness < 0, "Best fitness should be negative (since it's negative distance)."
        print(f"Evolutionary Algorithm: Success w/ best solution {best_solution} and fitness {best_fitness}")
    except Exception as e:
        print(f"Evolutionary Algorithm: Failed with error {e}")
    print(f"Time taken {time() - start:.4f} seconds")
    

def eval_q3_mcts():
    print("====== Question 3: Monte Carlo Tree Search ======")
    from student import MCTS
    from connectfour import ConnectFour, play_connectfour, Player
    
    try:
        print("====== Single-Agent MCTS w/ Random Simulation ======")
        from student import mcts_1_random_config
        start = time()
        connectfour_one_agent = ConnectFour(title="Single Random")
        mcts1 = MCTS(connectfour_one_agent, **mcts_1_random_config)
        winner = play_connectfour(connectfour_one_agent, num_iterations=1000, mcts1=mcts1, verbose=True)
        assert winner in [Player.PLAYER1, Player.PLAYER2, None], "Winner should be PLAYER1, PLAYER2, or None (draw)."
        print(f"MCTS: Success w/ winner {winner}")
    except Exception as e:
        print(f"MCTS: Failed with error {e}")
    print(f"Time taken {time() - start:.4f} seconds")
        
    try:
        print("====== Single-Agent MCTS w/ Greedy Simulation ======")
        from student import mcts_1_greedy_config
        start = time()
        connectfour_one_agent = ConnectFour(title="Single Greedy")
        mcts1 = MCTS(connectfour_one_agent, **mcts_1_greedy_config)
        winner = play_connectfour(connectfour_one_agent, num_iterations=1000, mcts1=mcts1, verbose=True)
        assert winner in [Player.PLAYER1, Player.PLAYER2, None], "Winner should be PLAYER1, PLAYER2, or None (draw)."
        print(f"MCTS: Success w/ winner {winner}")
    except Exception as e:
        print(f"MCTS: Failed with error {e}")
    print(f"Time taken {time() - start:.4f} seconds")
        
    try:
        print("====== Single-Agent MCTS w/ Heuristic Simulation ======")
        from student import mcts_1_heuristic_config
        start = time()
        connectfour_one_agent = ConnectFour(title="Single Heuristic")
        mcts1 = MCTS(connectfour_one_agent, **mcts_1_heuristic_config)
        winner = play_connectfour(connectfour_one_agent, num_iterations=1000, mcts1=mcts1, verbose=True)
        assert winner in [Player.PLAYER1, Player.PLAYER2, None], "Winner should be PLAYER1, PLAYER2, or None (draw)."
        print(f"MCTS: Success w/ winner {winner}")
    except Exception as e:
        print(f"MCTS: Failed with error {e}")
    print(f"Time taken {time() - start:.4f} seconds")
        
    try:
        print("====== Two-Agent MCTS: Greedy vs Heuristic ======")
        from student import mcts_2_greedy_config
        from student import mcts_2_heuristic_config
        start = time()
        connectfour_two_agents = ConnectFour(title="Two-Agent Greedy vs Heuristic")
        mcts1 = MCTS(connectfour_two_agents, **mcts_2_greedy_config)
        mcts2 = MCTS(connectfour_two_agents, **mcts_2_heuristic_config)
        winner = play_connectfour(connectfour_two_agents, num_iterations=1000, mcts1=mcts1, mcts2=mcts2, verbose=True)
        assert winner in [Player.PLAYER1, Player.PLAYER2, None], "Winner should be PLAYER1, PLAYER2, or None (draw)."
        print(f"MCTS: Success w/ winner {winner}")
    except Exception as e:
        print(f"MCTS: Failed with error {e}")
    print(f"Time taken {time() - start:.4f} seconds")
    
    
def eval_q4_rl():
    from wumpus_world import WumpusWorld
    from student import QLearningAgent, SARSAAgent
    
    print("====== Question 4: Tabular Reinforcement Learning ======")

    try:
        print("====== SARSA Agent ======")
        from student import sarsa_config
        start = time()
        env = WumpusWorld(grid_size=4, max_actions=500, seed=0)
        agent = SARSAAgent(env, **sarsa_config)
        train_rewards = agent.train()
        assert len(train_rewards) == sarsa_config['episodes'], "Should have reward for each episode."
        print(f"SARSA: Success w/ final train average reward {np.mean(train_rewards[-100:]):.4f}")
        test_rewards = agent.test(episodes=1)
        # We only need 1 test episode since we are now using the deterministic target policy.
        assert len(test_rewards) == 1, "Should have reward for each test episode."
        print(f"SARSA: Success w/ average test reward {np.mean(test_rewards):.4f}")
    except Exception as e:
        print(f"SARSA: Failed with error {e}")
    print(f"Time taken {time() - start:.4f} seconds")
    
    try:
        print("====== Q-Learning Agent ======")
        from student import q_learning_config
        start = time()
        env = WumpusWorld(grid_size=4, max_actions=500, seed=0)
        agent = QLearningAgent(env, **q_learning_config)
        train_rewards = agent.train()
        assert len(train_rewards) == q_learning_config['episodes'], "Should have reward for each episode."
        print(f"Q-Learning: Success w/ final average training reward {np.mean(train_rewards[-100:]):.4f}")
        test_rewards = agent.test(episodes=1)
        assert len(test_rewards) == 1, "Should have reward for each test episode."
        print(f"Q-Learning: Success w/ average test reward {np.mean(test_rewards):.4f}")
    except Exception as e:
        print(f"Q-Learning: Failed with error {e}")
    print(f"Time taken {time() - start:.4f} seconds")
    

def eval_q5_grad_descent():
    from curves import curve0, curve1, curve2
    from student import gradient_descent, newton_raphson
    print("====== Question 5: Optimization ======")
    
    try:
        print("====== Gradient Descent ======")
        from student import gd_iters, gd_stepsizes, gd_h
        start = time()
        for i, curve in enumerate([curve0, curve1, curve2]):
            print("===== GD Curve:", curve.__name__, "=====")
            history = gradient_descent(curve, x0=15, y0=15, max_iter=gd_iters[i], step_size=gd_stepsizes[i], h=gd_h[i])
            print(f"Gradient Descent: Success on {curve.__name__} w/ final coordinates ({history[-1][0]:.4f}, {history[-1][1]:.4f}, {history[-1][2]:.4f})")
            min_point = min(history, key=lambda p: p[2])
            print(f"Gradient Descent: Minimum value found on {curve.__name__} at ({min_point[0]:.4f}, {min_point[1]:.4f}) with value {min_point[2]:.4f}")
    except Exception as e:
        print(f"Gradient Descent: Failed with error {e}")
    print(f"Time taken {time() - start:.4f} seconds")
        
    try:
        print("====== Newton-Raphson ======")
        from student import nr_iters, nr_h
        start = time()
        for i, curve in enumerate([curve0, curve1, curve2]):
            print("===== NR Curve:", curve.__name__, "=====")
            history = newton_raphson(curve, x0=15, y0=15, max_iter=nr_iters[i], h=nr_h[i])
            print(f"Newton-Raphson: Success on {curve.__name__} w/ final coordinates ({history[-1][0]:.4f}, {history[-1][1]:.4f}, {history[-1][2]:.4f})")
            min_point = min(history, key=lambda p: p[2])
            print(f"Newton-Raphson: Minimum value found on {curve.__name__} at ({min_point[0]:.4f}, {min_point[1]:.4f}) with value {min_point[2]:.4f}")
    except Exception as e:
        print(f"Newton-Raphson: Failed with error {e}")
    print(f"Time taken {time() - start:.4f} seconds")


def eval_q6_nn():
    from student import GRADUATE_OR_HONORS
    if GRADUATE_OR_HONORS:
        from student import RMSPropNeuralNet as NN
        from student import RMSPROP_SETTINGS as neural_net_dict
        statement = "Graduate/Honors - RMSProp Neural Net"
    else:
        from student import NeuralNet as NN
        from student import NEURAL_NET_SETTINGS as neural_net_dict
        statement = "Undergraduate - Basic Neural Net"
    print(f"====== Question 6: Neural Networks ({statement}) ======")
    from dataloader import load_susy

    random.seed(42)
    np.random.seed(42)
    
    trainset, testset = load_susy(7000, 3000)
    
    start = time()
    try:
        nn = NN(parameters=neural_net_dict)
        nn.learn(trainset[0], trainset[1])
        predictions = nn.predict(testset[0])
        
        def getaccuracy(ytest, predictions):
            correct = 0
            for i in range(len(ytest)):
                if ytest[i] == predictions[i]:
                    correct += 1
            return (correct/float(len(ytest))) * 100.0
        
        error = getaccuracy(testset[1], predictions)
        
        print("Neural Network: Success w/ test acc of {:.2f}%".format(error))
    except Exception as e:     
        print(f"Neural Network: Failed with error {e}")
    print(f"Time taken {time() - start:.4f} seconds")


if __name__ == "__main__":

    eval_q1_search()
    eval_q2_evol_alg()
    eval_q3_mcts()
    eval_q4_rl()
    eval_q5_grad_descent()
    eval_q6_nn()