# aiLab

##Q1 Graph Search Algorithms:

In this question you will implement several graph search algorithms. Look at the provided graph.py. You may even run graph.py itself to see a graph visualization of what you’re working with. Using the function handles in student.py, you are to implement:

a) Breadth-First Search ‘bfs’ function.
b) Depth-First Search ‘dfs’ function.
c) A* search ‘astar’ function. 

Pay close attention to the argument structure of the bfs, dfs and astar functions, as well as what the eval_harness calls expects to see. 6/10 marks for this question based on code. Further, in your PDF, answer the following questions:
a) Compare BFS and DFS. Which is worse in this scenario and by how much?
b) Compare the three heuristics for A* search. When implementing the code, which did you think would be the best? Which ended up being the best? What is most surprising about using these A* search heuristics?

##Q2 Search in Complex Environments:

You are to implement an Evolutionary Genetic Algorithm. See scatter.py which defines the environment you are working with – the same 2D plan as graph.py except the nodes are just coordinate points and there are no edges – for now. Individuals in your Evolutionary Algorithm are series of points, e.g., [3, 4, 6, 2, 10] for instance, where each integer corresponds to one labeled point. Another individual could be [9, 4, 6, 2, 10], in fact this is a one-edit mutation of the first point created by changing the first index from 3->9.

If you run scatter.py, you can see that it formulates a loop path traveling through 5 points, and the console prints out an associated “Total Distance”. The fitness of each individual is the total path distance multiplied by -1, e.g., we are aiming to find sequences of ordered points where the path between the points is minimized. Note that this path could be a closed loop where the route includes the distance between the first or last node – that is determined by the ScatterPlain class instance, not your evolutionary algorithm. Also note that solutions that contain repeated/duplicate entries, e.g., [9, 4, 6, 2, 2] are invalid and should be discarded with a fitness of −∞.

To facilitate this, you are to fill out the ScatterPlainEvolutionaryAlgorithm class in student.py. You are not to edit any of the fields or methods that do exist, but you can add new functions, e.g., for selection, crossover, and mutation, at your leisure. 5/15 (20) marks will be based on your code implementation.

Questions:
a) Explain your selection strategy.
b) Explain your crossover strategy.
c) Explain your mutation strategy.
d) Did you use elitism? Why/why not?
e) What is the best individual solution you found? In terms of the route and distance?

##Q3 Adversarial Search:

In this question you will implement Monte Carlo Tree Search (MCTS) to play Connect Four.
You will evaluate your agent using several rollout strategies, comparing to a random move
agent and even having 2 instances of your own code face oT! Most of the necessary code is already provided for you in terms of the Connect Four game - provided in connectfour.py, which you can study but cannot change), as well as the appropriate section in student.py, specifically the mcts_***config dictionaries and MCTS class. As you can see, each config dictionary is referenced once in eval_harness.py, to play a game by itself or in the case of the final two config dictionary, to play against each other. Each configuration dictionary specifies the exploration constant 𝐶 in the UCB1 formula aswell as the simulation/rollout strategy, random, heuristic or greedy – these are specified in the MCTS class. 

Your job is to implement the “_select”, “_backpropagate” and “_ucb” methods of the MCTS class, your implementation of is worth 6/15 marks.

Questions:
a) Rank the three functions in terms of implementation diTiculty.
b) Experiment with the exploration constant 𝐶. Do you believe this value to be optimal for any of the provided heuristics? Why/why not?
c) Based on your experiments with the exploration constant, compare the greedy and heuristic simulation strategies. Which do you believe is better against the random opponent? Which is better when they are pitted against each other? Why?

##Q4 Reinforcement Learning:

You will implement the On-Policy and OT-Policy Temporal DiTerence (TD) learning algorithms for tabular Reinforcement Learning, SARSA and Q-Learning, respectively, in a modified version of the Wumpus World setting from the AIMA textbook. Primarily, we have made 2 modifications to the environment:
- The agent has no sense of “direction” like the AIMA book, so firing the arrow will kill the Wumpus if it is in the same row or column as the agent, no questions asked.
- We have placed a limit on the maximum number of steps the agent can take, after which the episode ends and a -1000 reward penalty is applied.

Much like the MCTS question, the environment is provided for you in wumpus_world.py which you are not allowed to change but can run to see what a random agent would do, which is to primarily wander about aimlessly and then either fall into a pit or get eaten.

Also, like the MCTS question, we have provided most of the code for the SARSA and Q- Learning agents in student.py. Your job is to fill in the “update” functions for both. Also, note the existence of two configuration dictionaries, ‘sarsa_config’ and ‘q_learning_config’ above the definition for each class. You are welcome to adjust the learning rate (𝛼), discount factor (𝛾), exploration probability (𝜀) and number of learning episodes at your leisure. The final dictionary field, “verbose” is a Boolean that controls how often the return on a training episode is printed – every episode (True) or every 10k (False). 8/15 marks will be assigned based on the correct implementation of these update rules.

Questions:
a) Do either of the algorithms elect to kill the Wumpus in this environment? To even fire the arrow?
b) If we removed the cap on the maximum number of steps an episode to take, which algorithm is more likely to spend a greater number of actions while learning? Why?
c) Play around with the learning parameters and try to make it so the test reward measured after training is a positive number. What parameters did you have to usefor both SARSA and Q-Learning?
d) Which algorithm has the potential for higher average training reward, given the initial 100k number of episodes being held constant?

##Q5 Gradient-based Optimization:

This question is about finding the minima of several 3D functions, ranging from simple tocomplex. The functions are defined in curves.py which you can run to visualize but cannot change. Please note that not all methods may not be able to find each minima.

Specifically, you are to implement the gradient descent method and the Newton-Ralphson hessian approximation method in code in student.py. You are free to modify gd_iters, gd_stepsizes, gd_h as well as nr_iters and nr_h at your leisure.

Questions:
a) Provide a table (or two, one per method) listing whether each method could find the minima per table as a yes/no Boolean. Further, the table should also include the best (minimal value) found by each method on each curve as well as the final value found given the parameters you set when submitting the code.
b) For curves 1 and 2, provide some discussion on the diTerence between when the best and final values found for:
a. Gradient descent
b. Newton-Ralphson

##Q6 Neural Networks and Backpropagation:

Note: First, look at the “GRADUATE_OR_HONORS” Boolean in student.py.
- If you are an undergraduate student, set this Boolean to be False; you will be completing the “NeuralNet” class and adjusting the “NEURAL_NET_SETTINGS” dictionary.
- If you are a graduate or honors student, set this Boolean to be True; you will be completing the “RMSPropNeuralNet” class and adjusting the “RMSPROP_SETTINGS” dictionary. Documentation on the RMSProp algorithm can be found in GeoTrey Hinton’s UToronto slides, the PyTorch optimizer documentation and others.

In this question you will complete the implementation of a Multi-Layered Perceptron (MLP) classifier that trains using the backpropagation algorithm. Graduate/Honors students have to implement an even bigger MLP using a more complex optimization scheme – RMSProp. The definition of the MLP is in student.py and that is what you shall change. This MLP trains on the data provided in susysubset.csv using a train/test split defined in dataloader.py, and with help from some functions in utilities.py (none of which you can change).

Specifically, the function in eval_harness.py will load the model, train it, evaluate it, and then report the test accuracy. The “__init__”, and “predict” functions are already provided for you and you do not need to change them; but you do need to complete the “learn” function, and properly implement the “feedforward”, “backprop” functions to make the network function and be capable of learning, respectively.

Questions:
a) When you got the neural network functioning, what was the accuracy with the default parameters set?
b) Play around with the parameters in the dictionary, particularly ‘nh’, the learning rate, decay and epochs but not ‘transfer’. Which seem to most strongly influence the performance of the network?
c) Look at utilities.py at the relu and drelu functions. Now change the ‘transfer’ setting from sigmoid to relu. What happens to the performance and speed? Is this counter-intuitive? Why??
