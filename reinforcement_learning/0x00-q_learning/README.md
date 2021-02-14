# Q-learning

![Q](https://perfectial.com/wp-content/uploads/2019/10/q_learning-01.jpg)

## Tasks

### [Load the Environment](./0-load_env.py)
- function def load_frozen_lake(desc=None, map_name=None, is_slippery=False): that loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gym

### [Initialize Q-table](./1-q_init.py)
- function def q_init(env): that initializes the Q-table

### [Epsilon Greedy](./2-epsilon_greedy.py)
- function def epsilon_greedy(Q, state, epsilon): that uses epsilon-greedy to determine the next action

### [Q-learning](./3-q_learning.py)
- function def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05): that performs Q-learning:

### [Play](./4-play.py)
- function def play(env, Q, max_steps=100): that has the trained agent play an episode
