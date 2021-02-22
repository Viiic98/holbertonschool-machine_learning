# Deep Q-learning

## Breakout
![breakout](https://miro.medium.com/max/320/1*jXpSVhjWRxgzDDKKVAQR8A.gif)

### [Train](./train.py)
- Write a python script train.py that utilizes keras, keras-rl, and gym to train an agent that can play Atari’s Breakout:
    - Your script should utilize keras-rl‘s DQNAgent, SequentialMemory, and EpsGreedyQPolicy
    - Your script should save the final policy network as policy.h5

### [Play](./play.py)
- Write a python script play.py that can display a game played by the agent trained by train.py:
    - Your script should load the policy network saved in policy.h5
    - Your agent should use the GreedyQPolicy