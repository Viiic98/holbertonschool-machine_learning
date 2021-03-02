#!/usr/bin/env python3
""" Sarsa Lambtha """
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """ Epsilon Greedy

        - Q is a numpy.ndarray containing the q-table
        - state is the current state
        - epsilon is the epsilon to use for the calculation
        - You should sample p with numpy.random.uniformn to
          determine if your algorithm should explore or exploit
        - If exploring, you should pick the next action with
          numpy.random.randint from all possible actions
        Returns: the next action index
    """
    p = np.random.uniform(0, 1)
    if p > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(1)
    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """ performs SARSA(λ)

        - v is the openAI environment instance
        - is a numpy.ndarray of shape (s,a) containing the
          Q table
        - lambtha is the eligibility trace factor
        - episodes is the total number of episodes to train over
        - max_steps is the maximum number of steps per episode
        - alpha is the learning rate
        - gamma is the discount rate
        - epsilon is the initial threshold for epsilon greedy
        - min_epsilon is the minimum value that epsilon should decay to
        - epsilon_decay is the decay rate for updating epsilon between episodes
        Returns: Q, the updated Q table
    """
    n = env.observation_space.n
    eps = epsilon
    Et = np.zeros((Q.shape))
    for i in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        for _ in range(max_steps):
            Et *= lambtha * gamma
            Et[state, action] += 1
            new_state, reward, done, _ = env.step(action)
            new_action = epsilon_greedy(Q, new_state, epsilon)
            if env.desc.reshape(n)[new_state] == b'G':
                reward = 1
            if env.desc.reshape(n)[new_state] == b'H':
                reward = -1
            deltat = (reward + gamma * Q[new_state, new_action]
                      - Q[state, action])
            Q[state, action] = (Q[state, action] + alpha
                                * deltat * Et[state, action])
            if done:
                break
            state = new_state
            action = new_action
        epsilon = (min_epsilon + (eps - min_epsilon)
                   * np.exp(-epsilon_decay * i))
    return Q
