#!/usr/bin/env python3
""" Q-learning """
import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """ Q-learning

        - env is the FrozenLakeEnv instance
        - Q is a numpy.ndarray containing the Q-table
        - episodes is the total number of episodes to train over
        - max_steps is the maximum number of steps per episode
        - alpha is the learning rate
        - gamma is the discount rate
        - epsilon is the initial threshold for epsilon greedy
        - min_epsilon is the minimum value that epsilon should
          decay to
        - epsilon_decay is the decay rate for updating epsilon
          between episodes
        - When the agent falls in a hole, the reward should be
          updated to be -1
        Returns: Q, total_rewards
            - Q is the updated Q-table
            - total_rewards is a list containing the rewards
              per episode
    """
    # List of rewards
    rewards = []

    # 2 For life or until learning is stopped
    for episode in range(episodes):
        # Reset the environment
        state = env.reset()
        total_rewards = 0

        for step in range(max_steps):
            # 3. Choose an action a in the current world state (s)
            # First we randomize a number
            exp_exp_tradeoff = np.random.uniform(0, 1)

            # If this number > greater than epsilon -->
            # exploitation (taking the biggest Q value for this state)
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(Q[state, :])

            # Else doing a random choice --> exploration
            else:
                action = env.action_space.sample()

            # Take the action (a) and observe the outcome state(s')
            # and reward (r)
            new_state, reward, done, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max
            # Q(s',a') - Q(s,a)]
            # Q[new_state,:] : all the actions we can take
            # from new state
            Q[state, action] = Q[state, action] + alpha *\
                (reward + gamma *
                 np.max(Q[new_state, :]) -
                 Q[state, action])

            total_rewards += reward

            # Our new state is state
            state = new_state

            # If done (if we're dead) : finish episode
            if done:
                break

        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (epsilon - min_epsilon) *\
            np.exp(-epsilon_decay * episode)
        rewards.append(total_rewards)
    env.close()
    return Q, rewards
