#!/usr/bin/env python3
""" Play """
import numpy as np


def play(env, Q, max_steps=100):
    """ Play

        env is the FrozenLakeEnv instance
        Q is a numpy.ndarray containing the Q-table
        max_steps is the maximum number of steps in the episode
        Each state of the board should be displayed via the console
        You should always exploit the Q-table
        Returns: the total rewards for the episode
    """
    total_rewards = 0
    state = env.reset()
    env.render()
    for step in range(max_steps):

        # Take the action (index) that have the maximum expected
        # future reward given that state
        action = np.argmax(Q[state, :])

        new_state, reward, done, info = env.step(action)

        env.render()

        state = new_state
        total_rewards += reward
        if done:
            break
    env.close()
    return total_rewards
