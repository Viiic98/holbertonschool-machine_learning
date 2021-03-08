#!/usr/bin/env python3
""" Train using gradient """
import gym
import numpy as np


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """ implements a full training

        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor
        Return: all values of the score
                (sum of all rewards during one episode loop)
    """
    # Initialize weights
    w = np.random.rand(4, 2)
    eps_rewards = []
    for e in range(nb_episodes):
        state = env.reset()[None, :]
        grads = []
        rewards = []
        score = 0
        while True:
            if show_result and (e % 1000 == 0):
                env.render()
            action, grad = policy_gradient(state, w)
            new_state, reward, done, _ = env.step(action)
            grads.append(grad)
            rewards.append(reward)
            score += reward
            state = new_state[None, :]
            if done:
                break
        # Update weights
        for i in range(len(grads)):
            w += (alpha * grads[i] *
                  sum([r * gamma**r for t, r in enumerate(rewards[i:])]))
        eps_rewards.append(score)
        print("{}: {}".format(e, score), end="\r", flush=False)
    return eps_rewards
