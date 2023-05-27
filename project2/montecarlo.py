import sys
from collections import defaultdict
import pandas as pd
import numpy as np
def generate_episode_from_limit_stochastic(env):

    episode = []

    state = env.reset()
    pre = (0,0,False,None)
    while True:
        # probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]

        action = np.random.choice(np.arange(10))

        next_state, reward, done, info = env.step(action,pre)
        pre = (next_state, reward, done, info)

        episode.append((state, action, reward))

        state = next_state

        if done:
            break
    env.close()
    return episode


def mc_prediction_q(env, num_episodes, generate_episode, gamma=0.9):
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))

    N = defaultdict(lambda: np.zeros(env.action_space.n))

    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for i_episode in range(1, num_episodes + 1):

        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")

            sys.stdout.flush()

        episode = generate_episode(env)

        states, actions, rewards = zip(*episode)
        # print(np.sum(rewards))
        discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])

        for i, state in enumerate(states):
            returns_sum[state][actions[i]] += sum(rewards[i:] * discounts[:-(1 + i)])

            N[state][actions[i]] += 1.0

            Q[state][actions[i]] = returns_sum[state][actions[i]] / N[state][actions[i]]

    return Q


def generate_episode_from_Q(env, Q, epsilon, nA):
    episode = []

    state = env.reset()

    while True:

        action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
            if state in Q else env.action_space.sample()

        next_state, reward, done, info = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if done:
            break

    return episode


def get_probs(Q_s, epsilon, nA):
    policy_s = np.ones(nA) * epsilon / nA

    best_a = np.argmax(Q_s)

    policy_s[best_a] = 1 - epsilon + (epsilon / nA)

    return policy_s


def update_Q(env, episode, Q, alpha, gamma):
    states, actions, rewards = zip(*episode)

    discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])

    for i, state in enumerate(states):
        old_Q = Q[state][actions[i]]

        Q[state][actions[i]] = old_Q + alpha * (sum(rewards[i:] * discounts[:-(1 + i)]) - old_Q)

    return Q