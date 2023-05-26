import gym
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import seaborn as sns

import glob
from PIL import Image


def make_gif():
    frames = ["1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png", "9.png"]
    f= [Image.open(image) for image in frames]
    frame_one = f[0]
    frame_one.save("hmgammaiter_vi.gif", format="GIF", append_images=f,
                   save_all=True, duration=300, loop=0)


def testPolicy(policy, trials=100):
    """
    Get the average rate of successful episodes over given number of trials
    : param policy: function, a deterministic policy function
    : param trials: int, number of trials
    : return: float, average success rate
    """
    env = gym.make("FrozenLake-v1")
    env.reset()
    success = 0
    cum_rewards = []
    for _ in range(trials):
        cumulative_reward = 0
        done = False
        obs = env.reset()
        while not done:
            action = policy[obs]
            obs, reward, done, _ = env.step(action)
            cumulative_reward += reward
            if obs == 15:
                success += 1
        cum_rewards.append(cumulative_reward)

    avg_success_rate = success / trials
    return avg_success_rate, cum_rewards


def plot_cum_rewards(policy,iters=10, trials=100, title=""):
    crs= []
    avgs = []
    for i in range(iters):
        avg, cr = testPolicy(policy, trials)
        crs.append(sum(cr)/len(cr))
        avgs.append(avg)

    plt.plot(np.arange(iters), crs)
    plt.xlabel("iter")
    plt.ylabel("cummulative reward")
    plt.savefig("cumavg_pi.png", dpi=150)
    plt.show()


def animate(data):
    plt.clf()
    sns.heatmap(data, vmax=.8, square=True, cbar=False)


def init():
    sns.heatmap(np.zeros((10, 10)), vmax=.8, square=True, cbar=False)


def learnModel(env, samples=1e5):
    """
    Get the transition probabilities and reward function
    : param env: object, gym environment
    : param samples: int, random samples
    : return:
        trans_prob: ndarray, transition probabilities p(s'|a, s)
        reward: ndarray, reward function r(s, a, s')
    """
    # get size of obs and action space
    num_state = env.observation_space.n
    num_action = env.action_space.n

    trans_prob = np.zeros((num_state, num_action, num_state))
    reward = np.zeros((num_state, num_action, num_state))
    counter_map = np.zeros((num_state, num_action, num_state))

    counter = 0
    while counter < samples:
        state = env.reset()
        done = False

        while not done:
            random_action = env.action_space.sample()
            new_state, r, done, _ = env.step(random_action)
            trans_prob[state][random_action][new_state] += 1
            reward[state][random_action][new_state] += r

            state = new_state
            counter += 1

    # normalization
    for i in range(trans_prob.shape[0]):
        for j in range(trans_prob.shape[1]):
            norm_coeff = np.sum(trans_prob[i, j, :])
            if norm_coeff:
                trans_prob[i, j, :] /= norm_coeff

    counter_map[counter_map == 0] = 1  # avoid invalid division
    reward /= counter_map

    return trans_prob, reward
