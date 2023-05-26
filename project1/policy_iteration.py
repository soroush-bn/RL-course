import time

import gym
import numpy as np
from matplotlib import animation
from wrappers import *
from utils import *
import seaborn as sns
import matplotlib.pyplot as plt

def policy_iteration(P, nS, nA, gamma=0.9, tol=1e-4):
    '''
    parameters:
        P: transition probability matrix
        nS: number of states
        nA: number of actions
        gamma: discount factor
        tol: tolerance for convergence
    returns:
        value_function: value function for each state
        policy: policy for each state
    '''
    # initialize value function and policy
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    # Implement policy iteration here #

    flag = True
    iter = 0
    n_iter = 100
    value_functions_over_iters=[]
    while flag and iter < n_iter:
        print(" ***************************  iteration number: "+str(iter) + " ***************************")
        value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
        new_policy = policy_improvement(P, nS, nA, value_function, policy, gamma)
        diff_policy = new_policy - policy
        value_functions_over_iters.append(value_function)
        if np.linalg.norm(diff_policy) == 0:
            flag = False
        policy = new_policy
        iter += 1
    fig = plt.figure()
    sns.heatmap(value_functions_over_iters, linewidth=0.3, cmap='coolwarm')

    plt.title("value function changing over iterations.\n"+"gamma:"+ str(gamma) +" tol:" + str(tol))
    plt.show()
    plt.savefig("hmvf"+str(gamma)+".png",dpi=150)
    if (iter == 100):
        print("Policy iteraction never converged. Exiting code.")
        exit()

    return value_function, policy


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    value_function = np.zeros(nS)

    n_iter =100
    error = 1
    j = 0
    while error > tol and j < n_iter:
        new_value_function = np.zeros(nS)
        # For each state
        for i in range(nS):
            # Get policy for that state
            a = policy[i]
            # With this policy, get next state
            # probability, nextState, reward, terminal = P[i][a]
            # value_function

            # Find all possible transitions, rewards, etc.
            transitions = P[i][a]
            for transition in transitions:
                prob, nextS, reward, term = transition
                # Calculated update value function
                new_value_function[i] += prob * (reward + gamma * value_function[nextS])
        error = np.max(
            np.abs(new_value_function - value_function))  # Find greatest difference in new and old value function
        print("________________________________________")
        print("===> " + str(j))
        print("new value function is : " + str(new_value_function))
        print("error: {} \n".format(error))
        value_function = new_value_function
        j += 1
    if j >= 100:
        print("Policy evaluation never converged. Exiting code.")
        exit()

    ############################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    new_policy = np.zeros(nS, dtype='int')

    ############################
    # YOUR IMPLEMENTATION HERE #

    # For each state
    for state in range(nS):
        # Get optimal action
        # If ties for optimal exist, choose random
        Qs = np.zeros(nA)
        # For each action
        for a in range(nA):
            # All possible next states from this state-action pair
            transitions = P[state][a]
            for transition in transitions:
                prob, nextS, reward, term = transition
                Qs[a] += prob * (reward + gamma * value_from_policy[nextS])
        # For this state
        # get maximum Q
        max_as = np.where(Qs == Qs.max())
        max_as = max_as[0]
        # Set new policy to this action that maximizes Q
        new_policy[state] = max_as[0]

    ############################
    return new_policy

def gamma_changer(env, start=.1, end=1, step=.1):
    cum_rewards = []
    avgs = []
    gammas = np.arange(start,end,step)

    for gamma in gammas:
        obs = env.reset()
        env.render()
        value_function, policy = policy_iteration(env.P, env.nS, env.nA, gamma=gamma, tol=1e-4)
        avg_success, cumulative_reward = testPolicy(policy)
        avgs.append(avg_success)
        cum_rewards.append(sum(cumulative_reward)/len(cumulative_reward))
    plt.plot(gammas, cum_rewards)
    plt.xlabel("gamma")
    plt.ylabel("cummulative reward")
    plt.savefig("gamma_reward.png", dpi=150)
    plt.show()
    plt.plot(gammas, avg_success)
    plt.xlabel("gamma")
    plt.ylabel("average success")
    plt.savefig("gamma_avg.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # create FrozenLake environment note that we are using a deterministic environment change is_slippery to True to use a stochastic environment
    env = gym.make("FrozenLake-v1", is_slippery=True)
    env = NegHole(env)
    print("this is the state space of your env " + str(env.nS))
    print("this is the action space of your env " + str(env.nA))

    # reset environment to start state
    obs = env.reset()
    env.render()
    # run value iteration algorithm
    value_function, policy = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-4)
    print("policy : " + str(policy))
    print("value function: " + str(value_function))
    for j in range(200):
        action = policy[obs]
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            env.reset()
            break

    success_rate=testPolicy(policy)
    # gamma_changer(env)

    plot_cum_rewards(policy,100,100,"policy_iteration")
    env.close()