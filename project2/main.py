from gym.wrappers import RecordEpisodeStatistics
# from stable_baselines3.common.monitor import Monitor
import gym
from qlearning import q_learning
# from project2.wrapper import AdditionalActions, DiagonalEnv
from wrapper import AdditionalActions, DiagonalEnv
# from gym.wrappers import Monitor
import numpy as np
from stable_baselines3.common.monitor import Monitor
from utils import plot_steps_reward
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from project2.utils import plot_results

"""
    0: Move south (down)
    1: Move north (up)
    2: Move east (right)
    3: Move west (left)
    4: Pickup passenger
    5: Drop off passenger
    6: up right
    7 : yp left
    8: down right
    9: down left

Passenger locations:
    0: R(ed)
    1: G(reen)
    2: Y(ellow)
    3: B(lue)
    4: in taxi
Destinations:
    0: R(ed)
    1: G(reen)
    2: Y(ellow)
    3: B(lue)
"""

if __name__ == '__main__':
    log_dir = "/tmp/qlearning"
    env = gym.make('Taxi-v3')
    env = RecordEpisodeStatistics(env)
    env = Monitor(env,"qlearning100k0gamma")
    env = AdditionalActions(env)
    env = DiagonalEnv(env)

    print(env.action_space)
    print(env.observation_space)

    obs = env.reset()
    env.render()
    print(obs)
    print("starting \n ")
    print(env.s)
    l = list(env.decode(env.s))
    print(l)
    # for i in range(20):
    #     action = 7  # env.action_space.sample()
    #     print("action is : " + str(action))
    #     env.step(action)
    #     env.render()

    # observation, reward, done, info = env.step(7)
    # print(observation)
    # observation, reward, done, info = env.step(9)
    # print(observation)
    # print(info)

    state = env.reset()
    done = False
    rewards = 0
    qtable = q_learning(env)
    pre = (0,0,False,None)

    state = env.reset()
    print("before playing: "  + str(list(env.decode(env.s))))
    arr_rewards= []
    for s in range(100):

        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))

        action = np.argmax(qtable[state, :])
        print("selected action is :" + str(action))
        new_state, reward, done, info = env.step(action,pre)
        pre = (new_state, reward, done, info)
        rewards += reward
        arr_rewards.append(reward)
        env.render()
        print(f"score: {rewards}")
        state = new_state
        if list(env.decode(env.s))[2]==list(env.decode(env.s))[3]:##passenger = destination
            done =True

        if done == True:
            break
    plot_steps_reward(arr_rewards,"qlearning100k0gamma")
    env.close()
