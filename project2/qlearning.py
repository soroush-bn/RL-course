import random
import time
import numpy as np


def q_learning(env, num_episodes=100000,
               max_steps=99, learning_rate=0.3,
               discount_rate=0,
               epsilon=1.0,
               decay_rate=0.005):
    # initialize q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # training

    start_time = time.time()
    for episode in range(num_episodes):

        # reset the environment
        state = env.reset()
        done = False
        pre = (0, 0, False, None)
        if episode%10000==0:
            print("Learning : Episode --> " + str(episode))
        for s in range(max_steps):

            # exploration-exploitation tradeoff
            if random.uniform(0, 1) < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state, :])

            # take action and observe reward
            new_state, reward, done, info = env.step(action,pre)
            pre = (new_state, reward, done, info)

            # Q-learning algorithm
            qtable[state, action] = qtable[state, action] + learning_rate * (
                    reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

            # Update to our new state
            state = new_state

            # if done, finish episode
            if done == True:
                break

        # Decrease epsilon
        epsilon = np.exp(-decay_rate * episode)
    finish_time = time.time()
    print(f"Training completed over {num_episodes} episodes \n time_elapsed:{finish_time-start_time}")
    input("Press Enter to watch trained agent...")
    np.save("qlearning100k0gamma",qtable)
    return qtable
