from time import sleep

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter
import os
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

matplotlib.use('TkAgg')  # !IMPORTANT
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()


def plot_steps_reward(rewards, name="plot"):
    png_dir = "./pngs/"
    os.makedirs(png_dir, exist_ok=True)
    plt.plot(np.arange(len(rewards)), rewards)
    plt.xlabel("time step")
    plt.ylabel("reward")
    plt.savefig(f'{png_dir+name}.png', dpi=150)
    plt.show()

def plot_csv_logs(path="./logs",name="monitor.csv"):
    plt.rcParams["figure.autolayout"] = True


    df = pd.read_csv(os.path.join(path,name),skiprows=1)
    df =df.iloc[1:]
    df = df.set_axis(['reward', 'steps', 'time'], axis=1, inplace=False)
    df= df.drop(["time"],axis=1)
    print(df.head(5))
    df.plot(use_index=True, y='reward',kind="line")
    plt.show()

if __name__ == '__main__':
    # log_dir = "E:\projects\RL-projects\project2"
    log_dir = "./logs/"
    # plot_results(log_dir)
    # results_plotter.plot_results([log_dir], 100, results_plotter.X_TIMESTEPS, "Breakout")
    # results_plotter.plot_results(
    #     [log_dir], 1e5, results_plotter.X_TIMESTEPS, "TD3 LunarLander"
    # )
    plot_csv_logs("./","qlearning1mil.monitor.csv")
    sleep(4)
