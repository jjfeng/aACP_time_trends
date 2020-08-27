import os
import numpy as np
from numpy import ndarray
import scipy.stats
import pickle
import json
from matplotlib import pyplot as plt


def pickle_to_file(obj, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f, protocol=-1)


def pickle_from_file(file_name):
    with open(file_name, "rb") as f:
        out = pickle.load(f)
    return out


def process_params(param_str, dtype, split_str=","):
    if param_str:
        return [dtype(r) for r in param_str.split(split_str)]
    else:
        return []

def plot_loss(loss_history, fig_name, alpha, ymin, ymax):
    T = loss_history.size + 1
    plt.clf()
    plt.plot(np.arange(T - 1), loss_history, "g-")
    plt.plot(np.arange(T - 1), np.cumsum(loss_history) / np.arange(1, T), "b--")
    plt.ylabel("loss")
    plt.xlabel("Time")
    plt.hlines(y=alpha, xmin=0, xmax=T)
    plt.ylim(ymin, ymax)
    plt.savefig(fig_name)

def plot_human_use(human_history, fig_name):
    T = human_history.size + 1
    plt.clf()
    plt.figure(figsize=(6, 6))
    plt.plot(np.arange(T - 1), human_history, "g-")
    plt.plot(np.arange(T - 1), np.cumsum(human_history) / np.arange(1, T), "b--")
    plt.ylim(0, 1)
    plt.ylabel("Human prob")
    plt.xlabel("Time")
    plt.savefig(fig_name)
