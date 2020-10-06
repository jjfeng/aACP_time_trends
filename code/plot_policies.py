import sys
import os
import shutil
import argparse
import logging
import numpy as np
from matplotlib import pyplot as plt

from approval_history import ApprovalHistory
from common import pickle_from_file, process_params


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--seed",
        type=int,
        help="Random number generator seed for replicability",
        default=12,
    )
    parser.add_argument("--y-max", type=float, default=10)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument(
        "--history-files", type=str, help="comma separated pickle files"
    )
    parser.add_argument(
        "--loss-plot", type=str, default="_output/approver_history_loss.png"
    )
    parser.add_argument(
        "--human-plot", type=str, default="_output/approver_history_human.png"
    )
    parser.set_defaults()
    args = parser.parse_args()

    args.history_files = process_params(args.history_files, str)

    return args


def plot_losses(approval_histories, fig_name, alpha, ymin, ymax):
    plt.clf()
    raw_ymin = ymax if ymin is None else ymin
    for approval_history in approval_histories:
        # plt.plot(np.arange(T - 1), approval_history, "g-")
        loss_history = np.array(approval_history.policy_loss_history)
        T = loss_history.size + 1
        running_avg = np.cumsum(loss_history) / np.arange(1, T)
        plt.plot(
            np.arange(T - 1),
            running_avg,
            label=approval_history.policy_name
            if approval_history.policy_name != "ValidationPolicy"
            else "MarkovHedge",
        )
        raw_ymin = min(np.min(running_avg), raw_ymin)
    plt.ylabel("Loss")
    plt.xlabel("Time")
    plt.legend()
    plt.hlines(y=alpha, xmin=0, xmax=T)
    plt.ylim(max(raw_ymin - 0.05, 0) if ymin is None else ymin, ymax)
    plt.savefig(fig_name)


def plot_human_uses(approval_histories, fig_name):
    plt.clf()
    # plt.figure(figsize=(6, 6))
    for approval_history in approval_histories:
        human_history = np.array(approval_history.human_history)
        T = human_history.size + 1
        plt.plot(
            np.arange(T - 1),
            np.cumsum(human_history) / np.arange(1, T),
            label=approval_history.policy_name
            if approval_history.policy_name != "ValidationPolicy"
            else "MarkovHedge",
        )
    plt.ylim(0, 1)
    plt.ylabel("Fail-safe prob")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig(fig_name)


def main(args=sys.argv[1:]):
    args = parse_args(args)
    print(args)
    np.random.seed(args.seed)

    approval_histories = [
        pickle_from_file(history_file) for history_file in args.history_files
    ]
    # for app, f in zip(approval_histories, args.history_files):
    #    app.policy_name = f.split("/")[-1].replace(".pkl", "")
    approval_histories = sorted(approval_histories, key=lambda x: x.policy_name)

    plot_losses(
        approval_histories,
        args.loss_plot,
        alpha=approval_histories[0].human_max_loss,
        ymin=args.y_min,
        ymax=min(approval_histories[0].human_max_loss * 5, args.y_max),
    )
    plot_human_uses(
        approval_histories,
        args.human_plot,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
