import sys
import os
import shutil
import argparse
import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

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
    parser.add_argument("--scale-loss", type=float, default=1)
    parser.add_argument("--plot-mean", action="store_true")
    parser.add_argument(
        "--history-files", type=str, help="comma separated pickle files"
    )
    parser.add_argument(
        "--loss-plot", type=str, default="_output/approver_history_loss.png"
    )
    parser.add_argument(
        "--human-plot", type=str, default="_output/approver_history_human.png"
    )
    parser.add_argument(
        "--log-file", type=str, default="_output/log.txt"
    )
    parser.set_defaults()
    args = parser.parse_args()

    args.history_files = process_params(args.history_files, str)

    return args


def plot_losses(approval_histories, fig_name, alpha, scale_loss, ymin, ymax,
        plot_mean):
    plt.clf()
    raw_ymin = ymax if ymin is None else ymin
    mean_data_frames = []
    raw_data_frames = []
    for approval_history in approval_histories:
        # plt.plot(np.arange(T - 1), approval_history, "g-")
        print(approval_history.policy_name)
        loss_history = np.array(approval_history.expected_policy_loss_history)
        T = loss_history.size + 1
        running_avg = np.cumsum(loss_history) / np.arange(1, T)
        policy_label = (approval_history.policy_name
            if approval_history.policy_name != "ValidationPolicy"
            else "MarkovHedge")
        mean_data = pd.DataFrame({
            "Time":np.arange(T - 1),
            "Loss": running_avg * scale_loss,
            "Policy": policy_label,
            })
        raw_data = pd.DataFrame({
            "Time":np.arange(T - 1),
            "Loss": loss_history * scale_loss,
            "Policy": policy_label,
            })
        mean_data_frames.append(mean_data)
        raw_data_frames.append(raw_data)
        print(approval_history.policy_name)
        print(loss_history)
        print(running_avg)
        raw_ymin = min(np.min(running_avg), raw_ymin)
    mean_data = pd.concat(mean_data_frames)
    raw_data = pd.concat(raw_data_frames)
    sns.lmplot(
        x="Time",
        y="Loss",
        hue="Policy",
        data=mean_data if plot_mean else raw_data,
        lowess=True,
        scatter=False,
    )
    plt.ylabel("Loss" if not plot_mean else "Cum avg Loss")
    plt.xlabel("Time")
    plt.hlines(y=alpha, xmin=0, xmax=T)
    plt.ylim(max(raw_ymin - 0.05, 0) if ymin is None else ymin, ymax)
    plt.savefig(fig_name)


    mean_loss = raw_data.groupby("Policy").mean().reset_index()
    print(mean_loss.to_latex())
    logging.info(mean_loss[["Policy", "Loss"]].to_latex(index=False))


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
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    np.random.seed(args.seed)

    orig_approval_histories = [
        pickle_from_file(history_file) for history_file in args.history_files
    ]
    approval_histories = [
        x for x in orig_approval_histories if x.policy_name != "Fixed"
    ]
    approval_histories = sorted(approval_histories, key=lambda x: x.policy_name)
    for x in orig_approval_histories:
        if x.policy_name == "Fixed":
            approval_histories.append(x)

    plot_losses(
        approval_histories,
        args.loss_plot,
        alpha=approval_histories[0].human_max_loss * args.scale_loss,
        scale_loss=args.scale_loss,
        ymin=args.y_min,
        ymax=min(approval_histories[0].human_max_loss * 5, args.y_max),
        plot_mean=args.plot_mean,
    )
    plot_human_uses(
        approval_histories,
        args.human_plot,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
