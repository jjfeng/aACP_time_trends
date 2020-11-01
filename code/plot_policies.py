import sys
import os
import shutil
import argparse
import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from typing import List

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
    parser.add_argument("--x-start", type=int, default=None)
    parser.add_argument("--x-skip", type=int, default=None)
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
    parser.add_argument("--log-file", type=str, default="_output/log.txt")
    parser.set_defaults()
    args = parser.parse_args()

    args.history_files = process_params(args.history_files, str)

    return args

def labeler(label):
    return label if label != "ValidationPolicy" else "MarkovHedge"

def plot_losses(approval_history_dict, fig_name, alpha, scale_loss, ymin, ymax,
        plot_mean, key_order: List[str], x_start: int = None, x_skip: int = None):
    plt.clf()
    raw_ymin = ymax if ymin is None else ymin
    mean_data_frames = []
    raw_data_frames = []
    for policy_key in key_order:
        policy_label = labeler(policy_key)
        for approval_history in approval_history_dict[policy_key]:
            # plt.plot(np.arange(T - 1), approval_history, "g-")
            loss_history = np.array(approval_history.expected_policy_loss_history)
            T = loss_history.size + 1
            running_avg = np.cumsum(loss_history) / np.arange(1, T)
            mean_data = pd.DataFrame(
                {
                    "Time": np.arange(T - 1),
                    "Loss": running_avg * scale_loss,
                    "Policy": policy_label,
                    "Policy_type": not policy_label.startswith("Learning-to-Approve"),
                }
            )
            raw_data = pd.DataFrame(
                {
                    "Time": np.arange(T - 1),
                    "Loss": loss_history * scale_loss,
                    "Policy": policy_label,
                    "Policy_type": not policy_label.startswith("Learning-to-Approve"),
                }
            )
            mean_data_frames.append(mean_data)
            raw_data_frames.append(raw_data)
            raw_ymin = min(np.min(running_avg), raw_ymin)
    mean_data = pd.concat(mean_data_frames)
    raw_data = pd.concat(raw_data_frames)
    if plot_mean:
        ax = sns.lineplot(
            x="Time",
            y="Loss",
            hue="Policy",
            data=mean_data,
            style="Policy_type",
            legend=False,
        )
    else:
        ax = sns.lmplot(
            x="Time",
            y="Loss",
            hue="Policy",
            style="Policy_type",
            data=raw_data,
            lowess=True,
            scatter=False,
        )
    plt.ylabel("Loss" if not plot_mean else "Cum avg Loss")
    plt.xlabel("Time")
    plt.hlines(y=alpha, xmin=0, xmax=T, colors='black')
    plt.ylim(max(raw_ymin - 0.05, 0) if ymin is None else ymin, ymax)
    if x_start is not None:
        tick_locs = np.arange(0, np.max(mean_data.Time), x_skip)
        plt.xticks(tick_locs,
                np.arange(x_start, x_start + tick_locs.size))
        plt.xlabel("Year")
    plt.legend([labeler(k) for k in key_order])
    plt.tight_layout()
    plt.savefig(fig_name)

    mean_loss = raw_data.groupby("Policy").mean().reset_index()
    print(mean_loss.to_latex())
    logging.info(mean_loss[["Policy", "Loss"]].to_latex(index=False))


def plot_human_uses(approval_history_dict, fig_name, plot_mean: bool, key_order,
        x_start: int = None, x_skip: int = None):
    plt.clf()
    # plt.figure(figsize=(6, 6))
    raw_data_frames = []
    mean_data_frames = []
    for policy_key in key_order:
        policy_label = labeler(policy_key)
        for approval_history in approval_history_dict[policy_key]:
            human_history = np.array(approval_history.human_history)
            T = human_history.size + 1
            mean_data_frames.append(pd.DataFrame(
                {
                    "Time": np.arange(T - 1),
                    "prob": np.cumsum(human_history) / np.arange(1, T),
                    "Policy": policy_label,
                    "Policy_type": not policy_label.startswith("Learning-to-Approve"),
                }
            ))
            raw_data_frames.append(pd.DataFrame(
                {
                    "Time": np.arange(T - 1),
                    "prob": human_history,
                    "Policy": policy_label,
                    "Policy_type": not policy_label.startswith("Learning-to-Approve"),
                }
            ))
    mean_data = pd.concat(mean_data_frames)
    raw_data = pd.concat(raw_data_frames)
    if plot_mean:
        ax = sns.lineplot(
            x="Time",
            y="prob",
            hue="Policy",
            style="Policy_type",
            data=mean_data,
            legend=False,
        )
    else:
        ax = sns.lmplot(
            x="Time",
            y="prob",
            hue="Policy",
            data=raw_data,
            lowess=True,
            scatter=False,
            legend=False,
        )
    plt.ylim(0, 1)
    plt.ylabel("Fail-safe-prob" if not plot_mean else "Cum avg Fail-safe prob")
    plt.xlabel("Time")
    if x_start is not None:
        tick_locs = np.arange(0, np.max(mean_data.Time), x_skip)
        plt.xticks(tick_locs,
                np.arange(x_start, x_start + tick_locs.size))
        plt.xlabel("Year")
    plt.legend([labeler(k) for k in key_order])
    plt.tight_layout()
    plt.savefig(fig_name)


def main(args=sys.argv[1:]):
    args = parse_args(args)
    print(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    np.random.seed(args.seed)

    # Load policy histories
    all_approval_histories = [
        pickle_from_file(history_file) for history_file in args.history_files
    ]
    approval_history_dict = {x: [] for x in all_approval_histories[0].keys()}
    approval_history_keys = list(approval_history_dict.keys())
    human_max_losses = []
    for curr_approval_hist in all_approval_histories:
        for k in approval_history_dict.keys():
            approval_history_dict[k].append(curr_approval_hist[k])
            human_max_losses.append(curr_approval_hist[k].human_max_loss)
    human_max_loss = np.mean(human_max_losses)
    print("HUMAN MAX", human_max_loss)

    # Sort keys
    sorted_keys = sorted([k for k in approval_history_keys if not
        k.startswith("Learning-to-Approve")])
    ordered_approval_history_keys = ["Learning-to-Approve-4",
            "Learning-to-Approve-15"] + [k for k in sorted_keys if k != "Fixed"]
    if "Fixed" in sorted_keys:
        ordered_approval_history_keys.append("Fixed")

    plot_losses(
        approval_history_dict,
        args.loss_plot,
        alpha=human_max_loss * args.scale_loss,
        scale_loss=args.scale_loss,
        ymin=args.y_min,
        ymax=args.y_max,
        plot_mean=args.plot_mean,
        key_order=ordered_approval_history_keys,
        x_start=args.x_start,
        x_skip=args.x_skip,
    )
    plot_human_uses(
        approval_history_dict,
        args.human_plot,
        plot_mean=args.plot_mean,
        key_order=ordered_approval_history_keys,
        x_start=args.x_start,
        x_skip=args.x_skip,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
