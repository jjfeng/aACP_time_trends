import sys
import os
import shutil
import argparse
import logging
import numpy as np
import scipy.stats
from numpy import ndarray
from typing import List
import pandas as pd

# from approval_simulation_common import *
from approval_history import ApprovalHistory
from common import pickle_from_file, pickle_to_file, plot_loss, plot_human_use


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
    parser.add_argument("--y-min", type=float, default=0.1)
    parser.add_argument(
        "--policy-name", type=str, help="name of approval policy",
    )
    parser.add_argument(
        "--history-file", type=str, default="_output/approver_history.pkl"
    )
    parser.add_argument(
        "--loss-plot", type=str, default="_output/approver_history_loss.png"
    )
    parser.add_argument(
        "--human-plot", type=str, default="_output/approver_history_human.png"
    )
    parser.set_defaults()
    args = parser.parse_args()

    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)
    print(args)
    np.random.seed(args.seed)

    approval_history = pickle_from_file(args.history_file)
    print(approval_history)

    title = "%s: loss %.3f, human %.2f" % (
        args.policy_name,
        np.mean(approval_history.expected_policy_loss_history),
        np.mean(approval_history.human_history),
    )
    plot_loss(
        np.array(approval_history.expected_policy_loss_history),
        args.loss_plot,
        title=title,
        alpha=approval_history.human_max_loss,
        ymin=0,
        ymax=min(approval_history.human_max_loss * 5, args.y_max),
    )
    plot_human_use(
        np.array(approval_history.human_history), args.human_plot, title=title
    )


if __name__ == "__main__":
    main(sys.argv[1:])
