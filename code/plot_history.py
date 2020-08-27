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
    parser.add_argument("--human-max-loss", type=float, default=0.9)
    parser.add_argument("--y-max", type=float, default=0.9)
    parser.add_argument("--y-min", type=float, default=0.1)
    parser.add_argument(
        "--policy-name",
        type=str,
        help="name of approval policy",
        default="FixedShare",
        choices=["FixedShare", "BlindApproval"],
    )
    parser.add_argument("--history-file", type=str, default="_output/approver_history.pkl")
    parser.add_argument("--loss-plot", type=str, default="_output/approver_history_loss.png")
    parser.add_argument("--human-plot", type=str, default="_output/approver_history_human.png")
    parser.set_defaults()
    args = parser.parse_args()

    return args

def main(args=sys.argv[1:]):
    args = parse_args(args)
    print(args)
    np.random.seed(args.seed)

    approval_history = pickle_from_file(args.history_file)
    print(approval_history)

    plot_loss(
            np.array(approval_history.policy_loss_history), args.loss_plot, alpha=args.human_max_loss, ymin=args.y_min, ymax=args.y_max)
    plot_human_use(np.array(approval_history.human_history), args.human_plot)


if __name__ == "__main__":
    main(sys.argv[1:])

