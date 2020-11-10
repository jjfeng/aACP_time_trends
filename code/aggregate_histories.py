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
from common import pickle_from_file, pickle_to_file, process_params


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--seed",
        type=int,
        help="Random number generator seed for replicability",
        default=12,
    )
    parser.add_argument(
        "--history-files", type=str, help="comma separated pickle files"
    )
    parser.add_argument(
        "--out-file", type=str, default="_output/approver_history_agg.pkl"
    )
    parser.set_defaults()
    args = parser.parse_args()

    args.history_files = process_params(args.history_files, str)

    return args



def main(args=sys.argv[1:]):
    args = parse_args(args)
    print(args)
    np.random.seed(args.seed)

    all_approval_list = [
        pickle_from_file(history_file) for history_file in args.history_files
    ]
    for hist in all_approval_list:
        print(hist)
    all_approval_dict = {x.policy_name: x for x in all_approval_list}
    pickle_to_file(all_approval_dict, args.out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
