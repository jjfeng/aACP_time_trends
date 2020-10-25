import sys
import time
import os
import shutil
import argparse
import numpy as np
import scipy.stats
from numpy import ndarray
from typing import List
import pandas as pd

from nature import Nature
from proposer import Proposer
from proposer import FixedProposerFromFile
from approval_history import ApprovalHistory
from model_preds_and_targets import ModelPredsAndTargets, AggModelPredsAndTargets
from common import pickle_from_file, pickle_to_file


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("path_template", type=str)
    parser.add_argument("--max-loss", type=float, default=4)
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--num-years", type=int, default=1)
    parser.add_argument("--start-num-year-splits", type=int, default=0)
    parser.add_argument("--end-num-year-splits", type=int, default=1)
    parser.add_argument(
        "--out-file", type=str, default="_output/model_preds_and_targets.pkl"
    )
    parser.set_defaults()
    args = parser.parse_args()

    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)

    agg_model_preds_and_targets = AggModelPredsAndTargets()
    for year in range(args.start_year, args.start_year + args.num_years):
        for split_idx in range(args.start_num_year_splits, args.end_num_year_splits):
            prefetch_file = args.path_template % (year, split_idx)
            model_preds_and_targets = pickle_from_file(prefetch_file)
            agg_model_preds_and_targets.append(model_preds_and_targets)

    pickle_to_file(agg_model_preds_and_targets, args.out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
