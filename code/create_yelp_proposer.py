"""
Creates model proposer for yelp data
"""
import os
import sys
import argparse
import logging
import numpy as np
from numpy import ndarray
from typing import List

from trial_data import TrialData
from nature import FixedNature
from proposer import FixedProposerFromFile
from common import pickle_to_file, process_params


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("path_template", type=str)
    parser.add_argument("--max-loss", type=float, default=5)
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--num-years", type=int, default=1)
    parser.add_argument("--num-months", type=int, default=1)
    parser.add_argument("--out-file", type=str, default="_output/proposer.pkl")
    parser.set_defaults()
    args = parser.parse_args()

    return args


def normalize_l1():
    torch.abs(x - y) / args.max_loss


def main(args=sys.argv[1:]):
    args = parse_args(args)

    YEARS = range(args.start_year, args.start_year + args.num_years)
    MONTHS = range(1, 1 + args.num_months)
    model_paths = []
    for year in YEARS:
        for month in MONTHS:
            model_file = args.path_template % (year, month)
            print("model", model_file)
            print(os.path.exists(model_file))
            if not os.path.exists(model_file):
                model_paths.append(prev_model_file)
            else:
                model_paths.append(model_file)
            prev_model_file = model_file
    proposer = FixedProposerFromFile(
        model_paths, criterion_str="l1", max_loss=args.max_loss
    )

    pickle_to_file(proposer, args.out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
