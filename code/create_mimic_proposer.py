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
from proposer import FixedProposer
from common import pickle_to_file, pickle_from_file


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("path_template", type=str)
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--num-years", type=int, default=1)
    parser.add_argument("--out-file", type=str, default="_output/proposer.pkl")
    parser.set_defaults()
    args = parser.parse_args()

    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)

    models = []
    for year in range(args.start_year, args.start_year + args.num_years):
        model_file = args.path_template % year
        print("model", model_file)
        assert os.path.exists(model_file)
        models.append(pickle_from_file(model_file))

    proposer = FixedProposer(models)

    pickle_to_file(proposer, args.out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
