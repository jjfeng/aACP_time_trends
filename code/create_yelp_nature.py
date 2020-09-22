"""
Creates "nature" object, which contains yelp data
"""
import sys
import argparse
import logging
import numpy as np
from numpy import ndarray
from typing import List

from trial_data import TrialDataFromDisk
from nature import FixedNature
from data_generator import *
from support_sim_settings import *
from dataset import Dataset
from common import pickle_to_file


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--num-years", type=int, default=1)
    parser.add_argument("--num-months", type=int, default=1)
    parser.add_argument("--log-file", type=str, default="_output/nature_log.txt")
    parser.add_argument("--out-file", type=str, default="_output/nature.pkl")
    parser.set_defaults()
    args = parser.parse_args()

    return args


def main(args=sys.argv[1:]):
    # TODO: need to check that this is valid data to use
    YELP_TEST = "data/yelp_academic_dataset_review_year_valid_%s_%s.json"

    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    print(args)
    logging.info(args)

    trial_data = TrialDataFromDisk()
    times = []
    models = []
    YEARS = range(args.start_year, args.start_year + args.num_years)
    MONTHS = range(1, 1 + args.num_months)
    for year in YEARS:
        for month in MONTHS:
            times.append((year, month))

    for idx, time_key in enumerate(times):
        path_time = YELP_TEST % time_key
        trial_data.add_batch(path_time)
    nature = FixedNature(trial_data=trial_data)

    pickle_to_file(nature, args.out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
