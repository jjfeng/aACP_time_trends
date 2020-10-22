"""
Creates "nature" object, which contains mimic data
"""
import sys
import argparse
import logging
import numpy as np
from numpy import ndarray
from typing import List

from trial_data import TrialData
from nature import FixedNature
from data_generator import *
from dataset import Dataset
from common import pickle_to_file


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument("--num-years", type=int, default=1)
    parser.add_argument("--log-file", type=str, default="_output/nature_log.txt")
    parser.add_argument("--out-file", type=str, default="_output/nature.pkl")
    parser.set_defaults()
    args = parser.parse_args()

    return args


def main(args=sys.argv[1:]):
    MIMIC_TEST = "experiment_mimic/_output/data/valid_data_%d_%d.csv"

    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    print(args)
    logging.info(args)

    trial_data = TrialData()
    times = []
    for time_key in range(args.start_year, args.start_year + args.num_years):
        for quarter in range(4):
            path_time = MIMIC_TEST % (time_key, quarter)
            raw_dataset = np.genfromtxt(path_time)
            if len(raw_dataset.shape) == 1:
                raw_dataset = raw_dataset.reshape((1, -1))
                print("VALIDATION DATA ONLY SIZE 1")
            print("year q", time_key, quarter)
            dataset = Dataset(raw_dataset[:, 1:], raw_dataset[:, 0], num_classes=2)
            trial_data.add_batch(dataset)
    nature = FixedNature(trial_data=trial_data)

    pickle_to_file(nature, args.out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
