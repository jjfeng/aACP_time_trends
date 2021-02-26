#!/usr/bin/env python

"""
Creates model for mimic data for year and month
"""
import sys
import argparse
import logging
import pickle
import numpy as np
from numpy import ndarray
from typing import List
from matplotlib import pyplot as plt
import seaborn as sns

def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--seed",
        type=int,
        help="Random number generator seed for replicability",
        default=0,
    )
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument("--num-years", type=int, default=1)
    parser.add_argument("--min-batch-size", type=int, default=20)
    parser.add_argument("--log-file", type=str, default="_output/drift_mimic.txt")
    parser.add_argument("--out-file", type=str, default="_output/drift_mimic.png")
    parser.set_defaults()
    args = parser.parse_args()

    return args


def main(args=sys.argv[1:]):
    MIMIC_TRAIN = "experiment_mimic/_output/data/train_data_%d_%d.csv"

    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    print(args)
    logging.info(args)

    np.random.seed(args.seed)

    avg_ys = []
    for year in range(args.start_year, args.start_year + args.num_years):
        for quarter in range(4):
            dat = np.genfromtxt(MIMIC_TRAIN % (year, quarter))
            ntrain = dat.shape[0]
            x_train = dat[:ntrain, 1:]
            y_train = dat[:ntrain, 0]
            avg_ys.append(np.mean(y_train))
    logging.info(avg_ys)
    drift_dat = pd.DataFrame({
        "Time": np.arange(len(avg_ys)),
        "Mean Y": avg_ys})

    ax = sns.lineplot(
        x="Time",
        y="Mean y",
        data=drift_dat,
    )
    plt.tight_layout()
    plt.savefig(args.out_file)

if __name__ == "__main__":
    main(sys.argv[1:])
