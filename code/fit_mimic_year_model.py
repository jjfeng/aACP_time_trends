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

from sklearn.linear_model import LogisticRegression


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--seed",
        type=int,
        help="Random number generator seed for replicability",
        default=0,
    )
    parser.add_argument("--year", type=int, default=2010)
    parser.add_argument("--log-file", type=str, default="_output/model_log.txt")
    parser.add_argument("--out-file", type=str, default="_output/model.pkl")
    parser.set_defaults()
    args = parser.parse_args()

    return args


def main(args=sys.argv[1:]):
    MIMIC_TRAIN = "experiment_mimic/_output/data/data_%d.csv"

    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    print(args)
    logging.info(args)

    np.random.seed(args.seed)

    dat = np.genfromtxt(MIMIC_TRAIN % args.year)
    print(dat)
    x_train = dat[:, 1:]
    y_train = dat[:, 0]
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    # Do save
    with open(args.out_file, "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main(sys.argv[1:])
