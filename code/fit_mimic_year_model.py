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

from proposer_lasso import LogisticRegressionCVWrap


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--seed",
        type=int,
        help="Random number generator seed for replicability",
        default=0,
    )
    parser.add_argument("--num-back-years", type=int, default=3)
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument("--year", type=int, default=2010)
    parser.add_argument("--quarter", type=int, default=0)
    parser.add_argument("--log-file", type=str, default="_output/model_log.txt")
    parser.add_argument("--out-file", type=str, default="_output/model.pkl")
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

    quarters = list(range(4))
    quarters = quarters[args.quarter + 1:] + quarters[: args.quarter + 1]
    if args.start_year == args.year:
        quarters = range(args.quarter + 1)
    dat = np.concatenate([
		np.genfromtxt(MIMIC_TRAIN % (year, quarter)) for year in range(max(args.year -
            args.num_back_years, args.start_year), args.year + 1) for quarter
        in quarters])
    x_train = dat[:, 1:]
    y_train = dat[:, 0]
    model = LogisticRegressionCVWrap(max_iter=1000, cv=3)
    model.fit(x_train, y_train)
    # Do save
    with open(args.out_file, "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main(sys.argv[1:])
