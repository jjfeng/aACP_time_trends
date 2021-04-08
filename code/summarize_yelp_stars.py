"""
Plot drift in yelp data
"""
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import json

def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--seed",
        type=int,
        help="Random number generator seed for replicability",
        default=0,
    )
    parser.add_argument("--train-file", type=str)
    parser.add_argument("--year", type=int, default=2008)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--log-file", type=str, default="_output/model_log.txt")
    parser.add_argument("--out-file", type=str, default="_output/model.pkl")
    parser.set_defaults()
    args = parser.parse_args()

    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    print(args)
    logging.info(args)

    np.random.seed(args.seed)

    num_stars = []
    with open(args.train_file, "r") as f:
        for line in f:
            line_dict = json.loads(line)
            num_stars.append(line_dict["stars"])
    dat = pd.DataFrame({
        "year": [args.year],
        "month": [args.month],
        "stars": [np.mean(num_stars)]})
    dat.to_csv(args.out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
