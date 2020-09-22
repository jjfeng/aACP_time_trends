"""
Creates model for yelp data for year and month
"""
import sys
import argparse
import logging
import numpy as np
from numpy import ndarray
from typing import List
import torch

from yelp_online_learning import train_rating_model_year_month


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--seed",
        type=int,
        help="Random number generator seed for replicability",
        default=0,
    )
    parser.add_argument("--year", type=int, default=2008)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-hidden", type=int, default=10)
    parser.add_argument("--log-file", type=str, default="_output/model_log.txt")
    parser.add_argument("--out-file", type=str, default="_output/model.pkl")
    parser.set_defaults()
    args = parser.parse_args()

    return args


def main(args=sys.argv[1:]):
    YELP_TRAIN = "data/yelp_academic_dataset_review_year_train_%s_%s.json"

    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    print(args)
    logging.info(args)

    np.random.seed(args.seed)

    model, fields = train_rating_model_year_month(
        YELP_TRAIN % (str(args.year), str(args.month)),
        n_epochs=args.epochs,
        num_hidden=args.num_hidden,
    )
    # Do save
    model_state_dict = {
        "state_dict": model.state_dict(),
        "num_hidden": args.num_hidden,
        "fields": fields,
        "year": args.year,
        "month": args.month,
    }
    torch.save(model_state_dict, args.out_file)


if __name__ == "__main__":
    main(sys.argv[1:])