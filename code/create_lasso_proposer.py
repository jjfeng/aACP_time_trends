"""
Creates model proposer
"""
import sys
import argparse
import logging
import numpy as np
from numpy import ndarray
from typing import List

from trial_data import TrialData
from nature import FixedNature
from proposer_lasso import LassoProposer
from common import pickle_to_file, process_params


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--seed",
        type=int,
        help="Random number generator seed for replicability",
        default=0,
    )
    parser.add_argument(
        "--density-parametric-form",
        type=str,
        default="bernoulli",
        help="The parametric form we are going to use for Y|X",
        choices=["gaussian", "bernoulli"],
    )
    parser.add_argument("--proposer-eps", type=float, default=1e-4)
    parser.add_argument("--proposer-cv", type=int, default=3)
    parser.add_argument("--proposer-batches", type=str, default="1")
    parser.add_argument("--proposer-alphas", type=int, default=30)
    parser.add_argument("--proposer-offset-scale", type=float, default=0)
    parser.add_argument("--min-y", type=float, default=-1)
    parser.add_argument("--max-y", type=float, default=1)
    parser.add_argument("--log-file", type=str, default="_output/proposer_log.txt")
    parser.add_argument("--out-file", type=str, default="_output/proposer.pkl")
    parser.set_defaults()
    args = parser.parse_args()

    assert args.min_y < args.max_y
    args.proposer_batches = process_params(args.proposer_batches, int)
    assert np.all(np.array(args.proposer_batches) > 0)
    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    print(args)
    logging.info(args)

    np.random.seed(args.seed)

    proposer = LassoProposer(
            args.density_parametric_form,
            eps=args.proposer_eps,
            n_alphas=args.proposer_alphas,
            cv=args.proposer_cv,
            num_back_batches=args.proposer_batches,
    )

    pickle_to_file(proposer, args.out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
