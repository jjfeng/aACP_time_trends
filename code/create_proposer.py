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
from data_generator import DataGenerator
from support_sim_settings import *
from proposer_fine_control import FineControlProposer, MoodyFineControlProposer
from common import pickle_to_file


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
        "--update-engine",
        type=str,
        help="which model updater to use",
        default="fine_control",
        choices=["fine_control", "lasso", "moody"],
    )
    parser.add_argument(
        "--density-parametric-form",
        type=str,
        default="bernoulli",
        help="The parametric form we are going to use for Y|X",
        choices=["bounded_gaussian", "bernoulli"],
    )
    parser.add_argument(
        "--sim-func-name", type=str, default="linear", choices=["linear", "curvy"]
    )
    parser.add_argument("--num-p", type=int, default=50)
    parser.add_argument(
        "--support-setting", type=str, default="constant", choices=["constant"]
    )
    parser.add_argument("--proposer-noise", type=float, default=0.01)
    parser.add_argument("--proposer-increment", type=float, default=0)
    parser.add_argument("--proposer-decay", type=float, default=0)
    parser.add_argument("--proposer-offset-scale", type=float, default=0)
    parser.add_argument("--min-y", type=float, default=-1)
    parser.add_argument("--max-y", type=float, default=1)
    parser.add_argument("--log-file", type=str, default="_output/proposer_log.txt")
    parser.add_argument("--out-file", type=str, default="_output/proposer.pkl")
    parser.set_defaults()
    args = parser.parse_args()

    assert args.min_y < args.max_y
    return args


def get_proposer(args, data_gen):
    if args.update_engine == "moody":
        return MoodyFineControlProposer(
            data_gen,
            noise=args.proposer_noise,
            init_period=args.proposer_init_period,
            period=args.proposer_period,
            increment=args.proposer_increment,
            decay=args.proposer_decay,
        )
    elif args.update_engine == "fine_control":
        return FineControlProposer(
            data_gen,
            noise=args.proposer_noise,
            increment=args.proposer_increment,
            decay=args.proposer_decay,
            offset_scale=args.proposer_offset_scale,
        )
    else:
        raise ValueError("which proposer?")


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    print(args)
    logging.info(args)

    np.random.seed(args.seed)

    if args.support_setting == "constant":
        support_sim_settings = SupportSimSettingsUniform(
            args.num_p,
            min_func_name="min_x_func_constant",
            max_func_name="max_x_func_constant",
        )
    elif args.support_setting == "changing":
        raise ValueError("huh? i can get here?")
        support_sim_settings = SupportSimSettingsNormal(
            args.num_p,
            std_func_name="std_func_changing",
            mu_func_name="mu_func_changing",
        )
    else:
        raise ValueError("Asdfasdf")
    data_gen = DataGenerator(
        args.density_parametric_form,
        args.sim_func_name,
        support_sim_settings,
        max_y=args.max_y,
        min_y=args.min_y,
    )

    proposer = get_proposer(args, data_gen)

    pickle_to_file(proposer, args.out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
