"""
Creates "nature" object, which is in chrage of spurting out batched data
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
from support_sim_settings import *
from dataset import Dataset
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
        "--density-parametric-form",
        type=str,
        default="bernoulli",
        help="The parametric form we are going to use for Y|X",
        choices=["gaussian", "bernoulli"],
    )
    parser.add_argument(
        "--sim-func-name", type=str, default="linear", choices=["linear", "curvy"]
    )
    parser.add_argument("--drift-cycle", type=int, default=0)
    parser.add_argument("--num-coef-drift", type=int, default=0)
    parser.add_argument("--num-p", type=int, default=50)
    parser.add_argument(
        "--support-setting", type=str, default="constant", choices=["constant"]
    )
    parser.add_argument("--min-y", type=float, default=-1)
    parser.add_argument("--max-y", type=float, default=1)
    parser.add_argument("--y-sigma", type=float, default=1)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--first-batch-size", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--batch-incr", type=int, default=0)
    parser.add_argument("--coef-scale", type=float, default=5)
    parser.add_argument("--num-coefs", type=int, default=5)
    parser.add_argument("--log-file", type=str, default="_output/nature_log.txt")
    parser.add_argument("--out-file", type=str, default="_output/nature.pkl")
    parser.set_defaults()
    args = parser.parse_args()

    assert args.num_batches > 1
    assert args.min_y < args.max_y
    args.batch_sizes = [args.first_batch_size] + [
        args.batch_size + args.batch_incr * i for i in range(args.num_batches - 1)
    ]
    return args


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
        noise_sd=args.y_sigma,
        max_y=args.max_y,
        min_y=args.min_y,
    )
    trial_data = TrialData(args.batch_sizes)
    init_coef = np.zeros(args.num_p)
    init_coef[: args.num_coefs] = args.coef_scale
    new_coef = init_coef
    coef_norm = np.sqrt(np.sum(np.power(init_coef, 2)))
    for batch_index in range(args.num_batches):
        do_drift = (
            batch_index % args.drift_cycle == args.drift_cycle - 1 if args.drift_cycle > 0 else False
        )
        if do_drift:
            print("DRIFT", do_drift)
            new_coef = np.copy(new_coef)
            to0_rand_idx = np.random.choice(
                np.where(np.abs(new_coef) > 0)[0], size=args.num_coef_drift
            )
            to1_rand_idx = np.random.choice(
                np.where(np.abs(new_coef) <= 1e-10)[0], size=args.num_coef_drift
            )
            new_coef[to0_rand_idx] = 0
            new_coef[to1_rand_idx] = np.max(init_coef)
        new_data = data_gen.create_data(
            args.batch_sizes[batch_index], batch_index, coef=new_coef
        )
        trial_data.add_batch(new_data)
    nature = FixedNature(
        data_gen, trial_data, coefs=[init_coef] * trial_data.num_batches
    )

    pickle_to_file(nature, args.out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
