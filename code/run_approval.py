import sys
import time
import os
import shutil
import argparse
import logging
import numpy as np
import scipy.stats
from numpy import ndarray
from typing import List
import pandas as pd

from simulation import *
from nature import Nature
from proposer import Proposer
from approval_history import ApprovalHistory
from policy import *
from validation_policies import *
from regret_bounds_restrict_drift_stochastic import get_regret_bounds
from common import pickle_from_file, pickle_to_file


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--seed",
        type=int,
        help="Random number generator seed for replicability",
        default=12,
    )
    parser.add_argument("--nature-file", type=str, default="_output/nature.pkl")
    parser.add_argument("--proposer-file", type=str, default="_output/proposer.pkl")
    parser.add_argument("--prefetched-file", type=str, default=None)
    parser.add_argument(
        "--policy-name", type=str, help="name of approval policy",
    )
    parser.add_argument("--human-max-loss", type=float, default=None)
    parser.add_argument("--drift-scale", type=float, default=1)
    parser.add_argument("--eta", type=float, default=1)
    parser.add_argument("--holdout-last-batch", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument(
        "--ci-alpha",
        type=float,
        default=0.025,
        help="""This will
            be the z-factor used (<0.5 means using lower bound, > 0.5 means
            using upper bound""",
    )
    parser.add_argument("--num-back-batches", type=int, default=3)
    parser.add_argument("--control-error-factor", type=float, default=1.5)
    parser.add_argument("--constraint-factor", type=float, default=None)
    parser.add_argument("--num-test-obs", type=int, default=1000)
    parser.add_argument("--log-file", type=str, default="_output/log.txt")
    parser.add_argument("--out-file", type=str, default="_output/approver_history.pkl")
    parser.add_argument("--out-nature-file", type=str, default=None)
    parser.set_defaults()
    args = parser.parse_args()

    if args.constraint_factor is None:
        args.constraint_factor = args.control_error_factor - 1
    return args


def create_policy(
    policy_name, args, human_max_loss, drift, total_time, num_experts, batch_size
):
    logging.info("MEAN BATCH SIZE %.2f", batch_size)
    constraint_ni_margin = args.constraint_factor * human_max_loss
    if policy_name == "MarkovHedge":
        policy = ValidationPolicy(
            num_experts=num_experts,
            etas=np.array([args.eta, 0, args.alpha, 0.05]),
            human_max_loss=human_max_loss,
            # const_baseline_weight=0.5,
        )
    elif policy_name == "MetaGridSearch":
        eta_grid = [
            np.exp(np.arange(-3, 3, 1)),
            np.exp(np.arange(-3, 4, 1)),
            np.arange(0, 1.01, 0.1),
            np.arange(0, 1, 0.1),
        ]
        policy = MetaGridSearch(
            eta=args.eta,
            eta_grid=eta_grid,
            num_experts=num_experts,
            human_max_loss=human_max_loss,
        )
    elif policy_name.startswith("MetaExpWeighting"):
        if policy_name == "MetaExpWeightingSmall":
            eta_list = [
                (0, 0, 0, 1),  # baseline
                (0, 0, 0.99, 0.0),  # blind
                (0, 10000, 0.5, 0),  # t-test
                (1.5, 0, 0.3, 0.05),  # online
            ]
        elif policy_name == "MetaExpWeighting":
            eta_list = [
                (0, 0, 0, 1),  # baseline
                (0, 0, 0.999, 0.0),  # blind
                (0, 10000, 0.5, 0),  # t-test
                (10, 0, 0.3, 0),
                (10, 10, 0.3, 0),
                (10, 100, 0.3, 0),
                (10, 0, 0.5, 0),
                (10, 10, 0.5, 0),
                (10, 100, 0.5, 0),
                (10, 0, 0.8, 0),
                (10, 10, 0.8, 0),
                (10, 100, 0.8, 0),
            ]
        meta_weights = np.ones(len(eta_list))
        meta_weights /= np.sum(meta_weights)
        print(meta_weights)
        lambdas = np.exp(np.arange(-6, 2, 0.02))
        regret_bounds = get_regret_bounds(
            meta_weights=meta_weights,
            T=total_time,
            delta=human_max_loss,
            ni_margin=constraint_ni_margin,
            drift=drift,
            lambdas=lambdas,
            n=batch_size,
            alpha=args.ci_alpha,
        )
        best_bound = np.min(regret_bounds)
        logging.info("baseline_weight %f", meta_weights[0] / np.sum(meta_weights))
        logging.info("human max %f", human_max_loss)
        logging.info(
            "BEST BOUND %f, desired control %f",
            best_bound,
            human_max_loss * args.control_error_factor,
        )
        print("BOUNDS...", best_bound, human_max_loss * args.control_error_factor)
        if best_bound > (human_max_loss * args.control_error_factor):
            logging.info("WARNING. regret bounds not satisfied: best %f < desired %f",
                    best_bound, human_max_loss * args.control_error_factor)
            print("WARNING: regret bounds not satisfies")
        loss_diffs = human_max_loss * args.control_error_factor - regret_bounds
        print("BATCH SIZE", batch_size)
        if np.all(loss_diffs < 0):
            eta_idx = np.argmin(regret_bounds)
            logging.info(
                "lambda %f with smallest bound %f",
                lambdas[eta_idx],
                regret_bounds[eta_idx],
            )
        else:
            eta_idx = np.max(np.where(loss_diffs >= 0))
            logging.info(
                "closest lambda %f, bound %f", lambdas[eta_idx], regret_bounds[eta_idx]
            )
        eta = lambdas[eta_idx]
        print("ETAS", eta)
        policy = MetaExpWeightingList(
            eta=eta,
            eta_list=eta_list,
            meta_weights=meta_weights,
            num_experts=num_experts,
            human_max_loss=human_max_loss,
            ci_alpha=args.ci_alpha,
            num_back_batches=args.num_back_batches,
            ni_margin=constraint_ni_margin,
        )
    elif policy_name == "BaselinePolicy":
        policy = BaselinePolicy(human_max_loss=human_max_loss)
    elif policy_name == "BlindApproval":
        policy = BlindApproval(human_max_loss=human_max_loss)
    elif policy_name == "MeanApproval":
        policy = MeanApproval(num_experts, human_max_loss=human_max_loss)
    elif policy_name == "FixedPolicy":
        policy = FixedPolicy(human_max_loss)
    elif policy_name == "TTestApproval":
        policy = TTestApproval(
            num_experts,
            human_max_loss=human_max_loss,
            ci_alpha=args.ci_alpha,
            ni_margin=constraint_ni_margin,
        )
    elif policy_name == "Oracle":
        policy = OracleApproval(human_max_loss=human_max_loss)
    else:
        raise ValueError("approval not found")
    return policy


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    print(args)
    logging.info(args)
    np.random.seed(args.seed)

    nature = pickle_from_file(args.nature_file)
    logging.info("BATCH SIZES %s", nature.batch_sizes)
    proposer = pickle_from_file(args.proposer_file)

    nature.next(None)
    model = proposer.propose_model(nature.get_trial_data(0), None)
    if args.human_max_loss is None:
        args.human_max_loss = np.mean(
            proposer.score_models(
                nature.create_test_data(time_t=0, num_obs=args.num_test_obs)
            )[0]
        )
        logging.info("HUMAN MAX %f", args.human_max_loss)
    nature.next(None)

    print("POLICY")
    policy = create_policy(
        args.policy_name,
        args,
        human_max_loss=args.human_max_loss,
        drift=args.human_max_loss * args.drift_scale,
        total_time=nature.total_time,
        num_experts=nature.total_time,
        batch_size=np.mean(nature.batch_sizes[1:]),
    )

    st_time = time.time()
    if args.prefetched_file is None:
        sim = Simulation(
            nature,
            proposer,
            policy,
            args.human_max_loss,
            num_test_obs=args.num_test_obs,
            holdout_last_batch=args.holdout_last_batch,
        )
    else:
        prefetched = pickle_from_file(args.prefetched_file)
        sim = SimulationPrefetched(
            nature,
            proposer,
            prefetched,
            policy,
            args.human_max_loss,
            num_test_obs=args.num_test_obs,
            holdout_last_batch=args.holdout_last_batch,
        )
    sim.run(lambda approval_hist: pickle_to_file(approval_hist, args.out_file))
    logging.info(sim.approval_hist)
    print(sim.approval_hist)
    logging.info("run time %d", time.time() - st_time)

    pickle_to_file(sim.approval_hist, args.out_file)
    if args.out_nature_file is not None:
        pickle_to_file(nature.to_fixed(), args.out_nature_file)


if __name__ == "__main__":
    main(sys.argv[1:])
