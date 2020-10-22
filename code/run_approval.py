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

from simulation import Simulation
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
    parser.add_argument(
        "--policy-name",
        type=str,
        help="name of approval policy",
    )
    parser.add_argument("--human-max-loss", type=float, default=None)
    parser.add_argument("--eta", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--ci-alpha", type=float, default=0.025)
    parser.add_argument("--control-error-factor", type=float, default=1.5)
    parser.add_argument("--num-test-obs", type=int, default=1000)
    parser.add_argument("--log-file", type=str, default="_output/log.txt")
    parser.add_argument("--out-file", type=str, default="_output/approver_history.pkl")
    parser.add_argument("--out-nature-file", type=str, default=None)
    parser.set_defaults()
    args = parser.parse_args()

    return args


def create_policy(
    policy_name, args, human_max_loss, total_time, num_experts, batch_size
):
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
    elif policy_name == "MetaExpWeightingSmall":
        eta_list = [
            (0, 0, 0, 1),  # baseline
            (1.5, 0, 0.5, 0.05),  # online
            (0, 0, 0.99, 0.0),  # blind
            (0, 10000, 0.5, 0),  # t-test
        ]
        meta_weights = np.ones(len(eta_list)) * 16 / 4
        meta_weights[0] = 1
        policy = MetaExpWeightingList(
            eta=args.eta,
            eta_list=eta_list,
            meta_weights=meta_weights,
            num_experts=num_experts,
            human_max_loss=human_max_loss,
        )
    elif policy_name == "MetaExpWeighting":
        eta_list = [
            (0, 0, 0, 1),  # baseline
            # (1.5, 0, 0.5, 0.05),  # online
            (0, 0, 0.99, 0.0),  # blind
            (0, 10000, 0.5, 0),  # t-test
            (10, 0, 0.10, 0),
            (10, 1, 0.10, 0),
            (10, 10, 0.10, 0),
            (10, 100, 0.10, 0),
            (10, 0, 0.3, 0),
            (10, 1, 0.3, 0),
            (10, 10, 0.3, 0),
            (10, 100, 0.3, 0),
            (10, 0, 0.5, 0),
            (10, 1, 0.5, 0),
            (10, 10, 0.5, 0),
            (10, 100, 0.5, 0),
        ]
        meta_weights = np.ones(len(eta_list))
        lambdas = np.exp(np.arange(-6, 2, 0.02))
        regret_bounds = get_regret_bounds(
            alpha=args.ci_alpha,
            m=len(eta_list),
            T=total_time,
            delta=human_max_loss,
            drift=human_max_loss,
            lambdas=lambdas,
            n=batch_size,
        )
        best_bound = np.min(regret_bounds)
        logging.info("baseline_weight %f", meta_weights[0] / np.sum(meta_weights))
        logging.info("human max %f", human_max_loss)
        logging.info(
            "BEST BOUND %f, desired control %f",
            best_bound,
            human_max_loss * args.control_error_factor,
        )
        assert best_bound < human_max_loss * args.control_error_factor
        loss_diffs = human_max_loss * args.control_error_factor - regret_bounds
        eta_idx = np.max(np.where(loss_diffs >= 0))
        eta = lambdas[eta_idx]
        logging.info("closest lambda %f, bound %f", eta, regret_bounds[eta_idx])
        policy = MetaExpWeightingList(
            eta=eta,
            eta_list=eta_list,
            meta_weights=meta_weights,
            num_experts=num_experts,
            human_max_loss=human_max_loss,
            ci_alpha=args.ci_alpha,
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
            num_experts, human_max_loss=human_max_loss
        )  # , factor=1.6)
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
    proposer = pickle_from_file(args.proposer_file)

    nature.next(None)
    model = proposer.propose_model(nature.get_trial_data(0), None)
    if args.human_max_loss is None:
        args.human_max_loss = np.mean(
            proposer.score_models(nature.create_test_data(0))[0]
        )
        logging.info("HUMAN MAX %f", args.human_max_loss)

    policy = create_policy(
        args.policy_name,
        args,
        human_max_loss=args.human_max_loss,
        total_time=nature.total_time,
        num_experts=nature.total_time,
        batch_size=nature.batch_sizes[-1],
    )

    st_time = time.time()
    sim = Simulation(
        nature, proposer, policy, args.human_max_loss, num_test_obs=args.num_test_obs
    )
    sim.run()
    logging.info(sim.approval_hist)
    print(sim.approval_hist)
    logging.info("run time %d", time.time() - st_time)

    pickle_to_file(sim.approval_hist, args.out_file)
    if args.out_nature_file is not None:
        pickle_to_file(nature.to_fixed(), args.out_nature_file)


if __name__ == "__main__":
    main(sys.argv[1:])
