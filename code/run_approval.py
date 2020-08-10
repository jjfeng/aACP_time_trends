import sys
import os
import shutil
import argparse
import logging
import numpy as np
from numpy import ndarray
from typing import List
import pandas as pd

# from approval_simulation_common import *
from time_trend_predictor import ARIMAPredictor
from nature import Nature
from proposer import Proposer
from approval_history import ApprovalHistory
from policy import Policy, OptimisticMirrorDescent
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
    parser.add_argument("--human-max-loss", type=float, default=0.9)
    parser.add_argument("--nature-file", type=str, default="_output/nature.pkl")
    parser.add_argument("--proposer-file", type=str, default="_output/proposer.pkl")
    parser.add_argument(
        "--policy-name",
        type=str,
        help="name of approval policy",
        default="OMD",
        choices=["OMD",],
    )
    parser.add_argument("--eta", type=float, default=1)
    parser.add_argument("--log-file", type=str, default="_output/log.txt")
    parser.add_argument("--out-file", type=str, default="_output/approver_history.pkl")
    parser.set_defaults()
    args = parser.parse_args()

    return args


def create_policy(policy_name, args, human_max_loss, num_experts):
    assert policy_name == "OMD"

    time_trend_predictor = ARIMAPredictor(
        order=(1, 2, 0), min_size=7, max_loss=human_max_loss + 0.1
    )
    policy = OptimisticMirrorDescent(
        num_experts,
        eta=args.eta,
        human_max_loss=human_max_loss,
        time_trend_predictor=time_trend_predictor,
    )
    return policy

def run_simulation(
        nature: Nature, proposer: Proposer, policy: Policy
):
    # Create the data generated each batch
    proposer.propose_model(nature.get_trial_data(0))

    # Run the platform trial
    indiv_loss_robot_t = None
    prev_weights = None
    approval_hist = ApprovalHistory()
    for t in range(nature.total_time - 1):
        print("TIME STEP", t)
        policy.update_weights(t, indiv_loss_robot_t, prev_weights=prev_weights)
        policy.add_expert(t)
        robot_weights, human_weight = policy.get_predict_weights(t)
        weights = np.concatenate([[human_weight], robot_weights])

        sub_trial_data = nature.get_trial_data(t + 1)
        indiv_loss_robot_t = proposer.score_models(sub_trial_data.batch_data[-1])
        batch_n = indiv_loss_robot_t.shape[1]
        all_loss_t = np.concatenate(
            [[policy.human_max_loss * batch_n], np.sum(indiv_loss_robot_t, axis=1)]
        )
        policy_loss_t = np.sum(all_loss_t * weights) / batch_n
        approval_hist.append(human_weight, policy_loss_t, all_loss_t)
        prev_weights = weights

        proposer.propose_model(sub_trial_data)

    return approval_hist

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

    policy = create_policy(
        args.policy_name,
        args,
        human_max_loss=args.human_max_loss,
        num_experts=nature.total_time,
    )

    approval_history = run_simulation(nature, proposer, policy)

    pickle_to_file(approval_history, args.out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
