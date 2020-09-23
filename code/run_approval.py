import sys
import os
import shutil
import argparse
import logging
import numpy as np
import scipy.stats
from numpy import ndarray
from typing import List
import pandas as pd

# from approval_simulation_common import *
from nature import Nature
from proposer import Proposer
from approval_history import ApprovalHistory
from policy import *
from validation_policies import *
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
        "--policy-name", type=str, help="name of approval policy",
    )
    parser.add_argument("--human-max-loss", type=float, default=None)
    parser.add_argument("--eta", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--log-file", type=str, default="_output/log.txt")
    parser.add_argument("--out-file", type=str, default="_output/approver_history.pkl")
    parser.add_argument("--out-nature-file", type=str, default=None)
    parser.set_defaults()
    args = parser.parse_args()

    return args


def create_policy(policy_name, args, human_max_loss, num_experts):
    if policy_name == "MarkovHedge":
        policy = ValidationPolicy(
            num_experts=num_experts,
            etas=np.array([args.eta, 0, args.alpha, 0.05]),
            human_max_loss=human_max_loss,
            const_baseline_weight=0.5,
        )
    elif policy_name == "MetaGridSearch":
        eta_grid = [
            np.exp(np.arange(-5, 1, 2)),
            np.exp(np.arange(-5, 7, 2)),
            np.arange(0, 1.01, 0.2),
            np.arange(0, 0.2, 0.05),
        ]
        policy = MetaGridSearch(
            eta=args.eta,
            eta_grid=eta_grid,
            num_experts=num_experts,
            human_max_loss=human_max_loss,
        )
    elif policy_name == "MetaExpWeighting":
        eta_list = [
            (0, 0, 0, 0),
            (10, 0, 0.3, 0.05),
            (0, 0, 1.0, 0.0),
            (0, 10000, 0.5, 0.05),
        ]
        meta_weights = np.ones(len(eta_list))
        meta_weights[1:] = 1/(len(eta_list) - 1)
        policy = MetaExpWeightingList(
            eta=args.eta,
            eta_list=eta_list,
            meta_weights=meta_weights,
            num_experts=num_experts,
            human_max_loss=human_max_loss,
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
        policy = TTestApproval(num_experts, human_max_loss=human_max_loss)
    else:
        raise ValueError("approval not found")
    return policy


def run_simulation(
    nature: Nature, proposer: Proposer, policy: Policy, human_max_loss: float
):
    approval_hist = ApprovalHistory(human_max_loss=human_max_loss)

    # Run the platform trial
    indiv_loss_robot_t = None
    prev_weights = None
    for t in range(nature.total_time - 1):
        logging.info("TIME STEP %d", t)
        policy.add_expert(t)
        policy.update_weights(t, indiv_loss_robot_t, prev_weights=prev_weights)
        robot_weights, human_weight = policy.get_predict_weights(t)
        # loss_predictions = policy.predict_next_losses(t)
        weights = np.concatenate([[human_weight], robot_weights])

        sub_trial_data = nature.get_trial_data(t + 1)
        indiv_loss_robot_t = proposer.score_models(sub_trial_data.batch_data[-1])
        batch_n = indiv_loss_robot_t.shape[1]
        all_loss_t = np.concatenate(
            [[policy.human_max_loss * batch_n], np.sum(indiv_loss_robot_t, axis=1)]
        )
        policy_loss_t = np.sum(all_loss_t * weights) / batch_n
        approval_hist.append(human_weight, robot_weights, policy_loss_t, all_loss_t)

        nature.next(approval_hist)

        prev_weights = weights
        print("time", t, "loss", policy_loss_t)
        logging.info("losses %s", all_loss_t / batch_n)
        # print("losses", all_loss_t/batch_n)
        # logging.info("loss pred %s", loss_predictions)
        # if loss_predictions.size > 2 and np.var(loss_predictions) > 0:
        #    logging.info("corr %s", scipy.stats.spearmanr(all_loss_t[1:]/batch_n, loss_predictions))
        logging.info("weights %s (max %d)", weights, np.argmax(weights))

        if t < nature.total_time - 2:
            proposer.propose_model(sub_trial_data, approval_hist)

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

    nature.next(None)
    model = proposer.propose_model(nature.get_trial_data(0), None)
    if args.human_max_loss is None:
        args.human_max_loss = np.mean(proposer.score_models(nature.create_test_data(0))[0])
        logging.info("HUMAN MAX %f", args.human_max_loss)

    policy = create_policy(
        args.policy_name,
        args,
        human_max_loss=args.human_max_loss,
        num_experts=nature.total_time,
    )

    approval_history = run_simulation(nature, proposer, policy, args.human_max_loss)
    logging.info(approval_history)
    print(approval_history)

    pickle_to_file(approval_history, args.out_file)
    if args.out_nature_file is not None:
        pickle_to_file(nature.to_fixed(), args.out_nature_file)


if __name__ == "__main__":
    main(sys.argv[1:])
