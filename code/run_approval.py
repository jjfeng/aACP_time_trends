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
from time_trend_predictor import ARIMAPredictor
from nature import Nature
from proposer import Proposer
from approval_history import ApprovalHistory
from policy import *
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
        default="FixedShare",
        choices=["FixedShare", "FixedShareWithBlind", "BlindApproval"],
    )
    parser.add_argument("--eta", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--log-file", type=str, default="_output/log.txt")
    parser.add_argument("--out-file", type=str, default="_output/approver_history.pkl")
    parser.set_defaults()
    args = parser.parse_args()

    return args


def create_policy(policy_name, args, human_max_loss, num_experts):
    time_trend_predictor = ARIMAPredictor(
        order=(2, 1, 0), min_size=7, max_loss=human_max_loss + 0.1
    )
    if policy_name == "OMD":
        policy = OptimisticMirrorDescent(
            num_experts,
            eta=args.eta,
            human_max_loss=human_max_loss,
            time_trend_predictor=time_trend_predictor,
        )
    elif policy_name == "Optimistic":
        policy = OptimisticPolicy(
            num_experts,
            eta=args.eta,
            human_max_loss=human_max_loss,
            time_trend_predictor=time_trend_predictor,
        )
    elif policy_name == "MD":
        policy = MirrorDescent(
            num_experts,
            eta=args.eta,
            human_max_loss=human_max_loss,
        )
    elif policy_name == "FixedShare":
        policy = FixedShare(
            num_experts,
            eta=args.eta,
            alpha=args.alpha,
            human_max_loss=human_max_loss,
        )
    elif policy_name == "FixedShareWithBlind":
        policy = FixedShareWithBlind(
            num_experts,
            eta=args.eta,
            alpha=args.alpha,
            human_max_loss=human_max_loss,
        )
    elif policy_name == "OptimisticFixedShare":
        policy = OptimisticFixedShare(
            num_experts,
            eta=args.eta,
            human_max_loss=human_max_loss,
            time_trend_predictor=time_trend_predictor,
        )
    elif policy_name == "BlindApproval":
        policy = BlindApproval(human_max_loss=human_max_loss)
    return policy


def run_simulation(nature: Nature, proposer: Proposer, policy: Policy, human_max_loss: float):
    approval_hist = ApprovalHistory(human_max_loss=human_max_loss)

    # Create the data generated each batch
    proposer.propose_model(nature.get_trial_data(0), approval_hist)

    # Run the platform trial
    indiv_loss_robot_t = None
    prev_weights = None
    for t in range(nature.total_time - 1):
        print("TIME STEP", t)
        logging.info("TIME STEP %d", t)
        policy.update_weights(t, indiv_loss_robot_t, prev_weights=prev_weights)
        policy.add_expert(t)
        robot_weights, human_weight = policy.get_predict_weights(t)
        #loss_predictions = policy.predict_next_losses(t)
        weights = np.concatenate([[human_weight], robot_weights])

        sub_trial_data = nature.get_trial_data(t + 1)
        indiv_loss_robot_t = proposer.score_models(sub_trial_data.batch_data[-1])
        batch_n = indiv_loss_robot_t.shape[1]
        all_loss_t = np.concatenate(
            [[policy.human_max_loss * batch_n], np.sum(indiv_loss_robot_t, axis=1)]
        )
        policy_loss_t = np.sum(all_loss_t * weights) / batch_n
        approval_hist.append(human_weight, robot_weights, policy_loss_t, all_loss_t)
        prev_weights = weights
        logging.info("losses %s", all_loss_t/batch_n)
        print("losses", all_loss_t/batch_n)
        #logging.info("loss pred %s", loss_predictions)
        #if loss_predictions.size > 2 and np.var(loss_predictions) > 0:
        #    logging.info("corr %s", scipy.stats.spearmanr(all_loss_t[1:]/batch_n, loss_predictions))
        logging.info("weights %s", weights)

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

    model = proposer.propose_model(nature.get_trial_data(0), None, do_append=False)
    human_max_loss = np.mean(model.loss(nature.get_trial_data(1).get_start_to_end_data(1)))
    human_max_loss = min(0.1, 1.25 * human_max_loss)
    print("HUMAN MAX", human_max_loss)

    policy = create_policy(
        args.policy_name,
        args,
        human_max_loss=human_max_loss,
        num_experts=nature.total_time,
    )

    approval_history = run_simulation(nature, proposer, policy, human_max_loss)
    logging.info(approval_history)
    print(approval_history)

    pickle_to_file(approval_history, args.out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
