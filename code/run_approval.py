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
    parser.add_argument("--human-max-loss", type=float, default=0.9)
    parser.add_argument("--nature-file", type=str, default="_output/nature.pkl")
    parser.add_argument("--proposer-file", type=str, default="_output/proposer.pkl")
    parser.add_argument(
        "--policy-name",
        type=str,
        help="name of approval policy",
        default="FixedShare",
        #choices=["FixedShare", "FixedShareWithBlind", "BlindApproval", "TTestApproval"],
    )
    parser.add_argument("--eta", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--log-file", type=str, default="_output/log.txt")
    parser.add_argument("--out-file", type=str, default="_output/approver_history.pkl")
    parser.set_defaults()
    args = parser.parse_args()

    return args


def create_policy(policy_name, args, human_max_loss, num_experts):
    if policy_name == "MarkovHedge":
        policy = ValidationPolicy(
            num_experts=num_experts,
            etas=np.array([args.eta,0, args.alpha, 0.05]),
            human_max_loss=human_max_loss,
            const_baseline_weight=1,
        )
    elif policy_name == "MetaExpWeighting":
        policy = MetaExpWeighting(
            eta=args.eta,
            eta_grid=[
                np.array([0,20]), # emp loss
                np.array([0,0.2,0.8,1]), # scaling
                #np.array([50]), # emp loss
                #np.array([1]), # scaling
                np.array([0,0.05,0.1,0.5,1]), # alpha
                np.array([0.05]) # baseline alpha
            ],
            num_experts=num_experts,
            human_max_loss=human_max_loss,
        )
    elif policy_name == "BaselinePolicy":
        policy = BaselinePolicy(
            human_max_loss=human_max_loss)
    elif policy_name == "BlindApproval":
        policy = BlindApproval(human_max_loss=human_max_loss)
    elif policy_name == "MeanApproval":
        policy = MeanApproval(num_experts, human_max_loss=human_max_loss)
    elif policy_name == "TTestApproval":
        policy = TTestApproval(num_experts, human_max_loss=human_max_loss)
    else:
        raise ValueError("approval not found")
    return policy


def run_simulation(nature: Nature, proposer: Proposer, policy: Policy, human_max_loss: float):
    approval_hist = ApprovalHistory(human_max_loss=human_max_loss)

    # Create the data generated each batch
    proposer.propose_model(nature.get_trial_data(0), approval_hist)
    #nature.next(approval_hist)

    # Run the platform trial
    indiv_loss_robot_t = None
    prev_weights = None
    for t in range(nature.total_time - 1):
        logging.info("TIME STEP %d", t)
        policy.add_expert(t)
        policy.update_weights(t, indiv_loss_robot_t, prev_weights=prev_weights)
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

        nature.next(approval_hist)

        prev_weights = weights
        logging.info("losses %s", all_loss_t/batch_n)
        #print("losses", all_loss_t/batch_n)
        #logging.info("loss pred %s", loss_predictions)
        #if loss_predictions.size > 2 and np.var(loss_predictions) > 0:
        #    logging.info("corr %s", scipy.stats.spearmanr(all_loss_t[1:]/batch_n, loss_predictions))
        logging.info("weights %s (max %d)", weights, np.argmax(weights))

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
    model = proposer.propose_model(nature.get_trial_data(0), None, do_append=False)
    time_0_test_data = nature.data_gen.create_data(1000, 9, nature.coefs[0])
    human_max_loss = np.mean(model.loss(time_0_test_data))
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
