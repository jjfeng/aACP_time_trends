import sys
import time
import os
import shutil
import argparse
import numpy as np
import scipy.stats
from numpy import ndarray
from typing import List
import pandas as pd

from nature import Nature
from proposer import FixedProposer
from model_preds_and_targets import ModelPredsAndTargets
from approval_history import ApprovalHistory
from common import pickle_from_file, pickle_to_file


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--max-loss", type=float, default=1)
    parser.add_argument("--nature-file", type=str, default="_output/nature.pkl")
    parser.add_argument("--model-file", type=str, default="_output/model.pkl")
    parser.add_argument(
        "--out-file", type=str, default="_output/model_preds_and_targets.pkl"
    )
    parser.set_defaults()
    args = parser.parse_args()

    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)

    nature = pickle_from_file(args.nature_file)

    approval_hist = ApprovalHistory(human_max_loss=1, policy_name="Placeholder")
    model = pickle_from_file(args.model_file)
    proposer = FixedProposer([model])

    # begin simulation
    # introduce the singleton model
    proposer.propose_model(None, None)
    model_pred_targets = ModelPredsAndTargets()
    for t in range(nature.total_time - 1):
        print("prefetcthing time", t)
        sub_trial_data = nature.get_trial_data(t + 1)
        obs_batch_data = sub_trial_data.batch_data[-1]
        batch_preds, batch_target = proposer.get_model_preds_and_target(obs_batch_data)
        model_pred_targets.append(batch_preds, batch_target)
        nature.next(approval_hist)

    pickle_to_file(model_pred_targets, args.out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
