#!/usr/bin/env python

"""
Creates model for mimic data for year and month
"""
import sys
import argparse
import logging
import pickle
import pandas as pd
import numpy as np
from numpy import ndarray
from typing import List
from matplotlib import pyplot as plt
import seaborn as sns
from proposer_random_forest import RandomForestWrap, RandomForestRWrap

def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--seed",
        type=int,
        help="Random number generator seed for replicability",
        default=0,
    )
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument("--num-years", type=int, default=11)
    parser.add_argument("--log-file", type=str, default="_output/drift_mimic.txt")
    parser.add_argument("--out-file", type=str, default="_output/drift_mimic.png")
    parser.set_defaults()
    args = parser.parse_args()

    return args


def main(args=sys.argv[1:]):
    MIMIC_TRAIN = "experiment_mimic/_output/data/train_data_%d_%d.csv"

    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    print(args)
    logging.info(args)

    np.random.seed(args.seed)

    quarters = range(4)
    dat0 = np.concatenate(
        [
            np.genfromtxt(MIMIC_TRAIN % (args.start_year, quarter))
            for quarter in quarters
        ]
    )
    oobs_scores = []
    for offset_year in range(1, args.num_years):
        dat1 = np.concatenate(
            [
                np.genfromtxt(MIMIC_TRAIN % (args.start_year + offset_year, quarter))
                for quarter in quarters
            ]
        )
        # Replace labels
        dat0[:,0] = 0
        dat1[:,0] = 1

        # Make new dataset for testing covariate shift
        train_size = min(dat0.shape[0], dat1.shape[0])
        dat = np.concatenate([
            dat0[np.random.choice(dat0.shape[0], size=train_size),:],
            dat1[np.random.choice(dat1.shape[0], size=train_size),:]])
        print([dat0.shape, dat1.shape])

        model = RandomForestWrap(
            n_estimators=100, max_depth=20, oob_score=True, n_jobs=10,
        )
        model.fit(dat[:,1:], dat[:,0])
        logging.info("OOB score %f", model.oob_score_)
        print("OOB", model.oob_score_)
        oobs_scores.append(model.oob_score_)
    scores = pd.DataFrame({
        "OOB": oobs_scores,
        "Year": np.arange(len(oobs_scores)) + args.start_year + 1
    })

    sns.set_context("paper", font_scale=1.5)
    ax = sns.lineplot(
        x="Year",
        y="OOB",
        data=scores,
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig(args.out_file)

if __name__ == "__main__":
    main(sys.argv[1:])
