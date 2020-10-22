"""
Assembles the yelp dataset: splits by year and month, shuffles entries, etc
"""
import sys
import argparse
import logging
import json
import random

from trial_data import TrialDataFromDisk
from nature import FixedNature
from data_generator import *
from support_sim_settings import *
from dataset import Dataset
from common import pickle_to_file


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--year", type=int, default=2008)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--split-ratio", type=float, default=0.5)
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-read", type=int, default=None)
    parser.add_argument(
        "--full-file", type=str, default="data/yelp_academic_dataset_review.json"
    )
    parser.add_argument(
        "--out-train-file", type=str, default="yelp_academic_dataset_review_train.json"
    )
    parser.add_argument(
        "--out-valid-file", type=str, default="yelp_academic_dataset_review_valid.json"
    )
    parser.set_defaults()
    args = parser.parse_args()

    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)
    random.seed(args.seed)
    print(args)

    year_month_lines = []
    with open(args.full_file, "r") as f:
        for line in f:
            line_dict = json.loads(line)
            line_year = int(line_dict["date"].split("-")[0])
            line_month = int(line_dict["date"].split("-")[1])
            if line_year == args.year and line_month == args.month:
                year_month_lines.append(json.dumps(line_dict) + "\n")
                if args.max_read is not None and len(year_month_lines) > args.max_read:
                    break
            else:
                continue

    random.shuffle(year_month_lines)
    num_train = min(args.max_train, int(len(year_month_lines) * args.split_ratio))
    print("year", args.year, "month", args.month, "NUM TRAIN", num_train)
    train_lines = year_month_lines[:num_train]
    valid_lines = year_month_lines[num_train:]
    with open(args.out_train_file, "w") as f:
        f.writelines(train_lines)

    with open(args.out_valid_file, "w") as f:
        f.writelines(valid_lines)


if __name__ == "__main__":
    main(sys.argv[1:])
