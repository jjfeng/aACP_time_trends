"""
Plot drift in yelp data
"""
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--file-template", type=str)
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--num-years", type=int, default=11)
    parser.add_argument("--out-file", type=str, default="_output/drift_yelp.png")
    parser.set_defaults()
    args = parser.parse_args()

    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)

    all_stars = []
    for year in range(args.start_year, args.start_year + args.num_years):
        for month in range(1, 13):
            star_file = args.file_template % (year, month)
            star_df = pd.read_csv(star_file)
            star_df["idx"] = 12 * star_df.year + star_df.month
            all_stars.append(star_df)
    all_stars = pd.concat(all_stars)

    sns.set_context("paper", font_scale=1.5)
    ax = sns.lineplot(
        x="idx",
        y="stars",
        data=all_stars,
    )
    plt.ylabel("Average number of stars")
    tick_locs = np.arange(np.min(all_stars.idx), np.max(all_stars.idx), 12)
    plt.xticks(tick_locs, np.arange(args.start_year, args.start_year + tick_locs.size))
    plt.xlabel("Year")
    plt.xticks(rotation='vertical')
    sns.despine()
    plt.tight_layout()
    plt.savefig(args.out_file)

if __name__ == "__main__":
    main(sys.argv[1:])

