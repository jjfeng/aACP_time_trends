"""
Assembles the yelp dataset: splits by year and month, shuffles entries, etc
"""
import json
import numpy as np
import itertools

YEARS = range(2008, 2019)
MONTHS = range(1, 13)
MAX_LINES = 2000
split_ratio = 0.5
valid_year = 5000
path = "/Users/jeanfeng/Downloads/10100_1035793_bundle_archive/yelp_academic_dataset_review.json"
train_lines = {k: [] for k in itertools.product(YEARS, MONTHS)}
valid_lines = {k: [] for k in itertools.product(YEARS, MONTHS)}
with open(path, "r") as f:
    for line in f:
        line_dict = json.loads(line)
        line_year = int(line_dict["date"].split("-")[0])
        line_month = int(line_dict["date"].split("-")[1])
        if line_year in YEARS and line_month in MONTHS:
            key = (line_year, line_month)
            if np.random.rand() < split_ratio:
                if len(train_lines[key]) > MAX_LINES:
                    continue
                line_dict["year"] = line_year
                line_dict["month"] = line_month
                train_lines[key].append(json.dumps(line_dict) + "\n")
            else:
                if len(valid_lines[key]) > MAX_LINES:
                    continue
                line_dict["year"] = valid_year
                line_dict["month"] = line_month
                line_dict["true_year"] = line_year
                valid_lines[key].append(json.dumps(line_dict) + "\n")

        is_done = [
            len(train_lines[key]) > MAX_LINES
            for key in itertools.product(YEARS, MONTHS)
        ] + [
            len(valid_lines[key]) > MAX_LINES
            for key in itertools.product(YEARS, MONTHS)
        ]
        if np.all(is_done):
            print("DONE")
            break

for key in itertools.product(YEARS, MONTHS):
    suffix = (str(key[0]), str(key[1]))
    path_train_new = "data/yelp_academic_dataset_review_year_train_%s_%s.json" % suffix
    path_valid_new = "data/yelp_academic_dataset_review_year_valid_%s_%s.json" % suffix

    with open(path_train_new, "w") as f:
        f.writelines(train_lines[key])

    with open(path_valid_new, "w") as f:
        f.writelines(valid_lines[key])
