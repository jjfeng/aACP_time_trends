import json
import numpy as np
import itertools

## Make a dummy 2008 thing
# YEARS = [2008]
# MAX_LINES = 7000
# split_ratio = [0.33, 0.66]
# valid_year = 5000
# path = "/Users/jeanfeng/Downloads/10100_1035793_bundle_archive/yelp_academic_dataset_review.json"
#
# train_lines = {year: [] for year in YEARS}
# valid_lines = {year: [] for year in YEARS}
# test_lines = {year: [] for year in YEARS}
# is_done = [False, False, False]
# with open(path, "r") as f:
#    for line in f:
#        line_dict = json.loads(line)
#        line_year = int(line_dict["date"].split("-")[0])
#        if line_year in YEARS:
#            rval = np.random.rand()
#            if rval < split_ratio[0]:
#                if len(train_lines[line_year]) > MAX_LINES:
#                    is_done[0] = True
#                    continue
#                line_dict["year"] = line_year
#                train_lines[line_year].append(json.dumps(line_dict) + "\n")
#            elif rval < split_ratio[1]:
#                if len(valid_lines[line_year]) > MAX_LINES:
#                    is_done[1] = True
#                    continue
#                line_dict["year"] = valid_year
#                line_dict["true_year"] = valid_year
#                valid_lines[line_year].append(json.dumps(line_dict) + "\n")
#            else:
#                if len(test_lines[line_year]) > MAX_LINES:
#                    is_done[2] = True
#                    continue
#                line_dict["year"] = line_year
#                test_lines[line_year].append(json.dumps(line_dict) + "\n")
#
#        if np.all(is_done):
#            break
#
# for year in YEARS:
#    suffix = str(year)
#    path_train_new = "data/yelp_academic_dataset_review_year_train_%s.json" % suffix
#    path_valid_new = "data/yelp_academic_dataset_review_year_valid_%s.json" % suffix
#    path_test_new = "data/yelp_academic_dataset_review_year_test_%s.json" % suffix
#
#    with open(path_train_new, "w") as f:
#        f.writelines(train_lines[year])
#
#    with open(path_valid_new, "w") as f:
#        f.writelines(valid_lines[year])
#
#    with open(path_test_new, "w") as f:
#        f.writelines(test_lines[year])

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
        if line_year in YEARS:
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
