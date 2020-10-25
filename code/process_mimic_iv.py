import os
import subprocess

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score

# items that we want to read
OUTCOME = "survival_status"
ITEM_IDS = [
    220045,  # Heart rate
    220210,  # Respiratory rate
    224690,  # Respiratory rate (total)
    220050,  # Arterial Blood Pressure systolic
    220051,  # Arterial Blood Pressure diastolic
    220052,  # Arterial Blood Pressure mean
    220179,  # Non Invasive Blood Pressure systolic
    220180,  # Non Invasive Blood Pressure diastolic
    220181,  # Non Invasive Blood Pressure mean
    223761,  # Temperature Fahrenheit
    220277,  # Oxygen
    220274,  # PH
    220734,  # PH
    223830,  # PH
    220235,  # CO2
    220645,  # Sodium
    220615,  # Creatinen
    229761,  # Creatine
    225651,  # Bili
    225690,  # Bilirubin
    # 220228,  # Hemoglobin
    # 220615,  # Creatinine (serum)
    # 220739,  # GCS eye
    # 223900,  # GCSVerbal
    # 223901,  # GCS motor
    225664,  # Glucose finger stick
    220621,  # Glucose (serum)
    226537,  # Glucose (whole blood)
    225624,  # BUN
    227456,  # Albumin
    220545,  # Hematocrit
    226540,  # Hematocrit
    220546,  # WBC
    # Urine is missing
]
ITEM_ID_STR = "\|".join([str(itemid) for itemid in ITEM_IDS])

EQUIV_ITEM_IDS = {
    220179: 220050,
    220180: 220051,
    220181: 220052,
    224690: 220210,
    225664: 220621,
    226537: 220621,
    220274: 223830,  # PH
    220734: 223830,  # PH
    229761: 220615,  # Creatinen
    225651: 225690,  # Bilirubin
    226540: 220545,  # Hematocrit
}


def create_features(chartevents_sub, mean_only=False):
    patient_age = chartevents_sub[
        ["subject_id", "hadm_id", "anchor_age"]
    ].drop_duplicates()
    chartvals = chartevents_sub[["subject_id", "hadm_id", "itemid", "valuenum"]]
    print("CAR VALS", chartvals)
    mean_item_values = (
        chartvals.groupby(by=["subject_id", "hadm_id", "itemid"]).mean().reset_index()
    )
    mean_item_values["itemid"] = mean_item_values.itemid.astype(str).str.cat(
        others=["mean"] * mean_item_values.shape[0], sep=""
    )
    mean_item_values = mean_item_values.pivot(
        index=["subject_id", "hadm_id"], columns="itemid", values="valuenum"
    )
    print(mean_item_values)
    max_item_values = (
        chartvals.groupby(by=["subject_id", "hadm_id", "itemid"]).max().reset_index()
    )
    max_item_values["itemid"] = max_item_values.itemid.astype(str).str.cat(
        others=["max"] * max_item_values.shape[0], sep=""
    )
    max_item_values = max_item_values.pivot(
        index=["subject_id", "hadm_id"], columns="itemid", values="valuenum"
    )
    min_item_values = (
        chartvals.groupby(by=["subject_id", "hadm_id", "itemid"]).min().reset_index()
    )
    min_item_values["itemid"] = min_item_values.itemid.astype(str).str.cat(
        others=["min"] * min_item_values.shape[0], sep=""
    )
    min_item_values = min_item_values.pivot(
        index=["subject_id", "hadm_id"], columns="itemid", values="valuenum"
    )
    if mean_only:
        variables = mean_item_values
    else:
        variables = pd.concat(
            [mean_item_values, max_item_values, min_item_values], axis=1
        )
    print("VAR", variables)

    variables = variables.merge(patient_age, on=["subject_id", "hadm_id"])
    print(variables)
    return variables


def impute_mean(variables):
    # TODO: impute better
    # Fill missing with mean
    fill_vals = variables.mean(axis=0, skipna=True)
    for s in ["max", "min", "mean"]:
        fill_vals["220045%s" % s] = 80
    for s in ["max", "min", "mean"]:
        fill_vals["220210%s" % s] = 15
    for s in ["max", "min", "mean"]:
        fill_vals["220050%s" % s] = 100
    for s in ["max", "min", "mean"]:
        fill_vals["220051%s" % s] = 70
    for s in ["max", "min", "mean"]:
        fill_vals["220052%s" % s] = 85
    for s in ["max", "min", "mean"]:
        fill_vals["223761%s" % s] = 98.6
    for s in ["max", "min", "mean"]:
        fill_vals["220277%s" % s] = 95
    for s in ["max", "min", "mean"]:
        fill_vals["223830%s" % s] = 7.4
    for s in ["max", "min", "mean"]:
        fill_vals["220235%s" % s] = 40
    for s in ["max", "min", "mean"]:
        fill_vals["220645%s" % s] = 140
    for s in ["max", "min", "mean"]:
        fill_vals["220615%s" % s] = 1
    for s in ["max", "min", "mean"]:
        fill_vals["225690%s" % s] = 1
    for s in ["max", "min", "mean"]:
        fill_vals["225624%s" % s] = 4
    for s in ["max", "min", "mean"]:
        fill_vals["220621%s" % s] = 100
    for s in ["max", "min", "mean"]:
        fill_vals["227456%s" % s] = 40
    for s in ["max", "min", "mean"]:
        fill_vals["220545%s" % s] = 45
    for s in ["max", "min", "mean"]:
        fill_vals["220546%s" % s] = 9
    # if "220739mean" in fill_vals.index:
    #    fill_vals["220739mean"] = 4
    # else:
    #    print("NO GCS 1")
    # if "223900mean" in fill_vals.index:
    #    fill_vals["223900mean"] = 5
    # else:
    #    print("NO GCS 2")
    # if "223901mean" in fill_vals.index:
    #    fill_vals["223901mean"] = 6
    # else:
    #    print("NO GCS 3")
    print(fill_vals)
    variables = variables.fillna(fill_vals)
    return variables


def extract_patients():
    patients = pd.read_csv("~/mimic_iv/core/patients.csv.gz")
    # Get the approximate year the patient was first treated
    patients[
        ["anchor_year_min", "anchor_year_max"]
    ] = patients.anchor_year_group.str.split(" - ", expand=True)
    patients["anchor_year_mean"] = (
        patients.anchor_year_min.astype(int) + patients.anchor_year_max.astype(int)
    ) / 2
    return patients


def extract_admissions(patients):
    admissions = pd.read_csv("~/mimic_iv/core/admissions.csv.gz")
    admissions = admissions.merge(patients, on="subject_id")
    admissions_dt = pd.to_datetime(admissions.admittime).dt
    admissions["in_year"] = (
        admissions_dt.year - admissions.anchor_year + admissions.anchor_year_mean
    )
    admissions["in_month"] = admissions_dt.month
    admissions["in_quarter"] = (admissions_dt.month / 4).astype(int)
    admissions_deathtime = pd.to_datetime(admissions.deathtime).dt
    admissions["survival_status"] = np.isnan(admissions_deathtime.year)
    return admissions


def extract_stays(thres=2):
    stays = pd.read_csv("~/mimic_iv/icu/icustays.csv.gz")
    stays["long_stay"] = stays.los > thres
    stays["log_los"] = np.log(stays.los)
    return stays


def extract_chartevents(admissions, file_name=None, nrows=1000000):
    if file_name is None:
        chartevents = pd.read_csv(
            "~/mimic_iv/icu/chartevents.csv.gz",
            nrows=nrows,
            usecols=[0, 1, 2, 3, 5, 6, 7],
        )
    else:
        chartevents = pd.read_csv(
            file_name,
            nrows=nrows,
            names="subject_id,hadm_id,stay_id,charttime,storetime,itemid,value,valuenum,valueuom,warning".split(
                ","
            ),
            usecols=[0, 1, 2, 3, 5, 6, 7],
        )
    chartevents = chartevents[chartevents.itemid.isin(ITEM_IDS)]
    for itemid in ITEM_IDS:
        print(itemid, np.sum(chartevents.itemid == itemid))

    chartevents = chartevents.replace({"itemid": EQUIV_ITEM_IDS})
    chartevents = chartevents.merge(
        admissions[["subject_id", "hadm_id", "admittime", "anchor_age"]],
        on=["subject_id", "hadm_id"],
    )
    chartevents["since_admittime"] = pd.to_datetime(
        chartevents.charttime
    ) - pd.to_datetime(chartevents.admittime)
    chartevents["within_24hr"] = chartevents.since_admittime < np.timedelta64(24, "h")
    return chartevents


def extract_labevents(admissions, file_name=None, nrows=1000):
    if file_name is None:
        labevents = pd.read_csv(
            "~/mimic_iv/hosp/labevents.csv.gz", nrows=nrows, usecols=[1, 2, 4, 5, 7, 8]
        )
    else:
        labevents = pd.read_csv(
            file_name,
            nrows=nrows,
            names="labevent_id,subject_id,hadm_id,specimen_id,itemid,charttime,storetime,value,valuenum,valueuom,ref_range_lower,ref_range_upper,flag,priority,comments".split(
                ","
            ),
        )
    labevents = labevents[labevents.itemid.isin(LAB_ITEM_IDS)]
    labevents = labevents.merge(
        admissions[["subject_id", "hadm_id", "admittime"]],
        on=["subject_id", "hadm_id"],
    )
    labevents["since_admittime"] = pd.to_datetime(labevents.charttime) - pd.to_datetime(
        labevents.admittime
    )
    labevents["within_24hr"] = labevents.since_admittime < np.timedelta64(24, "h")
    return labevents


# output = subprocess.check_output(
#   "zgrep '%s' ~/mimic_iv/icu/chartevents.csv.gz > ~/mimic_iv/icu/chartevents_filtered.csv" % ITEM_ID_STR,
#   shell=True)

if not os.path.exists("experiment_mimic/_output/data"):
    os.makedirs("experiment_mimic/_output/data")

patients = extract_patients()
admissions = extract_admissions(patients)
stays = extract_stays(thres=2)

# Filter for patients staying at least one day
stays = stays[stays.los >= 1]
admissions = admissions.merge(stays, on=["subject_id", "hadm_id"])

chartevents = extract_chartevents(
    admissions,
    "~/mimic_iv/icu/chartevents_filtered.csv",
    # max rows 51575974
    nrows=100000000,
)
# Filter for events only within the first 24 hours
chartevents = chartevents[chartevents["within_24hr"]]
# Extract X
chart_features = create_features(chartevents)
features = impute_mean(chart_features)

# features = lab_features.merge(chart_features, on=["subject_id", "hadm_id"])

print(features)
# Merge with Y and year
full_xy_df = (
    (admissions[["subject_id", "hadm_id", "in_year", "in_quarter", OUTCOME]])
    .merge(features, on=["subject_id", "hadm_id"])
    .drop(columns=["subject_id", "hadm_id"])
)
full_xy_df = full_xy_df.sample(frac=1)

MIN_BATCH_SIZE = 30
all_xy_dfs = []
for (in_year, in_quarter), year_df in full_xy_df.groupby(["in_year", "in_quarter"]):
    print(in_year, year_df.shape)
    print(year_df)
    xy = year_df.drop(columns=["in_year", "in_quarter"]).to_numpy()
    print(in_year, in_quarter, xy.shape)
    if xy.shape[0] > MIN_BATCH_SIZE:
        nvalid = max(int(xy.shape[0] / 4), MIN_BATCH_SIZE)
        ntrain = xy.shape[0] - nvalid
        xy_train = xy[:ntrain, :]
        xy_valid = xy[ntrain:, :]
    else:
        ntrain = 0
        xy_train = xy[:0, :]
        xy_valid = xy
    np.savetxt(
        "experiment_mimic/_output/data/train_data_%d_%d.csv" % (in_year, in_quarter),
        xy_train,
    )
    np.savetxt(
        "experiment_mimic/_output/data/valid_data_%d_%d.csv" % (in_year, in_quarter),
        xy_valid,
    )

"""
Just test out a prediction model and see what we get
"""
all_xy = full_xy_df.drop(columns=["in_year", "in_quarter"]).to_numpy()

x = all_xy[:, 1:]
y = all_xy[:, 0].astype(int)
ntrain = int(y.size * 3 / 4)
x_train = x[:ntrain]
y_train = y[:ntrain]
x_test = x[ntrain:]
y_test = y[ntrain:]
print(x.shape, y.shape)

model = RandomForestClassifier(n_estimators=1000, max_depth=3, n_jobs=8)
model.fit(x_train, y_train)
print("SCORE", model.score(x_test, y_test))
predictions = model.predict_proba(x_test)[:, 1]
print("predictions", predictions)
print("AUC", roc_auc_score(y_test, predictions))
# print("mean y", y.mean())
