import os
import subprocess

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
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
    # 225690,  # Bilirubin
    # 220228,  # Hemoglobin
    # 220615,  # Creatinine (serum)
    220739,  # GCS eye
    223900,  # GCSVerbal
    223901,  # GCS motor
    225664,  # Glucose finger stick
    220621,  # Glucose (serum)
    226537,  # Glucose (whole blood)
]
# LAB_ITEMS = [
#    51221, # hematocrit
#    51222, # hemoglobin
#    51248, # MCH
#    51249, # MCHC
#    51250, # MCV
#    51265, # platelets
#    51279, # RBC
#    51277, # RDW
#    52159, # RDW SD
#    51301  # WBC
# ]
ITEM_ID_STR = "\|".join([str(itemid) for itemid in ITEM_IDS])

EQUIV_ITEM_IDS = {
    220179: 220050,
    220180: 220051,
    220181: 220052,
    224690: 220210,
    225664: 220621,
    226537: 220621,
}


def create_features(chartevents_sub):
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
    variables = pd.concat([mean_item_values, max_item_values, min_item_values], axis=1)
    print("VAR", variables)
    return variables


def impute_mean(variables):
    # Fill missing with mean
    variables = variables.fillna(variables.mean(axis=0, skipna=True))
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
    return stays


def extract_chartevents(file_name = None, nrows=1000000):
    if file_name is None:
        chartevents = pd.read_csv("~/mimic_iv/icu/chartevents.csv.gz", nrows=nrows)
    else:
        chartevents = pd.read_csv(file_name, nrows=nrows,
        names="subject_id,hadm_id,stay_id,charttime,storetime,itemid,value,valuenum,valueuom,warning".split(","))
    chartevents = chartevents[chartevents.itemid.isin(ITEM_IDS)]
    chartevents = chartevents.replace({"itemid": EQUIV_ITEM_IDS})
    chartevents = chartevents.merge(
        admissions[["subject_id", "hadm_id", "admittime"]],
        on=["subject_id", "hadm_id"],
    )
    chartevents["since_admittime"] = pd.to_datetime(
        chartevents.charttime
    ) - pd.to_datetime(chartevents.admittime)
    chartevents["within_24hr"] = chartevents.since_admittime < np.timedelta64(24, "h")
    return chartevents


#output = subprocess.check_output(
#   "zgrep '%s' ~/mimic_iv/icu/chartevents.csv.gz > ~/mimic_iv/icu/chartevents_filtered.csv" % ITEM_ID_STR,
#   shell=True)

if not os.path.exists("experiment_mimic/_output/data"):
    os.makedirs("experiment_mimic/_output/data")

patients = extract_patients()
admissions = extract_admissions(patients)
print(admissions)
stays = extract_stays(thres=2)

# Filter for patients staying at least one day
stays = stays[stays.los >= 1]
admissions = admissions.merge(stays, on=["subject_id", "hadm_id"])

chartevents = extract_chartevents("~/mimic_iv/icu/chartevents_filtered.csv",
        nrows=10000000)
print(chartevents)
# Filter for events only within the first 24 hours
chartevents = chartevents[chartevents["within_24hr"]]
# Extract X
features = create_features(chartevents)
features = impute_mean(features)
print(features)
# Merge with Y and year
full_xy_df = (admissions[["subject_id", "hadm_id", "in_year", "in_quarter", OUTCOME]]).merge(
    features, on=["subject_id", "hadm_id"]
).drop(columns=["subject_id", "hadm_id"])
full_xy_df = full_xy_df.sample(frac=1)

# chartevents.to_csv("~/mimic_iv/test.csv")
# chartevents = pd.read_csv("~/mimic_iv/test.csv")

all_xy_dfs = []
for (in_year, in_quarter), year_df in full_xy_df.groupby(["in_year", "in_quarter"]):
    print(in_year, year_df.shape)
    print(year_df)
    xy = year_df.drop(columns=["in_year", "in_quarter"]).to_numpy()
    ntrain = int(xy.shape[0] * 3 / 4)
    xy_train = xy[:ntrain, :]
    xy_valid = xy[ntrain:, :]
    np.savetxt("experiment_mimic/_output/data/train_data_%d_%d.csv" % (in_year,
        in_quarter), xy_train)
    np.savetxt("experiment_mimic/_output/data/valid_data_%d_%d.csv" % (in_year,
            in_quarter), xy_valid)

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
print(x, y)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
print("SCORE", model.score(x_test, y_test))
predictions = model.predict_proba(x_test)[:, 1]
print("predictions", predictions)
print("AUC", roc_auc_score(y_test, predictions))
print("mean y", y.mean())
