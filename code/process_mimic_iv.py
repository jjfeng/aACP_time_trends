import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# items that we want to read
OUTCOME = "big_los"
ITEM_IDS = [
    220045,  # Heart rate
    220210,  # Respiratory rate
    220050,  # Arterial Blood Pressure systolic
    220051,  # Arterial Blood Pressure diastolic
    220052,  # Arterial Blood Pressure mean
    223761,  # Temperature Fahrenheit
    220277,  # Oxygen
    225690,  # Bilirubin
    220621,  # Glucose
    220228,  # Hemoglobin
    220615,  # Creatinine (serum)
    # 220739, # GCS eye
    # 223900, # GCSVerbal
    # 223901, # GCS motor
    # 227456, # Albumin
    # 227586
]

patients = pd.read_csv("~/mimic_iv/core/patients.csv.gz")
patients[["anchor_year_min", "anchor_year_max"]] = patients.anchor_year_group.str.split(
    " - ", expand=True
)
patients["anchor_year_mean"] = (
    patients.anchor_year_min.astype(int) + patients.anchor_year_max.astype(int)
) / 2
print(patients)

admissions = pd.read_csv("~/mimic_iv/core/admissions.csv.gz")
admissions = admissions.merge(patients, on="subject_id")
admissions_dt = pd.to_datetime(admissions.admittime).dt
admissions["in_year"] = (
    admissions_dt.year - admissions.anchor_year + admissions.anchor_year_mean
)
admissions["in_month"] = admissions_dt.month
admissions_deathtime = pd.to_datetime(admissions.deathtime).dt
print(pd.to_datetime(admissions.deathtime).dt)
admissions["survival_status"] = np.isnan(admissions_deathtime.year)
print(admissions)


stays = pd.read_csv("~/mimic_iv/icu/icustays.csv.gz")
admissions = admissions.merge(stays, on=["subject_id", "hadm_id"])
admissions["big_los"] = admissions.los > 3

chartevents = pd.read_csv("~/mimic_iv/icu/chartevents.csv.gz", nrows=5000000)
# chartevents = pd.read_csv("~/mimic_iv/icu/chartevents_small.csv", nrows=10000)
chartevents = chartevents[chartevents.itemid.isin(ITEM_IDS)]
chartevents = chartevents.merge(
    admissions[["subject_id", "hadm_id", "in_year", "in_month", "admittime", OUTCOME]],
    on=["subject_id", "hadm_id"],
)
chartevents["since_admittime"] = pd.to_datetime(chartevents.charttime) - pd.to_datetime(
    chartevents.admittime
)
chartevents["within_24hr"] = chartevents.since_admittime < np.timedelta64(24, "h")
print(chartevents)
chartevents = chartevents[chartevents["within_24hr"]]

chartevents.to_csv("~/mimic_iv/test.csv")
chartevents = pd.read_csv("~/mimic_iv/test.csv")

all_xy_dfs = []
all_years = []
for in_year, chartevents_sub in chartevents.groupby(["in_year"]):
    chartvals = chartevents_sub[["subject_id", "hadm_id", "itemid", "value"]]
    survs = chartevents_sub[["subject_id", "hadm_id", OUTCOME]].drop_duplicates()
    mean_item_values = (
        chartvals.groupby(by=["subject_id", "hadm_id", "itemid"]).mean().reset_index()
    )
    mean_item_values["itemid"] = mean_item_values.itemid.astype(str).str.cat(
        others=["mean"] * mean_item_values.shape[0], sep=""
    )
    mean_item_values = mean_item_values.pivot(
        index=["subject_id", "hadm_id"], columns="itemid", values="value"
    )
    print(mean_item_values)
    max_item_values = (
        chartvals.groupby(by=["subject_id", "hadm_id", "itemid"]).max().reset_index()
    )
    max_item_values["itemid"] = max_item_values.itemid.astype(str).str.cat(
        others=["max"] * max_item_values.shape[0], sep=""
    )
    max_item_values = max_item_values.pivot(
        index=["subject_id", "hadm_id"], columns="itemid", values="value"
    )
    min_item_values = (
        chartvals.groupby(by=["subject_id", "hadm_id", "itemid"]).min().reset_index()
    )
    min_item_values["itemid"] = min_item_values.itemid.astype(str).str.cat(
        others=["min"] * min_item_values.shape[0], sep=""
    )
    min_item_values = min_item_values.pivot(
        index=["subject_id", "hadm_id"], columns="itemid", values="value"
    )
    variables = pd.concat([mean_item_values, max_item_values, min_item_values])
    xy_df = survs.merge(variables, on=["subject_id", "hadm_id"])
    print(in_year, xy_df.shape)
    all_xy_dfs.append(xy_df)
    all_years.append(in_year)
    # xy_df = xy_df.to_numpy()[:,2:]

    # save year month data
    # np.savetxt("experiment_mimic/_output/data/data_%d.csv" % in_year, xy_df)

"""
Just test out a prediction model and see what we get
"""
all_xy_df = pd.concat(all_xy_dfs)

# FIll with zeros for fun
print("MEAN", all_xy_df.mean(axis=0, skipna=True))
all_xy_df = all_xy_df.fillna(all_xy_df.mean(axis=0, skipna=True))

start_idx = 0
for subgroup_df, in_year in zip(all_xy_dfs, all_years):
    xy_df = all_xy_df.iloc[start_idx : start_idx + subgroup_df.shape[0]].to_numpy()[:,2:]
    ntrain = int(xy_df.shape[0] * 3/4)
    xy_train_df = xy_df[:ntrain,:] 
    xy_valid_df = xy_df[ntrain:,:] 
    start_idx += subgroup_df.shape[0]
    np.savetxt("experiment_mimic/_output/data/train_data_%d.csv" % in_year, xy_train_df)
    np.savetxt("experiment_mimic/_output/data/valid_data_%d.csv" % in_year, xy_valid_df)

# Shuffle
all_xy_df = all_xy_df.sample(frac=1)

x = all_xy_df.to_numpy()[:, 3:]
y = all_xy_df.to_numpy()[:, 2].astype(int)
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
