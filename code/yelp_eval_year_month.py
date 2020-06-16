import os
import sys
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


import torch
from torch import nn
from torch import optim
import torchtext
from torchtext import data

from mixture_experts import TimeTrendForecaster
from yelp_online_learning import *

def plot_time_trends(time_res, fig_name):
    plt.figure(figsize=(6,6))
    plt.plot(time_res.time, time_res["losses"])
    plt.plot(time_res.time, time_res["arima_output"])
    plt.ylabel("Loss")
    plt.xlabel("Time")
    plt.savefig(fig_name)

def main(args=sys.argv[1:]):
    torch.manual_seed(0)
    YELP_TRAIN = "data/yelp_academic_dataset_review_year_train_%s_%s.json"
    YELP_TEST = "data/yelp_academic_dataset_review_year_valid_%s_%s.json"
    OUT_TEMPLATE = "_output/model_%d_%d.pth"

    num_hidden = 10
    N_EPOCHS = 40
    year = 2008
    month = 1

    out_model_file = OUT_TEMPLATE % (year, month)
    if not os.path.exists(out_model_file):
        model, fields = train_rating_model_year_month(YELP_TRAIN % (str(year), str(month)), n_epochs=N_EPOCHS, num_hidden=num_hidden)
        # Do save
        model_state_dict = {"state_dict": model.state_dict(), "fields": fields, "year": year, "month": month}
        torch.save(model_state_dict, out_model_file)

    model_dict = torch.load(out_model_file)
    TEXT = model_dict["fields"]["text"][1]
    model = TextSentiment(vocab_size = len(TEXT.vocab), vocab=TEXT.vocab, embed_dim = 50, num_class=1, num_hidden=num_hidden)
    model.load_state_dict(model_dict["state_dict"])
    fields = model_dict["fields"]

    model.eval()
    model_dict = {"model": model, "fields": fields, "year": year, "month": month}

    YEARS = range(2008,2019)
    MONTHS = range(1,13)
    min_size = 7
    path_func = lambda x: YELP_TEST % x

    forecaster = TimeTrendForecaster(1, order=(2,1,0), min_size=min_size)

    time_res = {"losses": [], "arima_output": [], "time": []}
    criterion = nn.L1Loss()
    for t_idx, t in enumerate(itertools.product(YEARS, MONTHS)):
        print(t_idx, t)
        time_res["time"].append(t_idx)

        if t_idx > forecaster.min_size:
            arima_output = forecaster.fit_arima_get_output(np.array(time_res["losses"]).flatten())
        else:
            arima_output = None
        time_res["arima_output"].append(arima_output)

        path_time = path_func((str(t[0]), str(t[1])))
        loss_t = run_test(model_dict['model'], path_time, fields=model_dict['fields'], criterion=criterion)
        time_res["losses"].append(loss_t)

    # Plot time trends
    fig_name = "_output/yelp_loss_%d_%d.png" % (year, month)
    plot_time_trends(pd.DataFrame(time_res), fig_name)
    print(fig_name)

if __name__ == "__main__":
    main(sys.argv[1:])

