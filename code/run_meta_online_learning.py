import os
import json
import sys
import itertools
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch import optim
import torchtext
from torchtext import data

from model import TextSentiment
from mixture_experts import BlindWeight
from mixture_experts import TimeTrendForecaster
from mixture_experts import MetaExpWeighting
from mixture_experts import ExpWeightingWithHuman
from test_yelp import train_rating_model, run_test

def train_rating_model_year_month(path, n_epochs, num_hidden=5):
    TEXT = data.Field(include_lengths=True, batch_first=True)
    LABEL = data.LabelField(use_vocab=False, dtype = torch.float, batch_first=True)
    fields = {"stars": ('label',LABEL), "text": ('text', TEXT)}
    criterion = nn.L1Loss()
    model = train_rating_model(path, fields, criterion, N_EPOCHS = n_epochs, split_ratio=0.9, num_hidden=num_hidden)
    return model, fields

def run_meta_forecaster_simulation(histories, path_func, times, meta_forecaster):
    MONTHS = range(1,13)
    forecaster_keys = meta_forecaster.forecaster_keys
    print(forecaster_keys)

    tot_loss = 0
    indiv_loss_t = None
    prev_weights = None
    loss_history = []
    expert_history = []
    for t, time_key in enumerate(times):
        print("=============", t)
        meta_forecaster.update_weights(t, indiv_loss_t, prev_weights=prev_weights)
        weights = meta_forecaster.get_predict_weights(t)
        print("meta weights", weights)

        indiv_loss_t = np.array([histories[k]["loss_history"][t] for k in forecaster_keys]).reshape((-1,1))
        expert_t = np.array([histories[k]["human_history"][t] for k in forecaster_keys])
        print("expert", expert_t)
        loss_t = np.sum(indiv_loss_t, axis=1)
        print("LOSSES", loss_t)
        step_loss = np.sum(loss_t * weights)
        print("step loss", step_loss)
        tot_loss += step_loss
        prev_weights = weights
        loss_history.append(step_loss)
        expert_history.append(np.sum(expert_t * weights))

    print("FINAL mean LOSS", tot_loss/len(times))
    print("FINAL WEI", weights)
    return np.array(loss_history), np.array(expert_history)

def run_forecaster_simulation(models, path_func, times, forecaster, human_cost=0):
    MONTHS = range(1,13)
    criterion = nn.L1Loss(reduce=False)

    indiv_loss_robot_t = None
    prev_weights = None
    loss_history = []
    expert_history = []
    for t, time_key in enumerate(times):
        curr_models = models[:t + 1]
        print("=============", t)
        forecaster.update_weights(t, indiv_loss_robot_t, prev_weights=prev_weights)
        forecaster.add_expert(t)
        robot_weights, human_weight = forecaster.get_predict_weights(t)
        print("rob", robot_weights, "hu", human_weight)
        weights = np.concatenate([[human_weight], robot_weights])

        path_time = path_func(time_key)
        #print("PATH", path_time)
        indiv_loss_robot_t = np.array([
                run_test(model['model'], path_time, fields=model['fields'], criterion=criterion) for model in curr_models])
        batch_n = indiv_loss_robot_t.shape[1]
        if np.sum(robot_weights) < 1e-10:
            #print("HUMAN")
            loss_t = human_cost * batch_n
            expert_history.append(1)
        else:
            assert np.isclose(np.sum(robot_weights) + human_weight, 1)
            #print("ROBOT", robot_weights)
            loss_t = np.concatenate([[batch_n * human_cost], np.sum(indiv_loss_robot_t, axis=1)])
            expert_history.append(human_weight)
        #print("LOSSES", np.mean(indiv_loss_robot_t, axis=1))
        step_loss = np.sum(loss_t * weights)/batch_n
        print("step loss", step_loss)
        prev_weights = weights
        loss_history.append(step_loss)

    print("FINAL mean LOSS", np.mean(loss_history))
    print("percent human", np.sum(expert_history)/len(times))
    print("FINAL WEI", weights)
    return np.array(loss_history), np.array(expert_history)

def plot_histories(loss_history, human_history, fig_name_loss, fig_name_human, alpha):
    T = loss_history.size + 1
    plt.figure(figsize=(6,6))
    plt.plot(np.arange(T - 1), np.cumsum(loss_history)/np.arange(1,T))
    plt.ylim(0.8, 1.0)
    plt.ylabel("Cumulative loss")
    plt.xlabel("Time")
    plt.hlines(y=alpha, xmin=0, xmax=T - 1)
    plt.savefig(fig_name_loss)

    plt.clf()
    plt.figure(figsize=(6,6))
    plt.plot(np.arange(T - 1), np.cumsum(human_history)/np.arange(1,T))
    plt.ylim(0.0, 1.0)
    plt.ylabel("Human prob")
    plt.xlabel("Time")
    plt.hlines(y=alpha, xmin=0, xmax=T - 1)
    plt.savefig(fig_name_human)

def main(args=sys.argv[1:]):
    torch.manual_seed(0)
    YELP_TRAIN = "data/yelp_academic_dataset_review_year_train_%s_%s.json"
    YELP_TEST = "data/yelp_academic_dataset_review_year_valid_%s_%s.json"
    OUT_TEMPLATE = "_output/model_%d_%d.pth"
    OUT_FORECASTER_TEMPLATE = "_output/history_%s_year_%d.json"

    num_hidden = 10
    N_EPOCHS = 40
    YEARS = range(2008,2011)
    MONTHS = range(1,13)
    #N_EPOCHS = 4
    #YEARS = range(2008,2009)
    #MONTHS = range(1,13)

    times = []
    models = []
    for year in YEARS:
        for month in MONTHS:
            out_model_file = OUT_TEMPLATE % (year, month)
            if os.path.exists(out_model_file):
                model_dict = torch.load(out_model_file)
                TEXT = model_dict["fields"]["text"][1]
                model = TextSentiment(vocab_size = len(TEXT.vocab), vocab=TEXT.vocab, embed_dim = 50, num_class=1, num_hidden=num_hidden)
                model.load_state_dict(model_dict["state_dict"])
                fields = model_dict["fields"]
            else:
                model, fields = train_rating_model_year_month(YELP_TRAIN % (str(year), str(month)), n_epochs=N_EPOCHS, num_hidden=num_hidden)
                # Do save
                model_state_dict = {"state_dict": model.state_dict(), "fields": fields, "year": year, "month": month}
                torch.save(model_state_dict, out_model_file)

            model.eval()
            model_dict = {"model": model, "fields": fields, "year": year, "month": month}
            models.append(model_dict)

            if not (year == YEARS[0] and month == MONTHS[0]):
                times.append((year, month))

    T = len(models)
    human_cost = 0.9
    alpha = 0.9
    ETA_FACTOR = 0.1
    path_func = lambda x: YELP_TEST % x
    forecasters = [
            ExpWeightingWithHuman(T, human_max_loss=alpha, eta_factor=0.1, new_model_eta=0.3),
            TimeTrendForecaster(human_max_loss=alpha),
            BlindWeight(),
    ]
    meta_forecaster = MetaExpWeighting(T, eta=0.1, num_experts=len(forecasters), forecaster_keys=[str(forecaster) for forecaster in forecasters])
    histories = {}
    for forecaster in forecasters:
        out_forecaster_file = OUT_FORECASTER_TEMPLATE % (str(forecaster), max(YEARS))
        print("forecaster file", out_forecaster_file)
        if os.path.exists(out_forecaster_file):
            history = json.load(open(out_forecaster_file, "r"))
        else:
            loss_history, human_history = run_forecaster_simulation(
                models,
                path_func,
                times,
                forecaster=forecaster,
                human_cost=human_cost)
            fig_name_loss = "_output/yelp_loss_%s.png" % str(forecaster)
            fig_name_human = "_output/yelp_human_%s.png" % str(forecaster)
            plot_histories(loss_history, human_history, fig_name_loss, fig_name_human, alpha)
            history = {
                "loss_history": loss_history.tolist(),
                "human_history": human_history.tolist()}
            # save history
            with open(out_forecaster_file, "w") as f:
                json.dump(history, f)
        histories[str(forecaster)] = history
        for k in history.keys():
            history[k] = np.array(history[k])

    loss_history, human_history = run_meta_forecaster_simulation(
            histories,
            path_func,
            times,
            meta_forecaster=meta_forecaster)
    print("LOSS", loss_history)
    print("HUMAN", human_history)
    fig_name_loss = "_output/meta_loss.png"
    fig_name_human = "_output/meta_human.png"
    plot_histories(loss_history, human_history, fig_name_loss, fig_name_human, alpha)

if __name__ == "__main__":
    main(sys.argv[1:])

