import os
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
from mixture_experts import TimeTrendForecaster
from mixture_experts import ExpWeightingWithHuman, BlindWeight, OraclePredictor
from test_yelp import train_rating_model, run_test

def train_rating_model_year_month(path, n_epochs, num_hidden=5):
    TEXT = data.Field(include_lengths=True, batch_first=True)
    LABEL = data.LabelField(use_vocab=False, dtype = torch.float, batch_first=True)
    fields = {"stars": ('label',LABEL), "text": ('text', TEXT)}
    criterion = nn.L1Loss()
    model = train_rating_model(path, fields, criterion, N_EPOCHS = n_epochs, split_ratio=0.9, num_hidden=num_hidden)
    return model, fields

def run_simulation(models, path_func, times, forecaster, is_oracle=False, human_max_loss=0):
    MONTHS = range(1,13)
    criterion = nn.L1Loss(reduce=False)

    tot_loss = 0
    indiv_loss_robot_t = None
    prev_weights = None
    loss_history = []
    expert_history = []
    was_human_round = False
    total_n = 0
    for t, time_key in enumerate(times):
        curr_models = models[:t + 1]
        print("=============", t)
        forecaster.update_weights(t, indiv_loss_robot_t, prev_weights=prev_weights)
        forecaster.add_expert(t)
        robot_weights, human_weight = forecaster.get_predict_weights(t)
        print("rob", robot_weights, "hu", human_weight)
        weights = np.concatenate([[human_weight], robot_weights])

        path_time = path_func(time_key)
        print("PATH", path_time)
        indiv_loss_robot_t = np.array([
                run_test(model['model'], path_time, fields=model['fields'], criterion=criterion) for model in curr_models])
        batch_n = indiv_loss_robot_t.shape[1]
        total_n += batch_n
        if np.sum(robot_weights) < 1e-10:
            print("HUMAN")
            loss_t = human_max_loss
            expert_history.append(1)
        # print("WIEHGT", forecaster.weights)
        else:
            assert np.isclose(np.sum(robot_weights) + human_weight, 1)
            print("ROBOT", robot_weights)
            loss_t = np.concatenate([[human_max_loss * batch_n], np.sum(indiv_loss_robot_t, axis=1)])
            expert_history.append(human_weight)
        print("LOSSES", np.mean(indiv_loss_robot_t, axis=1))
        tot_loss += np.sum(loss_t * weights)
        # print("round loss", round_loss)
        # print("cum mean LOSS", tot_loss/(batch_n * (t + 1)))
        prev_weights = weights
        loss_history.append(tot_loss/((t + 1) * batch_n))

    print("FINAL mean LOSS", tot_loss/total_n)
    print("percent human", np.sum(expert_history)/len(times))
    print("FINAL WEI", weights)
    return np.array(loss_history), np.array(expert_history)

def main(args=sys.argv[1:]):
    torch.manual_seed(0)
    YELP_TRAIN = "data/yelp_academic_dataset_review_year_train_%s_%s.json"
    YELP_TEST = "data/yelp_academic_dataset_review_year_valid_%s_%s.json"
    OUT_TEMPLATE = "_output/model_%d_%d.pth"

    num_hidden = 10
    N_EPOCHS = 40
    YEARS = range(2008,2009)
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
    alpha = 0.9
    ETA_FACTOR = 0.1
    path_func = lambda x: YELP_TEST % x
    forecasters = [
            #ExpWeightingWithHuman(T, human_max_loss=alpha, eta_factor=0.1, new_model_eta=0.1),
            #BlindWeight(),
            #OraclePredictor([path_func(t) for t in times], models, human_max_loss=alpha)
            TimeTrendForecaster(num_experts=len(models), eta=0.1,  human_max_loss=alpha)
    ]
    for forecaster in forecasters:
        loss_history, human_history = run_simulation(models, path_func, times, forecaster=forecaster, human_max_loss=alpha)
        print("PROB HUMAN HIST", human_history)

        plt.figure(figsize=(6,6))
        plt.plot(np.arange(T - 1), loss_history)
        plt.ylabel("Cumulative loss")
        plt.xlabel("Time")
        plt.hlines(y=alpha, xmin=0, xmax=T)
        fig_name = "_output/yelp_mixture_%s.png" % str(forecaster)
        print(fig_name)
        plt.savefig(fig_name)

        plt.clf()
        plt.figure(figsize=(6,6))
        plt.plot(np.arange(T - 1), np.cumsum(human_history)/np.arange(1,T))
        plt.ylabel("Human prob")
        plt.xlabel("Time")
        plt.hlines(y=alpha, xmin=0, xmax=T)
        fig_name = "_output/yelp_human_%s.png" % str(forecaster)
        print(fig_name)
        plt.savefig(fig_name)

if __name__ == "__main__":
    main(sys.argv[1:])

