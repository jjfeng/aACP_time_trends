import sys
import json
import random
import numpy as np
from scipy.stats import ttest_ind

import seaborn as sns
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch import optim
import torchtext
from torchtext import data
from torch.utils.data import Subset

from test_yelp import train_rating_model


def run_star_test(model, paths, fields, criterion, star, target_func=None):
    print("STARS", star)
    test_losses = []
    for idx, path in enumerate(paths):
        test_data = data.TabularDataset(path=path, format="json", fields=fields)
        test_iterator = data.Iterator(
            test_data,
            batch_size=len(test_data),
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            shuffle=False,
        )

        for batch in test_iterator:
            # retrieve text and no. of words
            text, text_lengths = batch.text

            # convert to 1D tensor
            predictions = model(text, text_lengths).squeeze()

            # compute the loss
            targets = batch.label if target_func is None else target_func(batch.label)
            mask = batch.label.detach().numpy() == star

            proportion = np.mean(mask)
            print("PROPORTION", idx, proportion)

            test_loss = criterion(predictions, targets).detach().numpy()
            test_losses.append(test_loss[mask])
            print("TEST LOSS", idx, "MEAN", test_loss[mask].mean())

    ttest_res = ttest_ind(test_losses[0], test_losses[1], equal_var=False)
    print(ttest_res)


def main(args=sys.argv[1:]):
    ####
    # Load data
    ####
    torch.manual_seed(0)
    YEARS = [2008, 2009]
    STARS = range(1, 6)
    YELP_TRAIN = "data/yelp_academic_dataset_review_year_train_%d.json" % YEARS[0]
    YELP_TESTS = [
        "data/yelp_academic_dataset_review_year_valid_%d.json" % year for year in YEARS
    ]

    TEXT = data.Field(include_lengths=True, batch_first=True)
    LABEL = data.LabelField(use_vocab=False, dtype=torch.float, batch_first=True)
    fields = {"stars": ("label", LABEL), "text": ("text", TEXT)}
    criterion = nn.L1Loss()
    model = train_rating_model(YELP_TRAIN, fields, criterion, N_EPOCHS=1)

    # Evaluate the model
    criterion = nn.L1Loss(reduce=False)
    for star in STARS:
        run_star_test(model, YELP_TESTS, fields, criterion, star=star)


if __name__ == "__main__":
    main(sys.argv[1:])
