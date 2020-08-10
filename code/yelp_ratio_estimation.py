import sys
import json
import random
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch import optim
import torchtext
from torchtext import data
from torch.utils.data import Subset

from model import DensityRatioModel, TextSentiment, TextYearModel

from test_yelp import train

# Code from https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/


def train_xy(model, iterator, optimizer, criterion, target_func=None):

    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # set the model in training phase
    model.train()

    for batch in iterator:

        # resets the gradients after every batch
        optimizer.zero_grad()

        # retrieve text and no. of words
        text, text_lengths = batch.text
        star_label = batch.star_label

        # convert to 1D tensor
        predictions = model(text, text_lengths, star_label).squeeze()
        # predictions = model(text, text_lengths).squeeze()

        # compute the loss
        targets = batch.label if target_func is None else target_func(batch.label)

        loss = criterion(predictions, targets)

        # backpropage the loss and compute the gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # loss and accuracy
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate_xy(model, iterator, criterion, target_func=None):
    model.eval()

    losses = 0
    accuracies = 0
    for batch in iterator:
        text, text_lengths = batch.text
        predictions = model(text, text_lengths, batch.star_label).squeeze()
        targets = batch.label if target_func is None else target_func(batch.label)
        loss = criterion(predictions, targets).detach().numpy()
        losses += loss

        label_pred = np.argmax(predictions.detach().numpy(), axis=1).flatten()
        # label_pred = np.around(predictions.detach().numpy()).flatten()
        targets = targets.detach().numpy().flatten()
        acc = np.mean(label_pred == targets)
        accuracies += acc

    print("TEST ACC", accuracies / len(iterator))
    print("LOSS", losses / len(iterator))


def evaluate(model, iterator, criterion, target_func=None):
    model.eval()

    losses = 0
    accuracies = 0
    for batch in iterator:
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze()
        targets = batch.label if target_func is None else target_func(batch.label)
        loss = criterion(predictions, targets).detach().numpy()
        losses += loss

        label_pred = np.argmax(predictions.detach().numpy(), axis=1).flatten()
        # label_pred = np.around(predictions.detach().numpy()).flatten()
        targets = targets.detach().numpy().flatten()
        acc = np.mean(label_pred == targets)
        accuracies += acc

    print("TEST ACC", accuracies / len(iterator))
    print("LOSS", losses / len(iterator))


def estimate_xy_density_ratio(
    YELP_TRAIN,
    fields,
    N_EPOCHS=20,
    num_hidden=30,
    split_ratio=0.7,
    embed_dim=50,
    actual_embed_dim=2,
):
    ####
    # Load data
    ####
    # YELP = "/Users/jeanfeng/Downloads/10100_1035793_bundle_archive/yelp_academic_dataset_review.json"

    SEED = 0
    BATCH_SIZE = 32

    # Load and process data
    train_data = data.TabularDataset(path=YELP_TRAIN, format="json", fields=fields)
    TEXT = fields["text"][1]
    TEXT.build_vocab(train_data, vectors="glove.6B.%dd" % embed_dim)

    # Load model
    # model = DensityRatioModel(vocab_size = len(TEXT.vocab), vocab=TEXT.vocab, embed_dim = EMBED_DIM, num_class=2, num_hidden=num_hidden)
    model = TextYearModel(
        vocab_size=len(TEXT.vocab),
        vocab=TEXT.vocab,
        embed_dim=actual_embed_dim,
        num_class=2,
        num_hidden=num_hidden,
        freeze=True,
    )

    # define optimizer and loss
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train_data, valid_data = train_data.split(split_ratio=split_ratio)
    target_func = lambda x: torch.tensor(x > 2008, dtype=torch.long)

    # Train the model
    train_iterator, valid_iterator = data.Iterator.splits(
        (train_data, valid_data),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        shuffle=True,
    )
    for epoch in range(N_EPOCHS):
        train_loss = train_xy(
            model, train_iterator, optimizer, criterion, target_func=target_func
        )
        print(f"\tTrain Loss: {train_loss:.3f}")

        evaluate_xy(model, valid_iterator, criterion, target_func=target_func)
    return model


def estimate_density_ratio(
    YELP_TRAIN,
    fields,
    N_EPOCHS=20,
    num_hidden=30,
    split_ratio=0.7,
    embed_dim=50,
    actual_embed_dim=2,
):
    ####
    # Load data
    ####
    # YELP = "/Users/jeanfeng/Downloads/10100_1035793_bundle_archive/yelp_academic_dataset_review.json"

    SEED = 0
    BATCH_SIZE = 32

    # Load and process data
    train_data = data.TabularDataset(path=YELP_TRAIN, format="json", fields=fields)
    TEXT = fields["text"][1]
    TEXT.build_vocab(train_data, vectors="glove.6B.%dd" % embed_dim)

    # Load model
    # model = DensityRatioModel(vocab_size = len(TEXT.vocab), vocab=TEXT.vocab, embed_dim = EMBED_DIM, num_class=2, num_hidden=num_hidden)
    model = TextSentiment(
        vocab_size=len(TEXT.vocab),
        vocab=TEXT.vocab,
        embed_dim=actual_embed_dim,
        num_class=2,
        num_hidden=num_hidden,
        freeze=True,
    )

    # define optimizer and loss
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train_data, valid_data = train_data.split(split_ratio=split_ratio)
    target_func = lambda x: torch.tensor(x > 2008, dtype=torch.long)

    # Train the model
    train_iterator, valid_iterator = data.Iterator.splits(
        (train_data, valid_data),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        shuffle=True,
    )
    for epoch in range(N_EPOCHS):
        train_loss = train(
            model, train_iterator, optimizer, criterion, target_func=target_func
        )
        print(f"\tTrain Loss: {train_loss:.3f}")

        evaluate(model, valid_iterator, criterion, target_func=target_func)
    return model


def main(args=sys.argv[1:]):
    ####
    # Load data
    ####
    torch.manual_seed(0)

    YELP_TRAIN = "data/yelp_academic_dataset_review_year_train_2008_2018.json"
    # YELP_TRAIN = "data/yelp_academic_dataset_review_year_2008_2008.json"
    TEXT = data.Field(include_lengths=True, batch_first=True)
    LABEL = data.LabelField(use_vocab=False, dtype=torch.float, batch_first=True)
    fields = {"text": ("text", TEXT), "year": ("label", LABEL)}

    model = estimate_density_ratio(
        YELP_TRAIN, fields, N_EPOCHS=45, num_hidden=10, embed_dim=50
    )


if __name__ == "__main__":
    main(sys.argv[1:])
