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

from model import TextSentiment

# Code from https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/


def train(model, iterator, optimizer, criterion, target_func=None):

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

        # convert to 1D tensor
        predictions = model(text, text_lengths).squeeze()

        # compute the loss
        targets = batch.label if target_func is None else target_func(batch.label)
        # print(batch.label[:5])
        # print(targets[:5])
        # print(np.mean(targets.detach().numpy()))

        loss = criterion(predictions, targets)

        # backpropage the loss and compute the gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # loss and accuracy
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def run_test(model, path, fields, criterion, target_func=None):
    test_data = data.TabularDataset(path=path, format="json", fields=fields)
    test_iterator = data.Iterator(
        test_data,
        batch_size=len(test_data),
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        shuffle=True,
    )

    for batch in test_iterator:
        # retrieve text and no. of words
        text, text_lengths = batch.text

        # convert to 1D tensor
        predictions = model(text, text_lengths).squeeze()

        # compute the loss
        targets = batch.label if target_func is None else target_func(batch.label)
        # targets = targets- 1
        test_loss = criterion(predictions, targets).detach().numpy()

        # label_pred = np.around(predictions.detach().numpy()).flatten()
        ##label_pred = np.argmax(predictions.detach().numpy(), axis=1).flatten()
        # targets = targets.detach().numpy().flatten()
        ##label_pred = predictions.detach().numpy().flatten()
        # acc = np.mean(label_pred == targets)
        # print("TEST ACC", acc)
    return test_loss


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

    print("EVAL LOSS", losses / len(iterator))
    return losses / len(iterator)


def filter_reviews(path, new_path, business_ids):
    reviews = []
    with open(path, "r") as f:
        for line in f:
            review_info = json.loads(line)
            if review_info["business_id"] in business_ids:
                reviews.append(line)
    with open(new_path, "w") as f:
        f.writelines(reviews)


def get_business_ids():
    YELP_BUSINESS = "/Users/jeanfeng/Downloads/10100_1035793_bundle_archive/yelp_academic_dataset_business.json"

    business_ids = []
    with open(YELP_BUSINESS, "r") as f:
        for line in f:
            business_info = json.loads(line)
            if business_info["categories"] is None:
                continue
            is_restaurant = "Restaurant" in business_info["categories"]
            if is_restaurant:
                business_ids.append(business_info["business_id"])
    return set(business_ids)


def train_rating_model(
    YELP_TRAIN,
    fields,
    criterion,
    N_EPOCHS=20,
    split_ratio=0.9,
    num_hidden=30,
    embed_dim=50,
    actual_embed_dim=50,
):
    SEED = 0
    BATCH_SIZE = 16

    # Load and process data
    train_data = data.TabularDataset(path=YELP_TRAIN, format="json", fields=fields)
    print(YELP_TRAIN)
    print("NUM TRAIN", len(train_data.examples))
    assert len(train_data.examples) > 2
    TEXT = fields["text"][1]
    TEXT.build_vocab(train_data, vectors="glove.6B.%dd" % embed_dim)

    # Load model
    model = TextSentiment(
        vocab_size=len(TEXT.vocab),
        vocab=TEXT.vocab,
        embed_dim=actual_embed_dim,
        num_class=1,
        num_hidden=num_hidden,
    )

    # define optimizer and loss
    optimizer = optim.Adam(model.parameters())
    # criterion = nn.CrossEntropyLoss()

    # Train the model
    random.seed(0)
    train_data, valid_data = train_data.split(
        split_ratio=split_ratio, random_state=random.getstate()
    )
    train_iterator, valid_iterator = data.Iterator.splits(
        (train_data, valid_data),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        shuffle=True,
    )
    # iterator = data.Iterator(
    #    train_data,
    #    batch_size = BATCH_SIZE,
    #    sort_key = lambda x: len(x.text),
    #    sort_within_batch=True,
    #    shuffle=True)
    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_iterator, optimizer, criterion)
        if epoch % 5 == 0:
            print(f"\tTrain Loss {epoch}: {train_loss:.3f}")
            evaluate(model, valid_iterator, criterion)

    evaluate(model, valid_iterator, criterion)
    return model


def main(args=sys.argv[1:]):
    ####
    # Load data
    ####
    # YELP = "/Users/jeanfeng/Downloads/10100_1035793_bundle_archive/yelp_academic_dataset_review.json"

    torch.manual_seed(0)
    YELP_TRAIN = "data/yelp_academic_dataset_review_year_train_2008_1.json"

    TEXT = data.Field(include_lengths=True, batch_first=True)
    LABEL = data.LabelField(use_vocab=False, dtype=torch.float, batch_first=True)
    # LABEL = data.LabelField(use_vocab=False, dtype = torch.long, batch_first=True)
    fields = {"stars": ("label", LABEL), "text": ("text", TEXT)}
    criterion = nn.L1Loss()
    # criterion = nn.CrossEntropyLoss()
    model = train_rating_model(YELP_TRAIN, fields, criterion, N_EPOCHS=50)

    # Evaluate the model
    test_losses = []
    for year in range(2008, 2019):
        for month in range(1, 13):
            print("YEAR", year, "month", month)
            yelp_file = "data/yelp_academic_dataset_review_year_valid_%d_%d.json" % (
                year,
                month,
            )
            test_loss = run_test(model, yelp_file, fields, criterion)
            test_losses.append(test_loss)

    sns.lineplot(np.arange(len(test_losses)), np.array(test_losses))
    plt.savefig("_output/test_losses.png")


if __name__ == "__main__":
    main(sys.argv[1:])
