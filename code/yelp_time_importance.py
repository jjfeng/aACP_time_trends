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

from model import TextYearModel, TextSentiment
from test_yelp import train_rating_model, run_test

def train_year(model, iterator, optimizer, criterion, target_func=None, year_func=None):

    #initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    #set the model in training phase
    model.train()

    for batch in iterator:

        #resets the gradients after every batch
        optimizer.zero_grad()

        #retrieve text and no. of words
        text, text_lengths = batch.text
        year_label = batch.year_label if year_func is None else year_func(batch.year_label)

        #convert to 1D tensor
        predictions = model(text, text_lengths, year_label).squeeze()
        #predictions = model(text, text_lengths).squeeze()

        #compute the loss
        targets = batch.label if target_func is None else target_func(batch.label)

        loss = criterion(predictions, targets)

        #backpropage the loss and compute the gradients
        loss.backward()

        #update the weights
        optimizer.step()

        #loss and accuracy
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def train_year_rating_model(YELP_TRAIN, fields, criterion, year_func, N_EPOCHS=20, split_ratio=0.5, num_hidden=30, embed_dim=50, actual_embed_dim=1):
    SEED = 0
    BATCH_SIZE = 16

    # Load and process data
    train_data = data.TabularDataset(path = YELP_TRAIN, format = 'json',fields = fields)
    TEXT = fields["text"][1]
    TEXT.build_vocab(train_data,vectors = "glove.6B.%dd" % embed_dim)

    # Load model
    model = TextYearModel(vocab_size = len(TEXT.vocab), vocab=TEXT.vocab, embed_dim = actual_embed_dim, num_class=1, num_hidden=num_hidden)

    #define optimizer and loss
    optimizer = optim.Adam(model.parameters())
    #criterion = nn.CrossEntropyLoss()

    # Train the model
    random.seed(0)
    train_data, valid_data = train_data.split(split_ratio=split_ratio, random_state=random.getstate())
    train_iterator, valid_iterator = data.Iterator.splits(
        (train_data, valid_data),
        batch_size = BATCH_SIZE * 2,
        sort_key = lambda x: len(x.text),
        sort_within_batch=True,
        shuffle=True)
    for epoch in range(N_EPOCHS):
        train_loss = train_year(model, train_iterator, optimizer, criterion, year_func=year_func)
        print(f'\tTrain Loss: {train_loss:.3f}')

    evaluate_year(model, valid_iterator, criterion, year_func=year_func)
    return model

def evaluate_year(model, iterator, criterion, target_func=None, year_func=None):
    model.eval()

    losses = 0
    accuracies = 0
    for batch in iterator:
        text, text_lengths = batch.text
        year_label = batch.year_label if year_func is None else year_func(batch.year_label)
        predictions = model(text, text_lengths, year_label).squeeze()
        #predictions = model(text, text_lengths).squeeze()
        targets = batch.label if target_func is None else target_func(batch.label)
        loss = criterion(predictions, targets).detach().numpy()
        losses += loss

    print("EVAL LOSS", losses/len(iterator))
    return losses/len(iterator)

def run_hypothesis_test(model0, model_year, path, fields0, fields_year, criterion, target_func=None, year_func=None):
    test_data0 = data.TabularDataset(path = path, format = 'json', fields = fields0)
    test_data_year = data.TabularDataset(path = path, format = 'json', fields = fields_year)
    test_iterator0 = data.Iterator(
        test_data0,
        batch_size = len(test_data0),
        shuffle=False)
    test_iterator_year = data.Iterator(
        test_data_year,
        batch_size = len(test_data_year),
        shuffle=False)

    for batch in test_iterator0:
        text, text_lengths0 = batch.text

        predictions = model0(text, text_lengths0).squeeze()
        targets = batch.label if target_func is None else target_func(batch.label)
        loss0 = criterion(predictions, targets)

    for batch in test_iterator_year:
        text, text_lengths = batch.text
        year_label = batch.year_label if year_func is None else year_func(batch.year_label)

        predictions = model_year(text, text_lengths, year_label).squeeze()
        targets = batch.label if target_func is None else target_func(batch.label)
        loss_year = criterion(predictions, targets)

    assert np.all(text_lengths.detach().numpy() == text_lengths0.detach().numpy())

    diffs = (loss_year - loss0).detach().numpy()
    average_loss = np.mean(diffs)
    average_se = np.sqrt(np.var(diffs)/diffs.size)
    print("AVG DIFF", average_loss, average_se)
    return average_loss, average_se

def main(args=sys.argv[1:]):
    split_ratio = 0.7
    n_epochs = 10
    embed_dim = 50
    actual_embed_dim = 1
    num_hidden = 8
    years = [2008, 2018]
    YELP_TRAIN_YEAR0 = "data/yelp_academic_dataset_review_year_train_%d.json" % years[0]
    YELP_TRAIN_YEAR1 = "data/yelp_academic_dataset_review_year_train_%d.json" % years[1]
    YELP_TRAIN_YEARS = "data/yelp_academic_dataset_review_year_train_%d_%d.json" % tuple(years)

    print("YEAR DEPENDENT MODEL", years)
    criterion = nn.L1Loss()
    TEXT = data.Field(include_lengths=True, batch_first=True)
    LABEL = data.LabelField(use_vocab=False, dtype = torch.float, batch_first=True)
    YEAR_LABEL = data.LabelField(use_vocab=False, dtype = torch.long, batch_first=True)
    fields_year = {"year": ('year_label',YEAR_LABEL),"stars": ('label',LABEL), "text": ('text', TEXT)}
    year_func = lambda x: torch.tensor(x > years[0], dtype=torch.float)
    torch.manual_seed(0)
    model_year = train_year_rating_model(YELP_TRAIN_YEARS, fields_year, criterion, year_func, N_EPOCHS = n_epochs, split_ratio=split_ratio, num_hidden=num_hidden, embed_dim=embed_dim, actual_embed_dim=actual_embed_dim)

    print("YEAR BLIND MODEL")
    TEXT_blind = data.Field(include_lengths=True, batch_first=True)
    LABEL_blind = data.LabelField(use_vocab=False, dtype = torch.float, batch_first=True)
    fields_blind = {"stars": ('label',LABEL), "text": ('text', TEXT)}
    torch.manual_seed(0)
    model_blind = train_rating_model(YELP_TRAIN_YEARS, fields_blind, criterion, N_EPOCHS=n_epochs, split_ratio=split_ratio, num_hidden=num_hidden, embed_dim=embed_dim, actual_embed_dim=actual_embed_dim)

    #print("YEAR 0 MODEL", years[0])
    #TEXT0 = data.Field(include_lengths=True, batch_first=True)
    #LABEL0 = data.LabelField(use_vocab=False, dtype = torch.float, batch_first=True)
    #fields0 = {"stars": ('label',LABEL), "text": ('text', TEXT)}
    #torch.manual_seed(0)
    #model0 = train_rating_model(YELP_TRAIN_YEAR0, fields0, criterion, N_EPOCHS=n_epochs, split_ratio=0.9, num_hidden=num_hidden, embed_dim=embed_dim, actual_embed_dim=actual_embed_dim)

    #print("YEAR 1 MODEL", years[1])
    #TEXT1 = data.Field(include_lengths=True, batch_first=True)
    #LABEL1 = data.LabelField(use_vocab=False, dtype = torch.float, batch_first=True)
    #fields1 = {"stars": ('label',LABEL), "text": ('text', TEXT)}
    #torch.manual_seed(0)
    #model1 = train_rating_model(YELP_TRAIN_YEAR1, fields1, criterion, N_EPOCHS=n_epochs, split_ratio=0.9, num_hidden=num_hidden, embed_dim=embed_dim, actual_embed_dim=actual_embed_dim)

    # Evaluate the model
    criterion = nn.L1Loss(reduce=False)
    YELP_TEST = "data/yelp_academic_dataset_review_year_valid_%d.json" % years[1]
    year_func = lambda x: x/x
    #print("TRAIN ONLY YEAR 0")
    #test_loss = run_hypothesis_test(model0, model_year, YELP_TEST, fields0, fields_year, criterion, year_func=year_func)
    #print("TRAIN ONLY YEAR 1")
    #test_loss = run_hypothesis_test(model1, model_year, YELP_TEST, fields1, fields_year, criterion, year_func=year_func)
    print("TRAIN BOTH YEARS")
    test_loss = run_hypothesis_test(model_blind, model_year, YELP_TEST, fields_blind, fields_year, criterion, year_func=year_func)

if __name__ == "__main__":
    main(sys.argv[1:])
