import sys

import numpy as np
from scipy.special import softmax

import torch
from torch import nn
from torch import optim
import torchtext
from torchtext import data
from torch.utils.data import Subset

from yelp_ratio_estimation import estimate_density_ratio, estimate_xy_density_ratio
from test_yelp import train_rating_model, run_test


def run_shift_check(
    model, dr_model, path, fields, fields_dr, criterion, target_func=None
):
    model.eval()
    dr_model.eval()
    test_data_ratings = data.TabularDataset(path=path, format="json", fields=fields)
    test_data_dr = data.TabularDataset(path=path, format="json", fields=fields_dr)
    test_iterator = data.Iterator(
        test_data_ratings, batch_size=len(test_data_ratings), shuffle=False
    )
    test_iterator_dr = data.Iterator(
        test_data_dr, batch_size=len(test_data_dr), shuffle=False
    )

    for batch in test_iterator_dr:
        text, text_lengths_dr = batch.text

        year_predictions = (
            dr_model(text, text_lengths_dr, batch.star_label).detach().numpy()
        )
        year_predictions = softmax(year_predictions, axis=1)
        density_ratios = year_predictions[:, 1] / year_predictions[:, 0]
        print("density ratio range", np.max(density_ratios), np.min(density_ratios))
        print(
            "EFFECTIVE SAMPLE SIZE",
            np.power(np.sum(density_ratios), 2) / np.sum(np.power(density_ratios, 2)),
        )
        print("ORIG SAMPLE SIZE", density_ratios.size)

    for batch in test_iterator:
        text, text_lengths = batch.text

        predictions = model(text, text_lengths).squeeze()
        targets = batch.label if target_func is None else target_func(batch.label)
        test_losses = criterion(predictions, targets).detach().numpy()

    assert np.all(text_lengths.detach().numpy() == text_lengths_dr.detach().numpy())

    print("DENSITY MEAN", np.mean(density_ratios))
    adjusted_loss = np.mean(test_losses.flatten() * density_ratios.flatten()) / np.mean(
        density_ratios
    )
    print("  ADJUSTED LOSS", adjusted_loss)
    return adjusted_loss


def run_shift_test(
    model, dr_model, path, fields, fields_dr, criterion, target_func=None
):
    model.eval()
    dr_model.eval()
    test_data_ratings = data.TabularDataset(path=path, format="json", fields=fields)
    test_data_dr = data.TabularDataset(path=path, format="json", fields=fields_dr)
    test_iterator = data.Iterator(
        test_data_ratings, batch_size=len(test_data_ratings), shuffle=False
    )
    test_iterator_dr = data.Iterator(
        test_data_dr, batch_size=len(test_data_dr), shuffle=False
    )

    for batch in test_iterator_dr:
        text, text_lengths_dr = batch.text

        # year_predictions = dr_model(text, text_lengths_dr, batch.star_label).detach().numpy()
        year_predictions = dr_model(text, text_lengths_dr).detach().numpy()
        year_predictions = softmax(year_predictions, axis=1)
        density_ratios = year_predictions[:, 1] / year_predictions[:, 0]
        print("density ratio range", np.max(density_ratios), np.min(density_ratios))
        print(
            "EFFECTIVE SAMPLE SIZE",
            np.power(np.sum(density_ratios), 2) / np.sum(np.power(density_ratios, 2)),
        )
        print("ORIG SAMPLE SIZE", density_ratios.size)

    for batch in test_iterator:
        text, text_lengths = batch.text

        predictions = model(text, text_lengths).squeeze()
        targets = batch.label if target_func is None else target_func(batch.label)
        test_losses = criterion(predictions, targets).detach().numpy()

    assert np.all(text_lengths.detach().numpy() == text_lengths_dr.detach().numpy())

    print("DENSITY MEAN", np.mean(density_ratios))
    adjusted_loss = np.mean(test_losses.flatten() * density_ratios.flatten()) / np.mean(
        density_ratios
    )
    print("  ADJUSTED LOSS", adjusted_loss)
    return adjusted_loss


def main(args=sys.argv[1:]):
    N_EPOCHS = 2
    N_EPOCHS_DR = 30
    split_ratio_dr = 0.5
    num_hidden_dr = 5
    num_hidden = 8
    embed_dim = 300
    actual_embed_dim = embed_dim
    orig_year = 2008
    new_year = 2009
    YELP_DENSITY_TRAIN = (
        "data/yelp_academic_dataset_review_year_train_2008_%d.json" % new_year
    )
    YELP_RATING_TRAIN = "data/yelp_academic_dataset_review_year_train_2008.json"
    YELP_TEST = "data/yelp_academic_dataset_review_year_valid_2008.json"
    YELP_ACTUAL_TEST = "data/yelp_academic_dataset_review_year_valid_%d.json" % new_year

    TEXT = data.Field(include_lengths=True, batch_first=True)
    LABEL = data.LabelField(use_vocab=False, dtype=torch.float, batch_first=True)
    fields = {"stars": ("label", LABEL), "text": ("text", TEXT)}
    TEXT1 = data.Field(include_lengths=True, batch_first=True)
    YEAR_LABEL1 = data.LabelField(use_vocab=False, dtype=torch.long, batch_first=True)
    STAR_LABEL1 = data.LabelField(use_vocab=False, dtype=torch.long, batch_first=True)
    fields_dr = {
        "text": ("text", TEXT1),
        "year": ("label", YEAR_LABEL1),
        "stars": ("star_label", LABEL),
    }

    torch.manual_seed(0)
    dr_xy_model = estimate_xy_density_ratio(
        YELP_DENSITY_TRAIN,
        fields_dr,
        N_EPOCHS=N_EPOCHS_DR,
        num_hidden=num_hidden_dr,
        embed_dim=embed_dim,
        split_ratio=split_ratio_dr,
        actual_embed_dim=actual_embed_dim,
    )
    dr_model = estimate_density_ratio(
        YELP_DENSITY_TRAIN,
        fields_dr,
        N_EPOCHS=N_EPOCHS_DR,
        num_hidden=num_hidden_dr,
        embed_dim=embed_dim,
        split_ratio=split_ratio_dr,
        actual_embed_dim=actual_embed_dim,
    )

    rating_criterion = nn.L1Loss()
    torch.manual_seed(0)
    rating_model = train_rating_model(
        YELP_RATING_TRAIN,
        fields,
        rating_criterion,
        N_EPOCHS=N_EPOCHS,
        split_ratio=0.9,
        num_hidden=num_hidden,
        embed_dim=embed_dim,
        actual_embed_dim=actual_embed_dim,
    )

    criterion = nn.L1Loss(reduce=False)
    print("NO SHIFT", orig_year)
    test_losses = run_test(rating_model, YELP_TEST, fields, criterion)
    print(
        "  TEST LOSS %.3f %.3f"
        % (np.mean(test_losses), np.sqrt(np.var(test_losses) / test_losses.size))
    )
    print("TRUE", new_year)
    test_losses = run_test(rating_model, YELP_ACTUAL_TEST, fields, criterion)
    print(
        "  TEST LOSS %.3f %.3f"
        % (np.mean(test_losses), np.sqrt(np.var(test_losses) / test_losses.size))
    )
    print("SHIFT?", new_year)
    run_shift_test(rating_model, dr_model, YELP_TEST, fields, fields_dr, criterion)
    run_shift_check(rating_model, dr_xy_model, YELP_TEST, fields, fields_dr, criterion)


if __name__ == "__main__":
    main(sys.argv[1:])
