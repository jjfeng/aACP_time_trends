from typing import List, Dict
import numpy as np

import torch
from torch import nn
from torchtext import data
from model import TextSentiment

from dataset import Dataset
from trial_data import TrialData
from approval_history import ApprovalHistory


class Proposer:
    def __init__(self):
        self.proposal_history = []

    def criterion(self, pred_y, y):
        return self.proposal_history[0]._criterion(pred_y, y)

    @property
    def num_models(self):
        return len(self.proposal_history)

    def propose_model(self, trial_data: TrialData):
        raise NotImplementedError()

    def score_models(self, dataset: Dataset):
        predictions, target = self.get_model_preds_and_target(dataset)
        return np.array(
            [
                self.criterion(predictions[i, :], target)
                for i in range(predictions.shape[0])
            ]
        )

    def get_model_preds_and_target(self, dataset: Dataset):
        """
        Get preds and target from the models
        """
        predictions = np.array(
            [model.predict(dataset.x) for model in self.proposal_history]
        )
        return predictions, dataset.y


class FixedProposer(Proposer):
    def __init__(self, models: List):
        self.pretrained_proposal_history = models
        self.proposal_history = []

    def criterion(self, pred_y, y):
        return self.pretrained_proposal_history[0]._criterion(pred_y, y)

    @property
    def num_models(self):
        return len(self.proposal_history)

    def propose_model(self, trial_data: TrialData, approval_hist=None, do_append=True):
        if do_append:
            self.proposal_history = self.pretrained_proposal_history[
                : (self.num_models + 1)
            ]
        return self.pretrained_proposal_history[self.num_models]


class FixedProposerFromFile(Proposer):
    """
    File here refers to the models in the file and the data in the files
    """

    def __init__(self, model_files: List, criterion_str, max_loss: float):
        self.model_files = model_files
        self.proposal_history = []
        self.max_loss = max_loss
        if criterion_str == "l1":
            self.raw_criterion = nn.L1Loss(reduce=False)
        else:
            raise ValueError("no other losses implemented right now")

    def criterion(self, pred_y, y):
        return np.abs(pred_y - y) / self.max_loss

    @property
    def num_models(self):
        return len(self.proposal_history)

    def propose_model(
        self,
        trial_data: TrialData,
        approval_hist: ApprovalHistory = None,
        do_append: bool = True,
    ):
        model_dict = torch.load(
            self.model_files[approval_hist.size if approval_hist is not None else 0]
        )
        if do_append:
            fields = model_dict["fields"]
            TEXT = fields["text"][1]
            model = TextSentiment(
                vocab_size=len(TEXT.vocab),
                vocab=TEXT.vocab,
                embed_dim=50,
                num_class=1,
                num_hidden=model_dict["num_hidden"],
            )
            model.load_state_dict(model_dict["state_dict"])
            self.proposal_history.append({"model": model, "fields": fields})
        return model

    def _run_test(self, model_dict, path, test_size, target_func=None):
        test_data = data.TabularDataset(
            path=path, format="json", fields=model_dict["fields"]
        )
        test_iterator = data.Iterator(
            test_data,
            batch_size=test_size,
            sort_key=lambda x: len(x.text),
            sort_within_batch=False,
            shuffle=False,
        )
        model = model_dict["model"]

        for batch in test_iterator:
            # retrieve text and no. of words
            text, text_lengths = batch.text

            # convert to 1D tensor
            predictions = model(text, text_lengths).squeeze()

            # compute the loss
            targets = batch.label if target_func is None else target_func(batch.label)

            # Only do one iteration of the test iterator
            break

        return (
            predictions.detach().numpy(),
            targets.detach().numpy(),
        )

    def get_model_preds_and_target(self, dataset_dict: Dict):
        """
        Get preds and target from the models
        """
        dataset_file = dataset_dict["path"]
        test_size = dataset_dict["batch_size"]
        predictions = []
        for model_dict in self.proposal_history:
            preds, targets = self._run_test(model_dict, dataset_file, test_size)
            predictions.append(preds)
        return np.array(predictions), targets
