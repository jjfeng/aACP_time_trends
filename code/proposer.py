from typing import List
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

    @property
    def num_models(self):
        return len(self.proposal_history)

    def propose_model(self, trial_data: TrialData):
        raise NotImplementedError()

    def score_models(self, dataset: Dataset):
        return np.array([model.loss(dataset) for model in self.proposal_history])


class FixedProposer(Proposer):
    def __init__(self, models: List):
        self.pretrained_proposal_history = models
        self.proposal_history = []

    @property
    def num_models(self):
        return len(self.proposal_history)

    def propose_model(self, trial_data: TrialData):
        self.proposal_history = self.pretrained_proposal_history[
            : (self.num_models + 1)
        ]

class FixedProposerFromFile(Proposer):
    def __init__(self, model_files: List):
        self.model_files = model_files
        self.proposal_history = []
        self.criterion = nn.L1Loss(reduce=False)

    @property
    def num_models(self):
        return len(self.proposal_history)

    def propose_model(self, trial_data: TrialData, approval_hist: ApprovalHistory=None, do_append: bool = True):
        model = torch.load(self.model_files[approval_hist.size if approval_hist is not None else 0])
        if do_append:
            self.proposal_history.append(model)
        return model

    def _run_test(self, model_dict, path, criterion, target_func=None):

        fields = model_dict["fields"]
        TEXT = fields["text"][1]
        test_data = data.TabularDataset(path=path, format="json", fields=fields)
        test_iterator = data.Iterator(
            test_data,
            batch_size=len(test_data),
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            shuffle=True,
        )
        model = TextSentiment(
            vocab_size=len(TEXT.vocab),
            vocab=TEXT.vocab,
            embed_dim=50,
            num_class=1,
            num_hidden=model_dict["num_hidden"],
        )
        model.load_state_dict(model_dict["state_dict"])

        for batch in test_iterator:
            # retrieve text and no. of words
            text, text_lengths = batch.text

            # convert to 1D tensor
            predictions = model(text, text_lengths).squeeze()

            # compute the loss
            targets = batch.label if target_func is None else target_func(batch.label)
            # targets = targets- 1
            test_loss = criterion(predictions, targets).detach().numpy()
        return test_loss

    def score_models(self, dataset_file: str):
        return np.array([self._run_test(model_dict, dataset_file, criterion=self.criterion) for model_dict in self.proposal_history])

