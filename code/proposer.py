from typing import List
import numpy as np

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

    def score_mixture_model(self, weights: np.ndarray, dataset: Dataset):
        """
        Score the ensemble model (where we get the weighted avg of the predictions, and then apply the loss)
        """
        predictions = np.array(
            [model.predict(dataset.x) for model in self.proposal_history]
        )
        avg_predictions = np.sum(predictions * np.reshape(weights, (-1, 1, 1)), axis=0)
        return self.proposal_history[0].loss_pred(avg_predictions, dataset.y)


class FixedProposer(Proposer):
    def __init__(self, models: List):
        self.pretrained_proposal_history = models
        self.proposal_history = []

    @property
    def num_models(self):
        return len(self.proposal_history)

    def propose_model(self, trial_data: TrialData, approval_hist = None, do_append=True):
        if do_append:
            self.proposal_history = self.pretrained_proposal_history[
                : (self.num_models + 1)
            ]
        return self.pretrained_proposal_history[self.num_models]


class FixedProposerFromFile(Proposer):
    import torch
    from torch import nn
    from torchtext import data
    from model import TextSentiment
    def __init__(self, model_files: List):
        self.model_files = model_files
        self.proposal_history = []
        self.criterion = nn.L1Loss(reduce=False)

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

    def _run_test(self, model_dict, path, criterion, target_func=None):
        test_data = data.TabularDataset(
            path=path, format="json", fields=model_dict["fields"]
        )
        test_iterator = data.Iterator(
            test_data,
            batch_size=len(test_data),
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
            test_loss = criterion(predictions, targets)
        return (
            test_loss.detach().numpy(),
            predictions.detach().numpy(),
            targets.detach().numpy(),
        )

    def score_models(self, dataset_file: str):
        return np.array(
            [
                self._run_test(model_dict, dataset_file, criterion=self.criterion)[0]
                for model_dict in self.proposal_history
            ]
        )

    def score_mixture_model(self, weights: np.ndarray, dataset_file: str):
        all_preds = []
        prev_targets = None
        for model_dict in self.proposal_history:
            _, preds, targets = self._run_test(
                model_dict, dataset_file, criterion=self.criterion
            )
            if prev_targets is None:
                prev_targets = targets
            assert np.all(prev_targets == targets)
            all_preds.append(preds)
        agg_pred = np.sum(np.array(all_preds) * weights.reshape((-1, 1)), axis=0)
        return (
            self.criterion(torch.Tensor(agg_pred), torch.Tensor(targets))
            .detach()
            .numpy()
        )
