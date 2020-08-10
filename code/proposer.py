from typing import List
import numpy as np

from dataset import Dataset
from trial_data import TrialData


class Proposer:
    def __init__(self):
        self.proposal_history = []

    @property
    def num_models(self):
        return len(self.proposal_history)

    def propose_model(self, trial_data: TrialData):
        raise NotImplementedError()

    def score_models(self, dataset: Dataset):
        return np.array([model.score(dataset) for model in self.proposal_history])

class FixedProposer:
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
