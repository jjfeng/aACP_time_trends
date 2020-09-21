from typing import List
import numpy as np

from approval_history import ApprovalHistory
from data_generator import DataGenerator
from trial_data import TrialData


class Nature:
    def get_batch(self, time_t: int):
        raise NotImplementedError()

    def get_trial_data(self, time_t: int):
        return self.trial_data.subset(time_t + 1)

class FixedNature(Nature):
    """
    This nature has all the trial data preloaded
    """

    def __init__(self, trial_data: TrialData):
        self.trial_data = trial_data

    @property
    def total_time(self):
        return len(self.trial_data.batch_data)

class AdversarialNature(Nature):
    """
    """

    def __init__(self, data_gen: DataGenerator, drift_speed: float, batch_sizes: List, init_coef: np.ndarray):
        self.data_gen = data_gen
        self.drift_speed = drift_speed
        self.batch_sizes = batch_sizes
        self.num_p = data_gen.num_p
        self.init_coef = init_coef
        self.trial_data = TrialData(self.batch_sizes)
        self.coefs = [self.init_coef]

    def next(self, approval_hist: ApprovalHistory=None):
        approval_direction = 0
        if approval_hist is not None and approval_hist.size > 2:
            curr_approv = np.sum(
                    np.arange(approval_hist.size) * approval_hist.expert_weights_history[-1]
            )
            prev_approv = np.sum(
                    np.arange(approval_hist.size - 1) * approval_hist.expert_weights_history[-2]
            )
            approval_direction = curr_approv - prev_approv

        if approval_direction > 0:
            # do drift
            print("time", self.curr_time, "DO DRIFT")
            coef_norm = np.sqrt(np.sum(np.power(self.coefs[-1], 2)))
            new_noise = np.random.randn(1, self.num_p)
            new_coef = self.coefs[-1] * (1 - self.drift_speed) + new_noise * self.drift_speed / np.sqrt(np.sum(np.power(new_noise, 2))) * coef_norm
            self.coefs.append(new_coef)
        else:
            # no drift
            self.coefs.append(self.coefs[-1])

        new_data = self.data_gen.create_data(
            self.batch_sizes[self.curr_time], self.curr_time, coef=self.coefs[-1]
        )
        self.trial_data.add_batch(new_data)

    @property
    def curr_time(self):
        return self.trial_data.num_batches

    @property
    def total_time(self):
        return len(self.batch_sizes)
