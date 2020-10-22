from typing import List
import numpy as np

from approval_history import ApprovalHistory
from data_generator import DataGenerator
from trial_data import TrialData


class Nature:
    def get_batch(self, time_t: int):
        raise NotImplementedError()

    def get_trial_data(self, time_t: int):
        subtrial_data = self.trial_data.subset(time_t + 1)
        subtrial_data.load()
        return subtrial_data

    def create_test_data(self, time_t: int):
        raise NotImplementedError()


class FixedNature(Nature):
    """
    This nature has all the trial data preloaded
    """

    def __init__(
        self,
        data_gen: DataGenerator = None,
        trial_data: TrialData = None,
        coefs: List = None,
    ):
        self.trial_data = trial_data
        self.data_gen = data_gen
        self.coefs = coefs

    def next(self, approval_hist: ApprovalHistory = None):
        # Do nothing
        return

    @property
    def batch_sizes(self):
        return self.trial_data.batch_sizes

    @property
    def total_time(self):
        return self.trial_data.num_batches

    def create_test_data(self, time_t: int, num_obs: int = 1000):
        if self.data_gen is not None:
            return self.data_gen.create_data(num_obs, time_t, self.coefs[time_t])
        else:
            return self.trial_data.batch_data[time_t]

    def to_fixed(self):
        return self


class AdversarialNature(Nature):
    """"""

    def __init__(
        self,
        data_gen: DataGenerator,
        num_coef_drift: int,
        batch_sizes: List,
        init_coef: np.ndarray,
        prob_revert_drift: float = 0
    ):
        self.data_gen = data_gen
        self.num_coef_drift = num_coef_drift
        self.batch_sizes = batch_sizes
        self.num_p = data_gen.num_p
        self.init_coef = init_coef
        self.trial_data = TrialData(self.batch_sizes)
        self.coefs = [self.init_coef]
        self.prob_revert_drift = prob_revert_drift
        self.last_coef_change = 0

    def next(self, approval_hist: ApprovalHistory = None):
        np.random.seed(len(self.coefs))
        approval_direction = 0
        if approval_hist is not None and approval_hist.size > 2:
            curr_approv = np.sum(
                np.arange(approval_hist.size) * approval_hist.expert_weights_history[-1]
            )
            prev_approv = np.sum(
                np.arange(approval_hist.size - 1)
                * approval_hist.expert_weights_history[-2]
            )
            approval_direction = curr_approv - prev_approv

        if approval_direction >= 1.0:
            # do drift
            print("time", self.curr_time, "DO DRIFT", approval_direction)
            new_coef = np.copy(self.coefs[-1])
            to0_rand_idx = np.random.choice(
                np.where(np.abs(new_coef) > 0)[0], size=self.num_coef_drift
            )
            to1_rand_idx = np.random.choice(
                np.where(np.abs(new_coef) <= 1e-10)[0], size=self.num_coef_drift
            )
            new_coef[to0_rand_idx] = 0
            new_coef[to1_rand_idx] = np.max(self.coefs[0])

            self.last_coef_change = len(self.coefs) - 1
        else:
            if np.random.rand() < self.prob_revert_drift:
                # Try reverting to old coefs
                new_coef = self.coefs[self.last_coef_change]
                self.last_coef_change = len(self.coefs) - 1
            else:
                # no drift at all
                new_coef = self.coefs[-1]

        self.coefs.append(new_coef)

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

    def create_test_data(self, time_t: int, num_obs: int = 1000):
        if self.data_gen is not None:
            return self.data_gen.create_data(num_obs, time_t, self.coefs[time_t])
        else:
            return self.trial_data.batch_data[time_t]

    def to_fixed(self):
        return FixedNature(self.data_gen, self.trial_data, self.coefs)
