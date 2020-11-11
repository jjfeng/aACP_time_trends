from typing import List
import numpy as np

from approval_history import ApprovalHistory
from data_generator import DataGenerator
from trial_data import TrialData
from dataset import Dataset


class Nature:
    def get_batch(self, time_t: int):
        raise NotImplementedError()

    def get_trial_data(self, time_t: int):
        """
        Return all trial data up to time_t, inclusive
        """
        subtrial_data = self.trial_data.subset(time_t + 1)
        return subtrial_data

    def create_test_data(self, time_t: int):
        """
        Return trial data for this time_t
        If None, means no new test data
        """
        # Simply return data from this batch
        return None


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
        self.batch_sizes = trial_data.batch_sizes
        self.data_gen = data_gen
        self.coefs = coefs

    def next(self, approval_hist: ApprovalHistory = None):
        # Do nothing
        return

    @property
    def total_time(self):
        return self.trial_data.num_batches

    def create_test_data(self, time_t: int, num_obs: int = 1000) -> Dataset:
        # Generate new test data if possible
        if self.data_gen is not None:
            return self.data_gen.create_data(num_obs, time_t, self.coefs[time_t])
        else:
            return None

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
        prob_revert_drift: float = 0,
    ):
        self.data_gen = data_gen
        self.num_coef_drift = num_coef_drift
        self.batch_sizes = batch_sizes
        self.num_p = data_gen.num_p
        self.init_coef = init_coef
        self.trial_data = TrialData([])
        self.coefs = []
        self.prob_revert_drift = prob_revert_drift
        self.last_coef_change = 0

    def next(self, approval_hist: ApprovalHistory = None):
        # np.random.seed(len(self.coefs))
        curr_approv = 0
        prev_approv = 0
        if approval_hist is not None and approval_hist.size >= 2:
            curr_approv = approval_hist.currently_approved_idx
            prev_approv = approval_hist.last_approved_idx

        if curr_approv is not None and prev_approv is not None and (curr_approv - prev_approv) >= 1:
            # do drift because the approval history approved something new
            print("time", self.curr_time, "DO DRIFT", curr_approv, prev_approv)
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
                new_coef = self.coefs[-1] if len(self.coefs) else self.init_coef

        self.coefs.append(new_coef)

        new_data = self.data_gen.create_data(
            self.batch_sizes[self.curr_time], self.curr_time, coef=self.coefs[-1]
        )
        self.trial_data.add_batch(new_data)
        assert self.trial_data.num_batches == len(self.coefs)

    @property
    def curr_time(self):
        return self.trial_data.num_batches

    @property
    def total_time(self):
        return len(self.batch_sizes)

    def create_test_data(self, time_t: int, num_obs: int = 1000):
        # Generate new test data if possible
        if self.data_gen is not None:
            return self.data_gen.create_data(num_obs, time_t, self.coefs[time_t])
        else:
            return None

    def to_fixed(self):
        return FixedNature(self.data_gen, self.trial_data, self.coefs)
