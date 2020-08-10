from trial_data import TrialData


class Nature:
    def get_batch(self, time_t: int):
        raise NotImplementedError()


class FixedNature:
    """
    This nature has all the trial data preloaded
    """

    def __init__(self, trial_data: TrialData):
        self.trial_data = trial_data

    @property
    def total_time(self):
        return len(self.trial_data.batch_data)

    def get_trial_data(self, time_t: int):
        return self.trial_data.subset(time_t + 1)
