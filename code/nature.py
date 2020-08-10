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

    def get_batch(self, time_t: int):
        return self.batch_data[time_t]
