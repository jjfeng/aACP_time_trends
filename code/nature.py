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

class AdversarialNature:
    """
    """

    def __init__(self, data_generator: DataGenerator, drift_speed: float, batch_sizes: np.ndarray):
        self.data_gen = data_gen
        self.drift_speed = drift_speed
        self.batch_sizes = batch_sizes

    def start(self):
        self.trial_data = TrialData(self.batch_sizes, data_gen)

    def next(self, did_approval: bool):
        if did_approval:
            self.trial_data.make_new_batch(drift_speed=self.drift_speed)
        else:
            self.trial_data.make_new_batch(drift_speed=0)

    @property
    def total_time(self):
        return self.batch_sizes.size

    def get_trial_data(self, time_t: int):
        return self.trial_data.subset(time_t + 1)
