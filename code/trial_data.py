from typing import List
import numpy as np
from numpy import ndarray

from data_generator import DataGenerator
from dataset import Dataset

class TrialData:
    def __init__(
        self, batch_sizes: ndarray = [], batch_data: List[Dataset] = [],
    ):
        self.batch_sizes = batch_sizes

        self.batch_data = batch_data

    @property
    def num_batches(self):
        return len(self.batch_data)

    def get_start_to_end_data(self, start_index: int, end_index: int = None,
            holdout_last_batch: float = 0) -> Dataset:
        cum_data = self.batch_data[start_index]
        if end_index is None:
            for data in self.batch_data[start_index + 1 : -1]:
                cum_data = cum_data.merge(data)
            end_index = len(self.batch_data) - 1
        else:
            for data in self.batch_data[start_index + 1 : end_index - 1]:
                cum_data = cum_data.merge(data)
        last_batch = self.batch_data[end_index]
        sub_last_batch = last_batch.subset(np.arange(int(last_batch.num_obs * (1 -
            holdout_last_batch))))
        cum_data = cum_data.merge(sub_last_batch)
        return cum_data

    def add_batch(self, data: Dataset):
        self.batch_data.append(data)
        self.batch_sizes.append(data.num_obs)

    def subset(self, end_index: int = None, holdout_last_batch: float = 0):
        """
        @param end_index: not inclusive index
        """
        if end_index is None:
            end_index = len(self.batch_data)
        sub_last_batch = self.batch_data[end_index - 1].get_train(
            holdout_last_batch)
        num_last = sub_last_batch.num_obs
        if end_index >= 2:
            new_batch_data = self.batch_data[:end_index - 1] + [sub_last_batch]
            batch_sizes = self.batch_sizes[:end_index - 1] + [num_last]
        else:
            new_batch_data = [sub_last_batch]
            batch_sizes = [num_last]

        return TrialData(batch_sizes, new_batch_data)


class TrialDataFromDisk(TrialData):
    def __init__(
        self, batch_data: List[str] = [], batch_sizes: ndarray = [],
    ):
        self.batch_data = batch_data
        self.batch_sizes = batch_sizes

    @property
    def num_batches(self):
        return len(self.batch_data)

    def add_batch(self, file_str: str, batch_size: int):
        self.batch_data.append({"path": file_str, "batch_size": batch_size})
        self.batch_sizes.append(batch_size)

    def subset(self, end_index: int):
        return TrialDataFromDisk(self.batch_data[:end_index])
