from typing import List
import numpy as np
from numpy import ndarray

from data_generator import DataGenerator
from dataset import Dataset

class TrialData:
    def __init__(
        self,
        batch_sizes: ndarray = None,
        batch_data: List[Dataset] = [],
    ):
        self.batch_sizes = batch_sizes

        self.batch_data = batch_data

    @property
    def num_batches(self):
        return len(self.batch_data)

    def get_start_to_end_data(self, start_index: int, end_index: int = None):
        cum_data = self.batch_data[start_index]
        if end_index is None:
            for data in self.batch_data[start_index + 1 :]:
                cum_data = cum_data.merge(data)
        else:
            for data in self.batch_data[start_index + 1 : end_index]:
                cum_data = cum_data.merge(data)
        return cum_data

    def add_batch(self, data_t: Dataset):
        self.batch_data.append(data_t)

    def subset(self, end_index: int):
        return TrialData(
            self.batch_sizes, self.batch_data[:end_index]
        )

class TrialDataFromDisk:
    def __init__(
        self,
        batch_data_files: List[str] = [],
    ):
        self.batch_data_files = batch_data_files

    @property
    def num_batches(self):
        return len(self.batch_data_files)

    def get_start_to_end_data(self, start_index: int, end_index: int = None) -> Dataset:
        raise NotImplementedError("implement meeee")

    def add_batch(self, file_str: str):
        self.batch_data_files.append(file_str)

    def subset(self, end_index: int):
        return TrialDataFromDisk(
            self.batch_data_files[:end_index]
        )
