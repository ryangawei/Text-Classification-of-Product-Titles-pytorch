# coding=utf-8
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


class TitleDataset(Dataset):
    def __init__(self, x_set, y_set):
        assert len(x_set) == len(y_set)
        self.x, self.y = x_set, y_set
        self._length = len(x_set)

    def __len__(self):
        # Return the size of the dataset.
        return self._length

    def __getitem__(self, idx):
        #  Fetching a data sample for a given key.
        sample_x = np.asarray(self.x[idx])
        sample_y = np.asarray(self.y[idx])

        return sample_x, sample_y