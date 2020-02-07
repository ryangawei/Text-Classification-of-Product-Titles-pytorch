# coding=utf-8
import torch
from torch.utils.data import DataLoader, Dataset
from nlputils.tokenizer import pad_sequence_to_fixed_length
import numpy as np
import config


class TitleDataset(Dataset):
    def __init__(self, x_set, y_set, max_length):
        assert len(x_set) == len(y_set)
        self.x_set, self.y_set = x_set, y_set
        self._length = len(x_set)
        self._max_length = max_length

    def __len__(self):
        # Return the size of the dataset.
        return self._length

    def __getitem__(self, idx):
        #  Fetching a data sample for a given key.
        sample_x = np.asarray(pad_sequence_to_fixed_length(self.x_set[idx], max_length=self._max_length))
        sample_y = np.asarray(self.y_set[idx])

        return sample_x, sample_y
