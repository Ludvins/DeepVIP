from torch.utils.data import Dataset
import numpy as np


class DVIP_Dataset(Dataset):
    def __init__(self, inputs, targets=None):
        self.inputs = inputs

        if targets is not None:
            self.targets_mean = np.mean(targets)
            self.targets_std = np.std(targets)
            self.targets = (targets - self.targets_mean) / self.targets_std

    def __getitem__(self, index):
        X = self.inputs[index]
        y = self.targets[index]

        return X, y

    def __len__(self):
        return len(self.inputs)