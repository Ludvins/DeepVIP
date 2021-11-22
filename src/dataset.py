from torch.utils.data import Dataset
import numpy as np


class DVIP_Dataset(Dataset):
    def __init__(self, inputs, targets=None):
        self.inputs = inputs

        if targets is not None:
            self.targets_mean = np.mean(targets)
            self.targets_std = np.std(targets)
            self.targets = (targets - self.targets_mean) / self.targets_std
        else:
            self.targets = None

    def __getitem__(self, index):

        if self.targets is None:
            return self.inputs[index]
        else:
            return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)