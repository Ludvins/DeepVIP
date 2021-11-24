from torch.utils.data import Dataset
import numpy as np


class DVIP_Dataset(Dataset):
    def __init__(self, inputs, targets=None, normalize=True):
        self.inputs = inputs

        if targets is not None:
            self.targets = targets
        else:
            self.targets = None

        if normalize:
            # self.inputs_mean = np.mean(inputs)
            # self.inputs_std = np.std(inputs)
            # self.inputs = (self.inputs - self.inputs_mean) / self.inputs_std

            if targets is not None:
                self.targets_mean = np.mean(targets)
                self.targets_std = np.std(targets)
                self.targets = (targets - self.targets_mean) / self.targets_std

    def __getitem__(self, index):

        if self.targets is None:
            return self.inputs[index]
        else:
            return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)
