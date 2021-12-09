from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class DVIP_Dataset(Dataset):
    def __init__(self, inputs, targets=None, normalize=True):
        self.inputs = inputs

        if targets is not None:
            self.targets = targets
        else:
            self.targets = None

        if normalize:
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


class Training_Dataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs

        self.targets_mean = np.mean(targets, axis=0)
        self.targets_std = np.std(targets, axis=0)
        self.targets = (targets - self.targets_mean) / self.targets_std
        self.n_samples = inputs.shape[0]
        self.input_dim = inputs.shape[1]
        self.output_dim = targets.shape[1]

        print("Number of samples: ", self.n_samples)
        print("Input dimension: ", self.input_dim)
        print("Label dimension: ", self.output_dim)
        print("Labels mean value: ", self.targets_mean)
        print("Labels standard deviation: ", self.targets_std)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)


class Test_Dataset(Dataset):
    def __init__(self, inputs, targets=None):
        self.inputs = inputs
        self.targets = targets
        self.n_samples = inputs.shape[0]
        self.input_dim = inputs.shape[1]
        self.output_dim = targets.shape[1]

    def __getitem__(self, index):
        if self.targets is None:
            return self.inputs[index]
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)


uci_base = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'


class Boston_Dataset(Dataset):
    def __init__(self):
        print("Loading boston....")

        data_url = '{}{}'.format(uci_base, 'housing/housing.data')
        raw_df = pd.read_fwf(data_url, header=None).to_numpy()
        self.inputs = raw_df[:, :-1]
        self.targets = raw_df[:, -1][..., np.newaxis]

        self.inputs = (self.inputs - np.mean(self.inputs, axis=0)) / np.std(
            self.inputs, axis=0)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)


class Energy_Dataset(Dataset):
    def __init__(self):
        print("Loading energy....")

        url = '{}{}'.format(uci_base, '00242/ENB2012_data.xlsx')
        data = pd.read_excel(url).values
        data = data[:, :-2]
        self.inputs = data[:, :8]
        self.targets = data[:, -1][..., np.newaxis]

        self.inputs = (self.inputs - np.mean(self.inputs, axis=0)) / np.std(
            self.inputs, axis=0)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)