from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Training_Dataset(Dataset):
    def __init__(self, inputs, targets, verbose=True):
        self.inputs = inputs

        self.targets_mean = np.mean(targets, axis=0, keepdims=True)
        self.targets_std = np.std(targets, axis=0, keepdims=True)
        self.targets = (targets - self.targets_mean) / self.targets_std

        self.n_samples = inputs.shape[0]
        self.input_dim = inputs.shape[1]
        self.output_dim = targets.shape[1]

        # Normalize inputs
        self.inputs_std = np.std(self.inputs, axis=0, keepdims=True)
        self.inputs_mean = np.mean(self.inputs, axis=0, keepdims=True)

        self.inputs_std[self.inputs_std == 0] = 1
        self.inputs = (self.inputs - self.inputs_mean) / self.inputs_std
        if verbose:
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
    def __init__(self, inputs, targets=None, inputs_mean=0.0, inputs_std=1.0):
        self.inputs = (inputs - inputs_mean) / inputs_std
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


uci_base = "https://archive.ics.uci.edu/ml/machine-learning-databases/"


class DVIPDataset(Dataset):
    def __init__(self):
        raise NotImplementedError

    def split_data(self, data):
        generator = torch.Generator()
        generator.manual_seed(2147483647)
        perm = torch.randperm(data.shape[0], generator=generator)
        self.inputs = data[perm, :-1]
        self.targets = data[perm, -1]

        if len(self.inputs.shape) == 1:
            self.inputs = self.inputs[..., np.newaxis]
        if len(self.targets.shape) == 1:
            self.targets = self.targets[..., np.newaxis]

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)


class Synthetic_Dataset(DVIPDataset):
    def __init__(self):
        rng = np.random.default_rng(seed=0)

        def f(x):
            return np.cos(5 * x) / (np.abs(x) + 1)

        inputs = rng.standard_normal(300)
        targets = f(inputs) + rng.standard_normal(inputs.shape) * 0.1

        self.inputs = inputs[..., np.newaxis]
        self.targets = targets[..., np.newaxis]


class Boston_Dataset(DVIPDataset):
    def __init__(self):
        print("Loading boston....")

        data_url = "{}{}".format(uci_base, "housing/housing.data")
        raw_df = pd.read_fwf(data_url, header=None).to_numpy()
        self.split_data(raw_df)


class Energy_Dataset(DVIPDataset):
    def __init__(self):
        print("Loading energy....")

        url = "{}{}".format(uci_base, "00242/ENB2012_data.xlsx")
        data = pd.read_excel(url).values
        data = data[:, :9]
        self.split_data(data)


class Concrete_Dataset(DVIPDataset):
    def __init__(self):
        print("Loading concrete....")

        url = "{}{}".format(uci_base, "concrete/compressive/Concrete_Data.xls")
        data = pd.read_excel(url).values
        self.split_data(data)


class Naval_Dataset(DVIPDataset):
    def __init__(self):
        print("Loading naval....")

        url = "{}{}".format(uci_base, "00316/UCI%20CBM%20Dataset.zip")
        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall("/tmp/")

        data = pd.read_fwf("/tmp/UCI CBM Dataset/data.txt", header=None).values
        data = data[:, :-1]
        self.split_data(data)


class Power_Dataset(DVIPDataset):
    def __init__(self):
        url = "{}{}".format(uci_base, "00294/CCPP.zip")
        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall("/tmp/")

        data = pd.read_excel("/tmp/CCPP//Folds5x2_pp.xlsx").values
        self.split_data(data)


class Protein_Dataset(DVIPDataset):
    def __init__(self):
        url = "{}{}".format(uci_base, "00265/CASP.csv")
        data = pd.read_csv(url).values
        data = np.concatenate([data[:, 1:], data[:, 0, None]], 1)
        self.split(data)


class Kin8nm_Dataset(DVIPDataset):
    def __init__(self):
        url = 'http://mldata.org/repository/data/download/csv/uci-20070111-kin8nm'
        data = pd.read_csv(url, header=None).values
        self.split_data(data)


class Yatch_Dataset(DVIPDataset):
    def __init__(self):
        url = "{}{}".format(uci_base, "00243/yacht_hydrodynamics.data")
        data = pd.read_csv(url, header=None).values
        self.split_data(data)


class WineRed_Dataset(DVIPDataset):
    def __init__(self):
        url = "{}{}".format(uci_base, "wine-quality/winequality-red.csv")
        data = pd.read_csv(url, delimiter=";").values
        self.split_data(data)


class WineWhite_Dataset(DVIPDataset):
    def __init__(self):
        url = "{}{}".format(uci_base, "wine-quality/winequality-white.csv")
        data = pd.read_csv(url, delimiter=";").values
        self.split_data(data)


def get_dataset(dataset_name):
    if dataset_name == "boston":
        return Boston_Dataset()
    elif dataset_name == "energy":
        return Energy_Dataset()
    elif dataset_name == "concrete":
        return Concrete_Dataset()
    elif dataset_name == "naval":
        return Naval_Dataset()
    elif dataset_name == "kin8nm":
        return Kin8nm_Dataset()
    elif dataset_name == "yatch":
        return Yatch_Dataset()
    elif dataset_name == "power":
        return Power_Dataset()
    elif dataset_name == "protein":
        return Protein_Dataset()
    elif dataset_name == "winered":
        return WineRed_Dataset()
    elif dataset_name == "winewhite":
        return WineWhite_Dataset()
    elif dataset_name == "synthetic":
        return Synthetic_Dataset()
    else:
        raise RuntimeError("No available dataset selected.")
