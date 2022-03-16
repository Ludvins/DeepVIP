from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


class Training_Dataset(Dataset):
    def __init__(self, inputs, targets,
                 verbose=True,
                 normalize_inputs = True,
                 normalize_targets=True):

        self.inputs = inputs
        if normalize_targets:
            self.targets_mean = np.mean(targets, axis=0, keepdims=True)
            self.targets_std = np.std(targets, axis=0, keepdims=True)
        else:
            self.targets_mean = 0
            self.targets_std = 1
        self.targets = (targets - self.targets_mean) / self.targets_std

        self.n_samples = inputs.shape[0]
        self.input_dim = inputs.shape[1]
        self.output_dim = targets.shape[1]

        # Normalize inputs
        if normalize_inputs:
            self.inputs_std = np.std(self.inputs, axis=0, keepdims=True) + 1e-6
            self.inputs_mean = np.mean(self.inputs, axis=0, keepdims=True)
        else:
            self.inputs_mean = 0
            self.inputs_std = 1     

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
        if self.targets is not None:
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
        self.inputs = data[:, :-1]
        self.targets = data[:, -1]

        if len(self.inputs.shape) == 1:
            self.inputs = self.inputs[..., np.newaxis]
        if len(self.targets.shape) == 1:
            self.targets = self.targets[..., np.newaxis]

    def get_split(self, test_size, seed):
        train_indexes, test_indexes = train_test_split(
            np.arange(len(self)), test_size=test_size, random_state=seed
        )

        train_dataset = Training_Dataset(
            self.inputs[train_indexes],
            self.targets[train_indexes],
            normalize_targets=self.type == "regression",
        )
        train_test_dataset = Test_Dataset(
            self.inputs[train_indexes],
            self.targets[train_indexes],
            train_dataset.inputs_mean,
            train_dataset.inputs_std,
        )
        test_dataset = Test_Dataset(
            self.inputs[test_indexes],
            self.targets[test_indexes],
            train_dataset.inputs_mean,
            train_dataset.inputs_std,
        )

        return train_dataset, train_test_dataset, test_dataset

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)

    def len_train(self, test_size):
        train_indexes, test_indexes = train_test_split(
            np.arange(len(self)), test_size=test_size, random_state=0
        )
        return len(train_indexes)


class SPGP_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1

        inputs = np.loadtxt("data/SPGP_dist/train_inputs")
        targets = np.loadtxt("data/SPGP_dist/train_outputs")

        self.inputs = inputs[..., np.newaxis]
        self.targets = targets[..., np.newaxis]


class Synthetic_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1

        rng = np.random.default_rng(seed=0)

        def f(x):
            return np.cos(5 * x) / (np.abs(x) + 1)

        # inputs = rng.standard_normal(300)
        inputs = np.linspace(-1.0, 1.0, 300)
        targets = f(inputs) + rng.standard_normal(inputs.shape) * 0.1

        self.inputs = inputs[..., np.newaxis]
        self.targets = targets[..., np.newaxis]


class Boston_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        data_url = "{}{}".format(uci_base, "housing/housing.data")
        raw_df = pd.read_fwf(data_url, header=None).to_numpy()
        self.split_data(raw_df)


class Energy_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        url = "{}{}".format(uci_base, "00242/ENB2012_data.xlsx")
        data = pd.read_excel(url).values
        data = data[:, :9]
        self.split_data(data)


class Concrete_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        url = "{}{}".format(uci_base, "concrete/compressive/Concrete_Data.xls")
        data = pd.read_excel(url).values
        self.split_data(data)


class Naval_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        try:
            data = pd.read_fwf("/tmp/UCI CBM Dataset/data.txt", header=None).values
        except:
            url = "{}{}".format(uci_base, "00316/UCI%20CBM%20Dataset.zip")
            with urlopen(url) as zipresp:
                with ZipFile(BytesIO(zipresp.read())) as zfile:
                    zfile.extractall("/tmp/")

            data = pd.read_fwf("/tmp/UCI CBM Dataset/data.txt", header=None).values
        data = data[:, :-1]
        self.split_data(data)


class Power_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        try:
            data = pd.read_excel("/tmp/CCPP//Folds5x2_pp.xlsx").values
        except:
            url = "{}{}".format(uci_base, "00294/CCPP.zip")
            with urlopen(url) as zipresp:
                with ZipFile(BytesIO(zipresp.read())) as zfile:
                    zfile.extractall("/tmp/")

            data = pd.read_excel("/tmp/CCPP//Folds5x2_pp.xlsx").values
        self.split_data(data)


class Protein_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        url = "{}{}".format(uci_base, "00265/CASP.csv")
        data = pd.read_csv(url).values
        data = np.concatenate([data[:, 1:], data[:, 0, None]], 1)
        self.split_data(data)


class Kin8nm_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        url = "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.arff"
        data = pd.read_csv(url, dtype=float).values
        self.split_data(data)


class Yatch_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        url = "{}{}".format(uci_base, "00243/yacht_hydrodynamics.data")
        data = pd.read_csv(url, header=None).values
        self.split_data(data)


class WineRed_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        url = "{}{}".format(uci_base, "wine-quality/winequality-red.csv")
        data = pd.read_csv(url, delimiter=";").values
        self.split_data(data)


class WineWhite_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        url = "{}{}".format(uci_base, "wine-quality/winequality-white.csv")
        data = pd.read_csv(url, delimiter=";").values
        self.split_data(data)


class C02_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        url = "https://scrippsco2.ucsd.edu/assets/data/atmospheric/stations/in_situ_co2/monthly/monthly_in_situ_co2_mlo.csv"
        data = pd.read_csv(url, comment='"', header=[0, 1, 2], dtype=float).values[
            :, [3, 4]
        ]
        mask = (data == -99.99).any(1)
        self.data = data[~mask]
        self.len_data = self.data.shape[0]
        
        split = self.len_data//5
        test_indexes = np.concatenate(
            (np.arange(split, 2*split),
            np.arange(3*split, 4*split)),
            axis = 0
        )
        train_indexes = np.setdiff1d(np.arange(self.len_data), test_indexes)

        
        train_data = self.data[train_indexes, :1]
        test_data = self.data[test_indexes, :1]
        train_targets = self.data[train_indexes, 1:]
        test_targets = self.data[test_indexes, 1:]

        
        self.train = Training_Dataset(train_data, train_targets)

        self.train_test = Test_Dataset(
            self.data[:, :1],
            self.data[:, 1:],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_data,
            test_targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )    

    def __len__(self):
        return 778
    
    def len_train(self, test_size):
        return len(self.train)

    def get_split(self, split, *args):
        return self.train, self.train_test, self.test


class MNIST_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "multiclass"
        self.classes = 10
        self.output_dim = 10
        train = datasets.MNIST(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        test = datasets.MNIST(
            root="./data", train=False, download=True, transform=transforms.ToTensor()
        )
        
        train_data = train.data.reshape(60000, -1)/255.
        test_data = test.data.reshape(10000, -1)/255.
        train_targets = train.targets.reshape(-1, 1)
        test_targets = test.targets.reshape(-1, 1)

        self.train = Training_Dataset(
            train_data.numpy(), train_targets.numpy(),
            normalize_targets=False, 
            normalize_inputs=False,
        )

        self.train_test = Test_Dataset(
            train_data.numpy(),
            train_targets.numpy(),
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_data.numpy(),
            test_targets.numpy(),
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return 70000

    def get_split(self, *args):
        return self.train, self.train_test, self.test

    def len_train(self, test_size):
        return 60000


class Iris_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "multiclass"
        self.classes = 3
        self.output_dim = 3
        url = "{}{}".format(uci_base, "iris/iris.data")

        data = pd.read_csv(url, header=None)
        data[4].replace(
            ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
            [0, 1, 2],
            inplace=True,
        )
        self.split_data(data.values)
        
class SolarIrradiance_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        url = "https://lasp.colorado.edu/lisird/latis/dap/nrl2_tsi_P1M.csv?&time,irradiance"

        self.data = pd.read_csv(url, header=[0], dtype=float).values
        self.len_data = self.data.shape[0]
        split = self.len_data//5
        d = self.len_data // 10
        test_indexes = np.concatenate(
            (np.arange(split, split + d),
            np.arange(2*split, 2*split + d),
            np.arange(3*split, 3*split + d),
            np.arange(4*split, 4*split + d)
            ),
            axis = 0
        )
        train_indexes = np.setdiff1d(np.arange(self.len_data), test_indexes)
        
        
        train_data = self.data[train_indexes, :1]
        test_data = self.data[test_indexes, :1]
        train_targets = self.data[train_indexes, 1:]
        test_targets = self.data[test_indexes, 1:]

        
        self.train = Training_Dataset(train_data, train_targets)

        self.train_test = Test_Dataset(
            self.data[:, :1],
            self.data[:, 1:],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_data,
            test_targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )    

    def __len__(self):
        return self.len_data
    
    def len_train(self, test_size):
        return len(self.train)

    def get_split(self, split, *args):
        return self.train, self.train_test, self.test
    
class Rectangles_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "binaryclass"
        self.classes = 2
        self.output_dim = 1
        
        train_data = np.loadtxt('data/rectangles/rectangles_im_train.amat')
        test_data = np.loadtxt('data/rectangles/rectangles_im_test.amat')
        self.len_data = train_data.shape[0] + test_data.shape[0]
        self.train = Training_Dataset(train_data[:, :-1],
                                      train_data[:, -1].reshape(-1, 1),
                                      normalize_targets=False, 
                                      normalize_inputs=True)

        self.train_test = Test_Dataset(
            train_data[:, :-1],
            train_data[:, -1].reshape(-1, 1),
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_data[:, :-1],
            test_data[:, -1].reshape(-1, 1),
            self.train.inputs_mean,
            self.train.inputs_std,
        )    

    def __len__(self):
        return self.len_data
    
    def len_train(self, test_size):
        return len(self.train)

    def get_split(self, split, *args):
        return self.train, self.train_test, self.test
        
    
class Banknote_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "binaryclass"
        self.classes = 2
        self.output_dim = 1
        
        url = "{}{}".format(uci_base, "00267/data_banknote_authentication.txt")
        data = np.loadtxt(url, delimiter =",")
        self.split_data(data)
        

class Bimodal_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1

        rng = np.random.default_rng(seed=0)

        x = rng.uniform(size=2000) * 8 - 4
        epsilon = rng.normal(size=2000)

        c = rng.normal(size=2000)
        y = np.zeros_like(x)
        y[c >= 0.5] = 10 * np.cos(x[c >= 0.5] - 0.5) + epsilon[c >= 0.5]

        y[c < 0.5] = 10 * np.sin(x[c < 0.5] - 0.5) + epsilon[c < 0.5]

        self.inputs = x[..., np.newaxis]
        self.targets = y[..., np.newaxis]


class Heterocedastic_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1

        rng = np.random.default_rng(seed=0)

        x = rng.uniform(size=2000) * 8 - 4
        epsilon = rng.normal(size=2000) * 2

        sin = np.sin(x)

        y = 7 * sin + epsilon * sin + 10
        self.inputs = x[..., np.newaxis]
        self.targets = y[..., np.newaxis]


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
    elif dataset_name == "SPGP":
        return SPGP_Dataset()
    elif dataset_name == "CO2":
        return C02_Dataset()
    elif dataset_name == "Solar":
        return SolarIrradiance_Dataset()
    elif dataset_name == "MNIST":
        return MNIST_Dataset()
    elif dataset_name == "Iris":
        return Iris_Dataset()
    elif dataset_name == "Rectangles":
        return Rectangles_Dataset()
    elif dataset_name == "Bimodal":
        return Bimodal_Dataset()
    elif dataset_name == "Heterocedastic":
        return Heterocedastic_Dataset()
    elif dataset_name == "Banknote":
        return Banknote_Dataset()
    else:
        raise RuntimeError("No available dataset selected.")
