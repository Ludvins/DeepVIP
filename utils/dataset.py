from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import torchvision.transforms as trn

class Training_Dataset(Dataset):
    def __init__(
        self,
        inputs,
        targets,
        verbose=True,
        normalize_inputs=True,
        normalize_targets=True,
    ):

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
        return self.inputs[index], self.targets[index], index

    def __len__(self):
        return len(self.inputs)


uci_base = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
data_base = "./data"


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
        full_dataset = Test_Dataset(
            self.inputs,
            self.targets,
            train_dataset.inputs_mean,
            train_dataset.inputs_std,
        )
        test_dataset = Test_Dataset(
            self.inputs[test_indexes],
            self.targets[test_indexes],
            train_dataset.inputs_mean,
            train_dataset.inputs_std,
        )

        return train_dataset, full_dataset, test_dataset

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
        test_inputs = np.loadtxt("data/SPGP_dist/test_inputs")
        #test_inputs = np.linspace(-40, 50, 200)
        mask = ...#((inputs < 1.5) | (inputs > 3.5)).flatten()

        
        
        self.train = Training_Dataset(
            inputs[mask, np.newaxis],
            targets[mask, np.newaxis],
            normalize_targets=True,
            normalize_inputs=True,
        )

        self.full = Test_Dataset(
            inputs[mask, np.newaxis],
            targets[mask, np.newaxis],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_inputs[..., np.newaxis],
            None,
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return 200

    def get_split(self, *args):
        return self.train, self.full, self.test

    def len_train(self, test_size):
        return 200

class Synthetic_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1

        rng = np.random.default_rng(seed=0)

        def f(x):
            return np.cos(5 * x) / (np.abs(x) + 1)

        inputs = rng.standard_normal(300)
        #inputs = np.linspace(-1.0, 1.0, 300)
        targets = f(inputs) + rng.standard_normal(inputs.shape)*0.1 * inputs

        self.inputs = inputs[..., np.newaxis]
        self.targets = targets[..., np.newaxis]

class Synthetic_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1

        rng = np.random.default_rng(seed=0)

        def f(x):
            return np.sin(4 * x) 

        # inputs = rng.standard_normal(300)
        inputs = np.linspace(-2.0,2.0, 300)
        targets = f(inputs) + rng.standard_normal(inputs.shape)*0.1
        
        mask = (((inputs > -1) * (inputs < -0.5)) | ((inputs < 1) * (inputs > 0.5))).flatten()
        #mask = ...
        
        self.train = Training_Dataset(
            inputs[mask, np.newaxis],
            targets[mask, np.newaxis],
            normalize_targets=True,
            normalize_inputs=False,
        )

        self.full = Test_Dataset(
            inputs[..., np.newaxis],
            targets[..., np.newaxis],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            inputs[..., np.newaxis],
            targets[..., np.newaxis],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return 300

    def get_split(self, *args):
        return self.train, self.full, self.test

    def len_train(self, test_size):
        return 200 



class Synthetic2_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1

        data = np.load("data/data.npy")
        inputs, targets = data[:, 0], data[:, 1]

        test_inputs = np.linspace(np.min(inputs)-5, np.max(inputs)+5, 200)

        self.train = Training_Dataset(
            inputs[..., np.newaxis],
            targets[..., np.newaxis],
            normalize_targets=True,
            normalize_inputs=False,
        )

        self.full = Test_Dataset(
            inputs[..., np.newaxis],
            targets[..., np.newaxis],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_inputs[..., np.newaxis],
            None,
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return 400

    def get_split(self, *args):
        return self.train, self.full, self.test

    def len_train(self, test_size):
        return 400 
    


class SyntheticBinary_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "binary"
        self.output_dim = 1

        rng = np.random.default_rng(seed=0)

        def f(x):
            y = np.zeros_like(x)
            mask1 = ((x >= -3) * (x < -1)) |((x >= 1) * (x < 4)) 
            mask2 = ((x >= -1) * (x < -0)) 
            mask3 = ((x >= 0) * (x < 2)) 
            
            y[mask1] = 0.
            y[mask2] = 1.

            r = rng.uniform(-1, 1, size = x[mask3].shape[0])
            y[mask3] = np.sign(r)
            y[y == -1] = 0
            return y
            
 

        # inputs = rng.standard_normal(300)
        inputs = np.linspace(-3.0,4.0, 300)
        targets = f(inputs)
        
        mask = (((inputs > -3) * (inputs < 1)) | ((inputs < 4) * (inputs > 3))).flatten()
                
        self.train = Training_Dataset(
            inputs[mask, np.newaxis],
            targets[mask, np.newaxis],
            normalize_targets=False,
            normalize_inputs=False,
        )

        self.full = Test_Dataset(
            inputs[..., np.newaxis],
            targets[..., np.newaxis],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            inputs[..., np.newaxis],
            targets[..., np.newaxis],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return 300

    def get_split(self, *args):
        return self.train, self.full, self.test

    def len_train(self, test_size):
        return 200 

    
class MNIST_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "multiclass"
        self.classes = 10
        self.output_dim = 10
        train = datasets.MNIST(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        test = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

        train_data = train.data.reshape(60000, -1) / 255.0
        test_data = test.data.reshape(10000, -1) / 255.0

        train_targets = train.targets.reshape(-1, 1)
        test_targets = test.targets.reshape(-1, 1)

        self.train = Training_Dataset(
            train_data.numpy(),
            train_targets.numpy(),
            normalize_targets=False,
            normalize_inputs=False,
        )

        self.full = Test_Dataset(
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
        return self.train, self.full, self.test

    def len_train(self, test_size):
        return 60000


class Banana_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "multiclass"
        self.classes = 2
        self.output_dim = 1
        
        data = pd.read_csv("./data/banana.csv",sep=",").to_numpy()
        inputs, targets = data[:, :-1], data[:, -1]
        
        targets[targets == -1] = 0
        self.inputs = inputs
        self.targets = targets[..., np.newaxis]
        

class Two_Moons(DVIPDataset):
    def __init__(self):
        self.type = "multiclass"
        self.classes = 2
        self.output_dim = 1
        
        from sklearn.datasets import make_moons
        
        inputs, targets = make_moons(1000, random_state = 0, noise = 0.2)
                
        targets[targets == -1] = 0
        self.inputs = inputs
        self.targets = targets[..., np.newaxis]
        
class Three_Blobs(DVIPDataset):
    def __init__(self):
        self.type = "multiclass"
        self.classes = 3
        self.output_dim = 3
        
        from sklearn.datasets import make_blobs
        
        inputs, targets = make_blobs(1000, centers = 3, random_state = 0, cluster_std = 0.2)
                
        self.inputs = inputs
        self.targets = targets[..., np.newaxis]
        
class Spiral3(DVIPDataset):
    def __init__(self):
        self.type = "binary"
        self.classes = 3
        self.output_dim = 3
        
        from numpy import pi
        rng = np.random.default_rng(1234)
        N = 200
        theta = np.sqrt(rng.random(N))*1*pi # np.linspace(0,2*pi,100)

        r_a = 1.5*theta + pi
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        x_a = data_a + rng.random((N,2)) * 3

        r_b = 1.5*theta
        data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
        x_b = data_b + rng.random((N,2)) * 3
        
        r_c = 1.5*theta - pi
        data_c = np.array([np.cos(theta)*r_c, np.sin(theta)*r_c]).T
        x_c = data_c + rng.random((N,2)) * 3

        res_a = np.append(x_a, np.zeros((N,1)), axis=1)
        res_b = np.append(x_b, np.ones((N,1)), axis=1)
        res_c = np.append(x_c, np.ones((N,1))+1, axis=1)
        
        res = np.concatenate([res_a, res_b, res_c], axis=0)
        rng.shuffle(res)
        
        self.inputs = res[:, :-1]
        self.targets = res[:, -1][:, np.newaxis]

class Spiral(DVIPDataset):
    def __init__(self):
        self.type = "binary"
        self.classes = 2
        self.output_dim = 1
        
        from numpy import pi
        rng = np.random.default_rng(1234)
        N = 400
        theta = np.sqrt(rng.random(N))*4*pi # np.linspace(0,2*pi,100)

        r_a = 2*theta + pi
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        x_a = data_a + rng.random((N,2)) * 6

        r_b = -2*theta - pi
        data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
        x_b = data_b + rng.random((N,2)) * 6

        res_a = np.append(x_a, np.zeros((N,1)), axis=1)
        res_b = np.append(x_b, np.ones((N,1)), axis=1)
        
        res = np.append(res_a, res_b, axis=0)
        rng.shuffle(res)
        
        
        self.inputs = res[:, :-1]
        self.targets = res[:, -1][:, np.newaxis]
        
        
        
def get_dataset(dataset_name):
    d = {
        "SPGP": SPGP_Dataset,
        "synthetic": Synthetic_Dataset,
        "synthetic2": Synthetic2_Dataset,
        "syntheticBinary": SyntheticBinary_Dataset,
        "MNIST": MNIST_Dataset,
        "Banana": Banana_Dataset,
        "Moons": Two_Moons,
        "Spiral": Spiral,
        "Spiral3": Spiral3,
        "Blobs": Three_Blobs,
    }

    return d[dataset_name]()
