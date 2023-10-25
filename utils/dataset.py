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
from torchvision.transforms.functional import rotate
from torchvision.transforms import GaussianBlur
import torch

class Training_Dataset(Dataset):
    def __init__(
        self,
        inputs,
        targets,
        output_dim,
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
        self.output_dim = output_dim

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
    def __init__(self, inputs, output_dim, targets=None, inputs_mean=0.0, inputs_std=1.0):
        self.inputs = (inputs - inputs_mean) / inputs_std
        self.targets = targets
        self.n_samples = inputs.shape[0]
        self.input_dim = inputs.shape[1]
        if self.targets is not None:
            self.output_dim = output_dim

    def __getitem__(self, index):
        if self.targets is None:
            return self.inputs[index]
        return self.inputs[index], self.targets[index]

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
        mask = ((inputs < 1.5) | (inputs > 3.5)).flatten()
        mask2 = ((inputs >= 1.5) & (inputs <= 3.5)).flatten()
        
        self.train = Training_Dataset(
            inputs[mask, np.newaxis],
            targets[mask, np.newaxis],
            normalize_targets=False,
            normalize_inputs=True,
        )

        self.full = Test_Dataset(
            inputs[:, np.newaxis],
            targets[:, np.newaxis],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            inputs[mask2, np.newaxis],
            targets[mask2, np.newaxis],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return 200

    def get_split(self, *args):
        return self.train, self.full, self.test

    def len_train(self):
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
            self.output_dim,
            normalize_targets=True,
            normalize_inputs=False,
        )

        self.full = Test_Dataset(
            inputs[..., np.newaxis],
            self.output_dim,
            targets[..., np.newaxis],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            inputs[..., np.newaxis],
            self.output_dim,
            targets[..., np.newaxis],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return 300

    def get_split(self, *args):
        return self.train, self.full, self.test

    def len_train(self):
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
            normalize_inputs=True,
        )

        self.full = Test_Dataset(
            test_inputs[..., np.newaxis],
            None,
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

    def len_train(self):
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

    def len_train(self):
        return 200 

    
class MNIST_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "multiclass"
        self.classes = 10
        self.output_dim = 10
        self.corruption_values = [15, 30, 45, 60, 75, 90, 105, 120]

        train = datasets.MNIST(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        test = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

        n_val = 5000
        train_data = train.data.reshape(60000, -1) / 255.0
        self.input_dim = train_data.shape[1]
        test_data = test.data.reshape(10000, -1) / 255.0
        train_targets = train.targets.reshape(-1, 1)
        test_targets = test.targets.reshape(-1, 1)

        self.train = Training_Dataset(
            train_data.numpy()[:-n_val],
            train_targets.numpy()[:-n_val],
            self.output_dim,
            normalize_targets=False,
            normalize_inputs=False,
        )

        self.val = Test_Dataset(
            train_data.numpy()[-n_val:],
            self.output_dim,
            train_targets.numpy()[-n_val:],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_data.numpy(),
            self.output_dim,
            test_targets.numpy(),
            self.train.inputs_mean,
            self.train.inputs_std,
        )

        test2 = datasets.FashionMNIST(
                    root="./data",
                    train=False,
                    download=True,
                    transform=transforms.ToTensor(),
                )

        test2_data = test2.data.reshape(10000, -1) / 255.0
        
        ood_test_data = np.concatenate([test_data, test2_data])
        ood_test_targets = np.concatenate(
            [
                np.zeros(test_data.shape[0]),
                np.ones(test2_data.shape[0])
            ]).reshape(-1, 1)

        self.ood_test = Test_Dataset(
            ood_test_data,
            self.output_dim,
            ood_test_targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )


    def __len__(self):
        return len(self.train) + len(self.val) + len(self.test)

    def get_split(self, *args):
        return self.train, self.val, self.test
    
    def get_corrupted_split(self, angle):
        test = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

        rotated_data = rotate(test.data, angle)
        
        test_data = rotated_data.reshape(10000, -1) / 255.0
        test_targets = test.targets.reshape(-1, 1)

        return Test_Dataset(
            test_data.numpy(),
            self.output_dim,
            test_targets.numpy(),
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def len_train(self):
        return len(self.train)
    
    def get_ood_datasets(self):
        return self.ood_test
    

class FMNIST_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "multiclass"
        self.classes = 10
        self.output_dim = 10
        self.corruption_values = [15, 30, 45, 60, 75, 90, 105, 120]
        train = datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        test = datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        self.input_shape = [1, 28, 28]

        n_val = 5000
        train_data = train.data.reshape(60000, -1) / 255.0

        self.input_dim = train_data.shape[1]
        test_data = test.data.reshape(10000, -1) / 255.0
        train_targets = train.targets.reshape(-1, 1)
        test_targets = test.targets.reshape(-1, 1)

        self.train = Training_Dataset(
            train_data.numpy()[:-n_val],
            train_targets.numpy()[:-n_val],
            self.output_dim,
            normalize_targets=False,
            normalize_inputs=False,
        )

        self.val = Test_Dataset(
            train_data.numpy()[-n_val:],
            self.output_dim,
            train_targets.numpy()[-n_val:],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_data.numpy(),
            self.output_dim,
            test_targets.numpy(),
            self.train.inputs_mean,
            self.train.inputs_std,
        )

        test2 = datasets.MNIST(
                    root="./data",
                    train=False,
                    download=True,
                    transform=transforms.ToTensor(),
                )

        test2_data = test2.data.reshape(10000, -1) / 255.0
        
        ood_test_data = np.concatenate([test_data, test2_data])
        ood_test_targets = np.concatenate(
            [
                np.zeros(test_data.shape[0]),
                np.ones(test2_data.shape[0])
            ]).reshape(-1, 1)

        self.ood_test = Test_Dataset(
            ood_test_data,
            self.output_dim,
            ood_test_targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return len(self.train) + len(self.val) + len(self.test)

    def get_split(self, *args):
        return self.train, self.val, self.test

    def get_corrupted_split(self, angle):
        test = datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

        rotated_data = rotate(test.data, angle)
        
        test_data = rotated_data.reshape(10000, -1) / 255.0
        test_targets = test.targets.reshape(-1, 1)

        return Test_Dataset(
            test_data.numpy(),
            self.output_dim,
            test_targets.numpy(),
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def len_train(self):
        return len(self.train)
    
    def get_ood_datasets(self):
        return self.ood_test
    

class CIFAR10_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "multiclass"
        self.classes = 10
        self.output_dim = 10
        self.corruption_values = [0,1,2,3,4]
        train = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        test = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        self.input_shape = [1, 32, 32]
        
        n_val = 5000
        train_data = train.data.mean(-1).reshape(50000, -1) / 255.0
        test_data = test.data.mean(-1).reshape(10000, -1) / 255.0

        self.input_dim = train_data.shape[1]
    
        train_targets = np.array(train.targets).reshape(-1, 1)
        test_targets = np.array(test.targets).reshape(-1, 1)
        

        self.train = Training_Dataset(
            train_data[:-n_val],
            train_targets[:-n_val],
            self.output_dim,
            normalize_targets=False,
            normalize_inputs=False,
        )

        self.val = Test_Dataset(
            train_data[-n_val:],
            self.output_dim,
            train_targets[-n_val:],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_data,
            self.output_dim,
            test_targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )

        test2 = datasets.SVHN(
            root="./data",
            split="test",
            download=True,
            transform=transforms.ToTensor(),
        )
        test2_data = test2.data.mean(1).reshape(26032, -1) / 255.0
        ood_test_data = np.concatenate([test_data, test2_data])
        ood_test_targets = np.concatenate(
            [
                np.zeros(test_data.shape[0]),
                np.ones(test2_data.shape[0])
            ]).reshape(-1, 1)

        self.ood_test = Test_Dataset(
            ood_test_data,
            self.output_dim,
            ood_test_targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return len(self.train) + len(self.val) + len(self.test)

    def get_split(self, *args):
        return self.train, self.val, self.test
    
    def get_corrupted_split(self, noise):
        map = np.lib.format.open_memmap("./data/gaussian_blur.npy", mode='r+')
        subset = map[noise*10000:(noise+1)*10000]
        
        test_data = subset.mean(-1).reshape(10000, -1) / 255.0

        test_targets = np.array(self.test.targets).reshape(-1, 1)

        return Test_Dataset(
            test_data,
            self.output_dim,
            test_targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def len_train(self):
        return len(self.train)

    def get_ood_datasets(self):
        return self.ood_test
    

class CIFAR10_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "multiclass"
        self.classes = 10
        self.output_dim = 10
        self.corruption_values = [0,1,2,3,4]
        train = datasets.CIFAR10(
            root="./data", train=True, download=True
        )
        test = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
        )
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        train_data = (train.data/255.0 - mean)/std
        test_data = (test.data/255.0 - mean)/std

        self.input_shape = train_data.shape[1:]
        n_val = 5000
        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))

        self.input_dim = train_data.shape[1]
    
        train_targets = np.array(train.targets).reshape(-1, 1)
        test_targets = np.array(test.targets).reshape(-1, 1)
        

        self.train = Training_Dataset(
            train_data[:-n_val],
            train_targets[:-n_val],
            self.output_dim,
            normalize_targets=False,
            normalize_inputs=False,
        )

        self.val = Test_Dataset(
            train_data[-n_val:],
            self.output_dim,
            train_targets[-n_val:],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_data,
            self.output_dim,
            test_targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )

        test2 = datasets.SVHN(
            root="./data",
            split="test",
            download=True,
            transform=transforms.ToTensor(),
        )
        mean = np.expand_dims(np.expand_dims(mean, -1),-1)
        std = np.expand_dims(np.expand_dims(std, -1),-1)

        test2_data = (test2.data/255.0 - mean)/std
        ood_test_data = np.concatenate([test_data, test2_data])
        ood_test_targets = np.concatenate(
            [
                np.zeros(test_data.shape[0]),
                np.ones(test2_data.shape[0])
            ]).reshape(-1, 1)

        self.ood_test = Test_Dataset(
            ood_test_data,
            self.output_dim,
            ood_test_targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return len(self.train) + len(self.val) + len(self.test)

    def get_split(self, *args):
        return self.train, self.val, self.test
    
    def get_corrupted_split(self, noise):
        map = np.lib.format.open_memmap("./data/gaussian_blur.npy", mode='r+')
        subset = map[noise*10000:(noise+1)*10000]
        
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        test_data = (subset/255.0 - mean)/std
        test_data = np.transpose(test_data, (0, 3, 1, 2))

        test_targets = np.array(self.test.targets)

        return Test_Dataset(
            test_data,
            self.output_dim,
            test_targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def len_train(self):
        return len(self.train)

    def get_ood_datasets(self):
        return self.ood_test
    

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
        

# class Two_Moons(DVIPDataset):
#     def __init__(self):
#         self.type = "multiclass"
#         self.classes = 2
#         self.output_dim = 1
        
#         from sklearn.datasets import make_moons
        
#         inputs, targets = make_moons(1000, random_state = 0, noise = 0.2)
                
#         targets[targets == -1] = 0
#         self.inputs = inputs
#         self.targets = targets[..., np.newaxis]
        
# class Three_Blobs(DVIPDataset):
#     def __init__(self):
#         self.type = "multiclass"
#         self.classes = 3
#         self.output_dim = 3
        
#         from sklearn.datasets import make_blobs
        
#         inputs, targets = make_blobs(1000, centers = 3, random_state = 0, cluster_std = 0.2)
                
#         self.inputs = inputs
#         self.targets = targets[..., np.newaxis]
        
# class Spiral3(DVIPDataset):
#     def __init__(self):
#         self.type = "binary"
#         self.classes = 3
#         self.output_dim = 3
        
#         from numpy import pi
#         rng = np.random.default_rng(1234)
#         N = 200
#         theta = np.sqrt(rng.random(N))*1*pi # np.linspace(0,2*pi,100)

#         r_a = 1.5*theta + pi
#         data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
#         x_a = data_a + rng.random((N,2)) * 3

#         r_b = 1.5*theta
#         data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
#         x_b = data_b + rng.random((N,2)) * 3
        
#         r_c = 1.5*theta - pi
#         data_c = np.array([np.cos(theta)*r_c, np.sin(theta)*r_c]).T
#         x_c = data_c + rng.random((N,2)) * 3

#         res_a = np.append(x_a, np.zeros((N,1)), axis=1)
#         res_b = np.append(x_b, np.ones((N,1)), axis=1)
#         res_c = np.append(x_c, np.ones((N,1))+1, axis=1)
        
#         res = np.concatenate([res_a, res_b, res_c], axis=0)
#         rng.shuffle(res)
        
#         self.inputs = res[:, :-1]
#         self.targets = res[:, -1][:, np.newaxis]

# class Spiral(DVIPDataset):
#     def __init__(self):
#         self.type = "binary"
#         self.classes = 2
#         self.output_dim = 1
        
#         from numpy import pi
#         rng = np.random.default_rng(1234)
#         N = 400
#         theta = np.sqrt(rng.random(N))*4*pi # np.linspace(0,2*pi,100)

#         r_a = 2*theta + pi
#         data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
#         x_a = data_a + rng.random((N,2)) * 6

#         r_b = -2*theta - pi
#         data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
#         x_b = data_b + rng.random((N,2)) * 6

#         res_a = np.append(x_a, np.zeros((N,1)), axis=1)
#         res_b = np.append(x_b, np.ones((N,1)), axis=1)
        
#         res = np.append(res_a, res_b, axis=0)
#         rng.shuffle(res)
        
        
#         self.inputs = res[:, :-1]
#         self.targets = res[:, -1][:, np.newaxis]



class Airline_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1

        data = pd.read_csv("./data/airline.csv")
        # Convert time of day from hhmm to minutes since midnight
        data.ArrTime = 60 * np.floor(data.ArrTime / 100) + np.mod(data.ArrTime, 100)
        data.DepTime = 60 * np.floor(data.DepTime / 100) + np.mod(data.DepTime, 100)

        # Pick out the data
        Y = data["ArrDelay"].values[:800000].reshape(-1, 1)
        names = [
            "Month",
            "DayofMonth",
            "DayOfWeek",
            "plane_age",
            "AirTime",
            "Distance",
            "ArrTime",
            "DepTime",
        ]
        X = data[names].values[:800000]

        self.n_train = 600000
        self.n_val = 100000
        self.train = Training_Dataset(X[: self.n_train], Y[: self.n_train],
            self.output_dim)
        self.input_dim = X.shape[1]

        self.train_test = Test_Dataset(
            X[:self.n_train],
            self.output_dim,
            Y[:self.n_train],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        
        self.val = Test_Dataset(
            X[self.n_train:self.n_train + self.n_val],
            self.output_dim,
            Y[self.n_train:self.n_train + self.n_val],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            X[self.n_train + self.n_val:],
            self.output_dim,
            Y[self.n_train + self.n_val:],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return self.len_data

    def len_train(self):
        return self.n_train

    def get_split(self, split, *args):
        return self.train, self.val, self.test


class Year_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        try:
            data = pd.read_csv(
                "./data/YearPredictionMSD.txt", header=None, delimiter=","
            ).values
        except:
            url = "{}{}".format(uci_base, "00203/YearPredictionMSD.txt.zip")
            with urlopen(url) as zipresp:
                with ZipFile(BytesIO(zipresp.read())) as zfile:
                    zfile.extractall("./data/")

            data = pd.read_csv(
                "./data/YearPredictionMSD.txt", header=None, delimiter=","
            ).values

        self.len_data = data.shape[0]

        X = data[:, 1:]
        self.input_dim = X.shape[1]
        Y = data[:, 0].reshape(-1, 1)

        self.n_train = 400000
        self.n_val = 63715
        self.train = Training_Dataset(X[: self.n_train], Y[: self.n_train],
            self.output_dim)

        self.train_test = Test_Dataset(
            X[:self.n_train],
            self.output_dim,
            Y[:self.n_train],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

        self.val = Test_Dataset(
            X[self.n_train:self.n_train + self.n_val],
            self.output_dim,
            Y[self.n_train:self.n_train + self.n_val],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            X[self.n_train + self.n_val:],
            self.output_dim,
            Y[self.n_train + self.n_val:],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return self.len_data

    def len_train(self):
        return self.n_train

    def get_split(self, split, *args):
        return self.train, self.val, self.test
        

class Taxi_Dataset(DVIPDataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1

        if os.path.exists("data/taxi.csv"):
            print("Taxi csv file found.")
            data = pd.read_csv("data/taxi.csv")
        elif os.path.exists("data/taxi.zip"):
            print("Taxi zip file found.")
            data = pd.read_csv("data/taxi.zip", compression="zip", dtype=object)
        else:
            print("Downloading Taxi Dataset...")
            url = (
                "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
            )
            data = pd.read_parquet(url)
            data.to_csv("data/taxi.csv")

        data["tpep_pickup_datetime"] = pd.to_datetime(data["tpep_pickup_datetime"])
        data["tpep_dropoff_datetime"] = pd.to_datetime(data["tpep_dropoff_datetime"])
        data["day_of_week"] = data["tpep_pickup_datetime"].dt.dayofweek
        data["day_of_month"] = data["tpep_pickup_datetime"].dt.day
        data["month"] = data["tpep_pickup_datetime"].dt.month
        print(data["tpep_pickup_datetime"].dt)
        data["time_of_day"] = (
            data["tpep_pickup_datetime"] - data["tpep_pickup_datetime"].dt.normalize()
        ) / pd.Timedelta(seconds=1)
        data["trip_duration"] = (
            data["tpep_dropoff_datetime"] - data["tpep_pickup_datetime"]
        ).dt.total_seconds()
        data = data[
            [
                "time_of_day",
                "day_of_week",
                "day_of_month",
                "month",
                "PULocationID",
                "DOLocationID",
                "trip_distance",
                "trip_duration",
                "total_amount"
            ]
        ]
        data = data[data["trip_duration"] >= 10]
        data = data[data["trip_duration"] <= 5 * 3600]
        data = data.astype(float)
        data = data.values
        X = data[:, :-1]
        Y = data[:, -1][:, np.newaxis]
        self.input_dim = X.shape[1]

        self.n_train = int(X.shape[0] * 0.8)
        self.n_val = int(X.shape[0] * 0.1)
        self.train = Training_Dataset(X[: self.n_train], Y[: self.n_train],
            self.output_dim)

        self.train_test = Test_Dataset(
            X[:self.n_train],
            self.output_dim,
            Y[:self.n_train],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

        self.val = Test_Dataset(
            X[self.n_train:self.n_train + self.n_val],
            self.output_dim,
            Y[self.n_train:self.n_train + self.n_val],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            X[self.n_train + self.n_val:],
            self.output_dim,
            Y[self.n_train + self.n_val:],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return self.len_data

    def len_train(self):
        return self.n_train

    def get_split(self, split, *args):
        return self.train, self.val, self.test

        
def get_dataset(dataset_name):
    d = {
        "SPGP": SPGP_Dataset,
        "synthetic": Synthetic_Dataset,
        "synthetic2": Synthetic2_Dataset,
        "syntheticBinary": SyntheticBinary_Dataset,
        "MNIST": MNIST_Dataset,
        "FMNIST": FMNIST_Dataset,
        "CIFAR10": CIFAR10_Dataset,
        #"CIFAR10o": CIFAR10o_Dataset,
        "Banana": Banana_Dataset,
        "Airline": Airline_Dataset,
        "Year": Year_Dataset,
        "Taxi": Taxi_Dataset
    }

    return d[dataset_name]()
