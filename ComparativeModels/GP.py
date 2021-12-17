import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

sys.path.append(".")

from src.likelihood import Gaussian
from utils.dataset import Synthetic_Dataset, Test_Dataset, Training_Dataset
from utils.metrics import Metrics
from utils.process_flags import manage_experiment_configuration


def RBF(X1, X2, l=1):
    """Computes the RBF kernel of the given input torch tensors.

    Arguments
    ---------

    X1 : torch tensor of shape (..., N, D)
         First input tensor
    X2 : torch tensor of shape (..., N, D)
         Second input tensor

    Returns
    -------
    K : torch tensor of shape (..., N, N)
        xddContains the application of the rbf kernel.
    """
    # Shape (..., N, 1, D)
    X = X1.unsqueeze(-2)
    # Shape (..., 1, N, D)
    Y = X2.unsqueeze(-3)
    # X - Y has shape (..., N, N, D) due to broadcasting
    K = ((X - Y)**2).sum(-1)

    return torch.exp(-K / l**2)


class GP(torch.nn.Module):
    def __init__(
        self,
        inputs,
        targets,
        kernel=RBF,
    ):
        """Encapsulates an exact Gaussian process. The Cholesky decomposition
        of the kernel is used in order to speed up computations.

        Arguments
        ---------
        input_dim : int
                    Dimensionality of the input values `x`.
        output_dim : int
                     Dimensionality of the function output.
        kernel : callable
                 The desired kernel function to use, must accept batched
                 inputs (..., N, D) and compute the kernel matrix with shape
                 (..., N, N).
                 Defaults to RBF.
        device : torch.device
                 The device in which the computations are made.
        dtype : data-type
                The dtype of the layer's computations and weights.
        """
        # Instantiate the kernel
        self.kernel = kernel
        K = self.kernel(inputs, inputs) + 1e-6 * torch.eye(inputs.shape[0])
        self.K_inv = torch.inverse(K)
        self.inputs = inputs
        self.targets = targets

    def predict(self, new_inputs):
        K_s = self.kernel(self.inputs, new_inputs)
        K_ss = self.kernel(new_inputs, new_inputs)

        mu = K_s.T @ self.K_inv @ self.targets
        cov = K_ss - K_s.T @ self.K_inv @ K_s

        return mu, cov


args = manage_experiment_configuration()

torch.manual_seed(2147483647)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
vars(args)["device"] = device

n_splits = 20
results = pd.DataFrame(columns=Metrics().get_dict().keys())

for split in range(n_splits):
    train_indexes, test_indexes = train_test_split(
        np.arange(len(args.dataset)),
        test_size=0.1,
        random_state=2147483647 + split)

    train_dataset = Training_Dataset(
        args.dataset.inputs[train_indexes],
        args.dataset.targets[train_indexes],
        verbose=False,
    )
    test_dataset = Test_Dataset(args.dataset.inputs[test_indexes],
                                args.dataset.targets[test_indexes],
                                train_dataset.inputs_mean,
                                train_dataset.inputs_std)

    gp = GP(torch.tensor(train_dataset.inputs),
            torch.tensor(train_dataset.targets))
    mu, cov = gp.predict(torch.tensor(test_dataset.inputs))

    mu = mu.unsqueeze(
        0) * train_dataset.targets_std + train_dataset.targets_mean
    cov = torch.diagonal(cov).unsqueeze(0).unsqueeze(
        -1).sqrt() * train_dataset.targets_std

    test_metrics = Metrics(len(test_dataset))
    test_metrics.update(torch.tensor(test_dataset.targets), 0, mu, cov)

    results = results.append(test_metrics.get_dict(), ignore_index=True)

results.to_csv(path_or_buf="results/dataset={}_exactGP.csv".format(
    args.dataset_name, str(args.vip_layers[0]), str(args.dropout), args.lr,
    "-".join(str(i) for i in args.bnn_structure)),
               encoding='utf-8')




synthetic_dataset = Synthetic_Dataset()
train_indexes, test_indexes = train_test_split(
        np.arange(len(synthetic_dataset)),
        test_size=0.1,
        random_state=2147483647)

train_dataset = Training_Dataset(
        synthetic_dataset.inputs[train_indexes],
        synthetic_dataset.targets[train_indexes],
        verbose=False,
    )

gp = GP(torch.tensor(train_dataset.inputs),
            torch.tensor(train_dataset.targets))
mu, cov = gp.predict(torch.tensor(train_dataset.inputs))

mu = mu * train_dataset.targets_std + train_dataset.targets_mean
cov = torch.diagonal(cov).sqrt() * train_dataset.targets_std


mu = mu.detach().cpu().numpy().flatten()
cov = cov.detach().cpu().numpy().flatten()
print(cov)
import matplotlib.pyplot as plt

plt.scatter(train_dataset.inputs.flatten(), 
            train_dataset.targets.flatten() * train_dataset.targets_std + train_dataset.targets_mean , 
            color = "blue", 
            s = 2,
            label = "Training Points"
            )
sort = np.argsort(train_dataset.inputs.flatten())
plt.plot(train_dataset.inputs[sort], mu[sort], color = "black")
plt.fill_between(train_dataset.inputs[sort].flatten(), 
                 (mu[sort]-3*cov[sort]),
                 (mu[sort]+3*cov[sort]), 
                 color='b', alpha=.1)
plt.show()